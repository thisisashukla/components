import numpy as np
import pandas as pd
from typing import Optional, List, Set, Union
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin


class BaseVariableSelector(BaseEstimator, TransformerMixin):
    def __init__(self, target: str, keep_vars: Optional[List[str]] = None):
        self.target = target
        self.keep_vars = keep_vars if keep_vars is not None else []
        self.selected_features: Set[str] = set()

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseVariableSelector":

        raise NotImplementedError("Subclasses should implement this!")

    def transform(self, X: pd.DataFrame) -> list[str]:

        valid_features = [
            c for c in X.columns if c in self.selected_features or c in self.keep_vars
        ]

        return valid_features

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:

        return self.fit(X, y).transform(X)


class PairwiseCorrelationSelector(BaseVariableSelector):
    def __init__(self, target: str, keep_vars: list[str], threshold: float = 0.5):
        super().__init__(target, keep_vars)
        self.threshold = threshold

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PairwiseCorrelationSelector":

        df_corr = X.corr()
        feature_corr = df_corr.drop(columns=[self.target], index=[self.target])

        # Extract pairwise correlations below the diagonal to avoid redundancy
        pairwise_corr = (
            feature_corr.where(np.tril(np.ones(feature_corr.shape), k=-1).astype(bool))
            .stack()
            .reset_index(name="pair_corr")
            .dropna()
            .rename(columns={"level_0": "ft_1", "level_1": "ft_2"})
        )

        # Merge correlations with target and filter by threshold
        pairwise_corr = pairwise_corr.merge(
            df_corr[self.target]
            .reset_index()
            .rename(columns={"index": "ft_1", self.target: "target_corr_ft_1"}),
            on="ft_1",
        ).merge(
            df_corr[self.target]
            .reset_index()
            .rename(columns={"index": "ft_2", self.target: "target_corr_ft_2"}),
            on="ft_2",
        )

        relevant_pairs = pairwise_corr[
            pairwise_corr["pair_corr"].abs() >= self.threshold
        ]
        relevant_pairs["rejected_feature"] = np.where(
            relevant_pairs[["target_corr_ft_1", "target_corr_ft_2"]]
            .abs()
            .idxmin(axis=1)
            == "target_corr_ft_1",
            relevant_pairs["ft_1"],
            relevant_pairs["ft_2"],
        )

        self.selected_features = (
            set(X.columns)
            - set(relevant_pairs["rejected_feature"].unique())
            - set(self.keep_vars)
        )
        return self


class RandomCutoffSelector(BaseVariableSelector):
    def __init__(self, target: str, keep_vars: list[str], num_random_ft: int = 4):
        super().__init__(target, keep_vars)
        self.num_random_ft = num_random_ft

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "RandomCutoffSelector":

        df = X.copy()
        lo, hi = df[self.target].min(), df[self.target].max()
        rng = np.random.default_rng()

        # Generate random features
        for i in range(self.num_random_ft):
            df[f"random_ft_{i}"] = rng.uniform(lo, hi, df.shape[0])

        # Fit the Gradient Boosting Regressor and determine feature importances
        gbm = GradientBoostingRegressor(
            n_estimators=5000,
            max_features=df.shape[1] - 1,
            max_depth=10,
            min_samples_leaf=1,
            min_samples_split=2,
            subsample=0.6,
        )
        gbm.fit(df.drop(columns=[self.target]), df[self.target])

        imp_df = pd.DataFrame(
            {"ft": gbm.feature_names_in_, "importance": gbm.feature_importances_}
        ).sort_values("importance", ascending=False)

        # Determine the cutoff based on the maximum importance of random features
        cutoff_imp = imp_df[imp_df["ft"].str.startswith("random_ft")][
            "importance"
        ].max()

        # Select features that have higher importance than the cutoff or are in keep_vars
        self.selected_features = set(
            imp_df[
                (imp_df["importance"] > cutoff_imp)
                | (imp_df["ft"].isin(self.keep_vars))
            ]["ft"]
        )
        return self


class VIFCutoffSelector(BaseVariableSelector):
    def __init__(self, target: str, keep_vars: list[str], r2_cutoff):
        super().__init__(target, keep_vars)  # Target is not used in VIF
        self.r2_cutoff = r2_cutoff

    def fit(self, X, y=None):

        self.vif_df = pd.DataFrame(index=X.columns, columns=["r2"])
        for col_idx, col_name in enumerate(X.columns):
            r2_value, _ = self._calculate_r2(X, col_idx)
            self.vif_df.at[col_name, "r2"] = r2_value

        # Select features based on the R² cutoff and keep_vars
        self.selected_features = set(
            self.vif_df.index[self.vif_df["r2"] <= self.r2_cutoff] | set(self.keep_vars)
        )
        return self

    def _calculate_r2(self, exog, exog_idx):

        t = exog.columns[exog_idx]
        model = LinearRegression()
        X_train = exog.drop(columns=[t])
        y_train = exog[t]
        model.fit(X_train, y_train)
        return model.score(X_train, y_train), model


class PValueBackwardEliminationSelector(BaseVariableSelector):
    def __init__(
        self,
        target: str,
        keep_vars: list[str],
        model_class,
        model_package: str,
        min_pvalue_threshold: float = 0.1,
        model_params: dict = {},
        fit_params: dict = None,
        fit_method: str = None,
    ):
        super().__init__(target, keep_vars)
        self.model_class = model_class
        self.model_package = model_package
        self.min_pvalue_threshold = min_pvalue_threshold
        self.model_params = model_params
        self.fit_params = fit_params
        self.fit_method = fit_method

    def get_pvalues(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray, list],
        features: list[str],
    ) -> pd.DataFrame:

        if self.model_package == "statsmodels":
            model = self.model_class(y, X[features], **self.model_params)

            if self.fit_method is None or self.fit_method == "fit":
                result = model.fit(**self.fit_params)
            else:
                result_fit_regularised = model.fit_regularized(**self.fit_params)
                result = model.fit(params=result_fit_regularised.params)

            pvalue_df = result.pvalues.reset_index().rename(
                columns={"index": "feature", 0: "pvalue"}
            )

        return pvalue_df

    def fit(self, X, y):

        features = X.columns
        step_count = 0

        df = X.copy()

        while True:
            pvalue_df = self.get_pvalues(df, y, features)
            significant_features = pvalue_df[
                pvalue_df["pvalue"] <= self.min_pvalue_threshold
            ]
            if significant_features.shape[0] == len(features):
                break
            elif significant_features.shape[0] == 0:
                break
            else:
                features = significant_features["feature"].values
                step_count += 1

        self.selected_features = features

        return self
