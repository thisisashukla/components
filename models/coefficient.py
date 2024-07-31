import pandas as pd
import statsmodels.api as sm
from models.base import BaseModel
from abc import ABCMeta, abstractmethod
from sklearn.linear_model import BayesianRidge
from statsmodels.regression import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from statsmodels.regression.quantile_regression import QuantReg

from common import utils
from selection import variable_selection


class CoefficientModel(BaseModel):
    def __init__(
        self,
        model_class: ABCMeta,
        model_params: dict,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            data, features, target, grain_columns, name, log_function, **kwargs
        )

        self.model_class = model_class
        self.model_params = model_params
        self.force_features = force_features
        self.keep_vars = keep_vars
        self.fit_params = fit_params
        self.min_pvalue_threshold = min_pvalue_threshold
        self.add_constant = add_constant

        self.dataset = {}
        for k, df in self.data.items():
            self.dataset[k] = self.get_dataset(df, k != "test")

    def get_dataset(self, df: pd.DataFrame, target=False) -> tuple:

        X = df[self.features]
        if self.add_constant:
            X = sm.add_constant(X, has_constant="add")

        if target:
            return X, df[self.config.TARGET]
        else:
            return X, None

    @abstractmethod
    def _initialize(self) -> None:

        pass

    @abstractmethod
    def _read_model(self, path: str) -> None:

        pass

    @abstractmethod
    def _train(self, df: pd.DataFrame) -> None:

        pass

    @abstractmethod
    def _predict(self, data: dict) -> pd.DataFrame:

        pass


class CoefficientStatsModel(CoefficientModel):
    def __init__(
        self,
        model_class: ABCMeta,
        model_params: dict,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            model_class,
            model_params,
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            fit_params,
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

    def _initialize(self) -> None:

        if self.force_features:
            self.final_features = self.features
        else:
            self.final_features = variable_selection.pvalue_based_backward_elimination(
                self.min_pvalue_threshold,
                self.model_class,
                self.dataset["train"][0],
                self.dataset["train"][1],
                self.keep_vars,
                self.model_params,
                self.fit_params,
                self.log_function,
                "fit_regularized" if self.fit_params else "fit",
            )

        assert all([x in self.final_features for x in self.keep_vars])

        X = self.dataset["train"][0][self.final_features]

        model = self.model_class(self.dataset["train"][1], X, **self.model_params)

        return model

    def _read_model(self, path: str) -> None:

        model_object = utils.read_pickle(path)
        model = model_object["model"]
        self.results = model_object["results"]
        self.fit_regularized = model_object["fit_regularized"]

        return model

    def _train(self, df: pd.DataFrame) -> None:

        X, y = self.get_dataset(df, True)

        self.model = self.model_class(y, X, **self.model_params)

        if self.fit_params:
            results = self.model.fit_regularized(**self.fit_params)
            fit_regularized = self.model.fit(params=fit_regularized.params)
        else:
            results = self.model.fit()
            fit_regularized = None

        self.results = results
        self.fit_regularized = fit_regularized

    def _predict(self, data: dict) -> pd.DataFrame:

        dataset = {}
        for k, df in data.items():
            data_x, _ = self.get_dataset(df)
            dataset[k] = data_x

        predictions = {}
        for k, df in dataset.items():
            if self.fit_params:
                pred = self.model.predict(self.fit_regularized.params, df)
            else:
                pred = self.model.predict(self.result.params, df)

            preddf = df[self.grain_columns]
            preddf["prediction"] = pred
            preddf["model_type"] = self.model_class.__name__
            predictions[k] = preddf

        return predictions

    def _save_model(self, path):

        utils.write_pickle(
            {
                "model": self.model,
                "results": self.results,
                "results_regularised": self.results_regularised,
            },
            path,
        )

    def profile(self):

        profile_df = self.results.pvalues.reset_index().rename(
            columns={"index": "feature", 0: "p-value"}
        )
        profile_df["coefficient"] = self.results.params
        profile_df["model_type"] = self.model_class.__name__
        profile_df["model_class"] = self.name

        return profile_df


class CoefficientSklearnModel(CoefficientModel):
    def __init__(
        self,
        model_class: ABCMeta,
        model_params: dict,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = False,
        **kwargs,
    ):

        super().__init__(
            model_class,
            model_params,
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            fit_params,
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

    def _initialize(self) -> None:

        model = self.model_class(**self.model_params)

        return model

    def _train(self, df: pd.DataFrame) -> None:

        X, y = self.get_dataset(df, True)

        self.model.fit(X, y)

    def _predict(self, data: dict) -> pd.DataFrame:

        dataset = {}
        for k, df in data.items():
            data_x, _ = self.get_dataset(df)
            dataset[k] = data_x

        predictions = {}
        for k, df in dataset.items():
            pred = self.model.predict(df)
            preddf = df[self.grain_columns]
            preddf["prediction"] = pred
            preddf["model_type"] = self.model_class.__name__
            predictions[k] = preddf

        return predictions

    def profile_model(self):

        profile_df = pd.DataFrame(
            {
                "feature": self.model.feature_names_in_,
                "coefficient": self.model.coef_,
                "model_type": self.model_class.__name__,
                "model_name": [self.name] * len(self.model.coef_),
            }
        )

        return profile_df


class BayesianRidgeModel(CoefficientSklearnModel):
    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = False,
        **kwargs,
    ):

        super().__init__(
            BayesianRidge,
            {},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            fit_params,
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )


class PossionModel(CoefficientSklearnModel):
    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = False,
        **kwargs,
    ):

        super().__init__(
            PoissonRegressor,
            {},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            fit_params,
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

        self.dataset = {}
        self.scalers = {}

        for col in self.features:
            self.scalers[col] = StandardScaler()
            self.scalers[col].fit(self.data["train"][col].values.reshape(-1, 1))

        for k, df in self.data.items():
            self.dataset[k] = self.get_data(df.copy())

    def get_dataset(self, df: pd.DataFrame, target: bool = False):

        for col in self.features:
            df[col] = self.scalers[col].transform(df[col].values.reshape(-1, 1))

        if target:
            return df[self.features], df[self.target].astype(int)
        else:
            return df[self.features], None


class QuantileModel(CoefficientStatsModel):
    def __init__(
        self,
        quantile: float,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            QuantReg,
            {},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            {"q": quantile, **fit_params},
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

        self.quantile = quantile


class OLSModel(CoefficientStatsModel):
    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            linear_model.OLS,
            {"missing": "raise"},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            {},
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )


class LassoModel(CoefficientStatsModel):
    def __init__(
        self,
        alpha: float,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            linear_model.OLS,
            {"missing": "raise"},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            {"alpha": alpha, "l1_wt": 1},
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

        self.alpha = alpha


class RidgeModel(CoefficientStatsModel):
    def __init__(
        self,
        alpha: float,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            linear_model.OLS,
            {"missing": "raise"},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            {"alpha": alpha, "l1_wt": 0},
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

        self.alpha = alpha


class ElasticModel(CoefficientStatsModel):
    def __init__(
        self,
        alpha: float,
        l1_wt: float,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        **kwargs,
    ):

        super().__init__(
            linear_model.OLS,
            {"missing": "raise"},
            data,
            features,
            target,
            grain_columns,
            name,
            log_function,
            force_features,
            keep_vars,
            {"alpha": alpha, "l1_wt": l1_wt},
            min_pvalue_threshold,
            add_constant,
            **kwargs,
        )

        self.alpha = alpha
        self.l1_wt = l1_wt


def generic_coefficient_tuning_objective(**kwargs):

    model = kwargs["model"](**kwargs)
    model.initialize()
    model.train()
    preds = model.predict()

    retrain = False
    drop_vars = []
    profile = model.profile_model()
    for var, sign in [("retail_price_per_unit", 1), ("retail_discount_per_unit", -1)]:
        if var in model.final_feature_set:
            if profile[profile["feature"] == var]["coefficient"].values[0] * sign > 0:
                kwargs["log_msg"](f"Dropping {var} and retraining model")
                retrain = True
                drop_vars.append(var)

    if retrain == True and len(drop_vars) > 0:
        features = [f for f in kwargs["features"] if not (f in drop_vars)]

        model = kwargs["model"](features=features, **kwargs)
        model.initialize()
        model.train()
        preds = model.predict()

    kwargs["metric"].fit(preds)
    model_selection = (
        kwargs["metric"]
        .predmetricdf.groupby(
            kwargs["metric"].selection_grain + [kwargs["metric"].model_name_indentifier]
        )
        .apply(kwargs["metric"].final_metric_method)
        .reset_index()
        .rename(columns={0: "final_metric"})
    )

    return model_selection, model.features


def lasso_model_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10.0)
    ms, final_feature_set = generic_coefficient_tuning_objective(
        alpha=alpha, l1_wt=1, **kwargs
    )

    trial.set_user_attr("final_feature_set", list(set(final_feature_set)))

    return ms


def ridge_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10)
    ms, final_feature_set = generic_coefficient_tuning_objective(
        alpha=alpha, l1_wt=0, **kwargs
    )

    trial.set_user_attr("final_feature_set", list(set(final_feature_set)))

    return ms


def elastic_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10.0)
    L1_wt = trial.suggest_float("L1_wt", 0.0, 1.0)
    ms, final_feature_set = generic_coefficient_tuning_objective(
        alpha=alpha, l1_wt=L1_wt, **kwargs
    )

    trial.set_user_attr("final_feature_set", list(set(final_feature_set)))

    return ms
