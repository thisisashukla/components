import math
import warnings
import pandas as pd
from sklearn.linear_model import BayesianRidge
from statsmodels.regression import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
from statsmodels.regression.quantile_regression import QuantReg
from statsmodels.tools.sm_exceptions import IterationLimitWarning

warnings.simplefilter("ignore", IterationLimitWarning)

from models.coefficient import CoefficientSklearnModel, CoefficientStatsModel


class BayesianRidgeModel(CoefficientSklearnModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        name: str = None,
        date_column: str = "date",
        alpha_1: float = 0.1,
        alpha_2: float = 0.1,
        lambda_1: float = 0.1,
        lambda_2: float = 0.1,
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=BayesianRidge,
            model_params={},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            date_column=date_column,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )


class PoissonModel(CoefficientSklearnModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        date_column: str = "date",
        scaler=StandardScaler,
    ):
        super().__init__(
            data={k: v.copy(deep=True) for k, v in data.items()},
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=PoissonRegressor,
            model_params={},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
            scaler=scaler,
            column_scalers={},
        )

    def get_dataset(
        self, df: pd.DataFrame, target: bool = False, features: list[str] = None
    ):
        if features is None:
            features = self.final_features

        for col in features:
            if not (col in self.column_scalers.keys()):
                self.column_scalers[col] = self.scaler()
                self.column_scalers[col].fit(
                    self.data["train"][col].values.reshape(-1, 1)
                )

            df[col] = self.column_scalers[col].transform(df[col].values.reshape(-1, 1))

        if target:
            return df[features], df[self.target].astype(int)
        else:
            return df[features], None


class QuantileModel(CoefficientStatsModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        date_column: str = "date",
        quantile: int = 0.5,
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=QuantReg,
            model_params={},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
            quantile=quantile,
        )

        self._model_type = f"{self.model_type}_{quantile}"


class OLSModel(CoefficientStatsModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        date_column: str = "date",
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=linear_model.OLS,
            model_params={"missing": "raise"},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
        )


class LassoModel(CoefficientStatsModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        alpha: float = 0.6,
        date_column: str = "date",
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=linear_model.OLS,
            model_params={"missing": "raise"},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params={
                "method": "elastic_net",
                "alpha": alpha,
                "L1_wt": 1,
            },
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
            alpha=alpha,
        )


class RidgeModel(CoefficientStatsModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        alpha: float = 0.1,
        date_column: str = "date",
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=linear_model.OLS,
            model_params={"missing": "raise"},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params={
                "method": "elastic_net",
                "alpha": alpha,
                "L1_wt": 0,
            },
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
            alpha=alpha,
        )


class ElasticModel(CoefficientStatsModel):

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        force_features: bool = False,
        keep_vars: list[str] = [],
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        alpha: float = 0.1,
        L1_wt: float = 0.5,
        date_column: str = "date",
    ):
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=linear_model.OLS,
            model_params={"missing": "raise"},
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params={
                "method": "elastic_net",
                "alpha": alpha,
                "L1_wt": L1_wt,
            },
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            date_column=date_column,
            alpha=alpha,
            L1_wt=L1_wt,
        )


def generic_coefficient_tuning_objective(**kwargs):

    model = kwargs["model"](
        **{k: v for k, v in kwargs.items() if not (k in ["trial", "metric", "model"])}
    )
    model.initialize()
    model.train()
    preds = model.predict()

    retrain = False
    drop_vars = []
    profile = model.profile()

    for var, sign in [("retail_price_per_unit", 1), ("retail_discount_per_unit", -1)]:
        if var in model.final_features:
            if profile[profile["feature"] == var]["coefficient"].values[0] * sign > 0:
                retrain = True
                drop_vars.append(var)

    if retrain == True and len(drop_vars) > 0:
        features = [f for f in model.final_features if not (f in drop_vars)]

        model = kwargs["model"](
            features=features,
            **{
                k: v
                for k, v in kwargs.items()
                if not (k in ["trial", "metric", "model", "features"])
            },
        )
        model.initialize()
        model.train()
        preds = model.predict()

    for k in preds.keys():
        preds[k] = preds[k].merge(
            kwargs["data"][k][model.grain_columns + [model.date_column, model.target]]
        )
        preds[k] = preds[k].rename(columns={model.target: "true"})

    kwargs["metric"].fit(preds)
    model_selection = (
        kwargs["metric"]
        .predmetricdf.groupby(
            kwargs["metric"].selection_grain + [kwargs["metric"].model_name_identifier]
        )
        .apply(kwargs["metric"].final_metric_method)
        .reset_index()
        .rename(columns={0: "final_metric"})
    )

    kwargs["trial"].set_user_attr("final_features", list(set(model.final_features)))
    kwargs["trial"].set_user_attr("best_model", model)
    kwargs["trial"].set_user_attr("prediction", preds)
    kwargs["trial"].set_user_attr("metric", kwargs["metric"])
    fit_params = {}
    if "alpha" in kwargs.keys():
        kwargs["trial"].set_user_attr("alpha", kwargs["alpha"])
        fit_params = {
            "method": "elastic_net",
            "alpha": kwargs["alpha"],
            "L1_wt": model.fit_params["L1_wt"],
        }
    if "L1_wt" in kwargs.keys():
        kwargs["trial"].set_user_attr("L1_wt", kwargs["L1_wt"])
        fit_params = {
            "method": "elastic_net",
            "alpha": kwargs["alpha"],
            "L1_wt": kwargs["L1_wt"],
        }
    kwargs["trial"].set_user_attr("fit_params", fit_params)

    return (
        model_selection["final_metric"].values[0],
        model.final_features,
        model,
        preds,
        kwargs["metric"],
        kwargs["trial"],
    )


def lasso_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10.0)
    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_coefficient_tuning_objective(trial=trial, alpha=alpha, **kwargs)

    return ms


def ridge_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10)
    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_coefficient_tuning_objective(trial=trial, alpha=alpha, **kwargs)

    return ms


def elastic_objective(trial, **kwargs):

    alpha = trial.suggest_float("alpha", 0.001, 10.0)
    L1_wt = trial.suggest_float("L1_wt", 0.0, 1.0)

    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_coefficient_tuning_objective(
        trial=trial, alpha=alpha, L1_wt=L1_wt, **kwargs
    )

    return ms


def poisson_objective(trial, **kwargs):

    alpha = trial.suggest_float(
        "alpha", [0.001, 0.01, 0.025, 0.05, 0.1, 0.5, 1, 2, 4, 6, 8, 10]
    )
    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_coefficient_tuning_objective(trial=trial, alpha=alpha, **kwargs)

    return ms


def bayesian_objective(trial, **kwargs):

    alpha_1 = trial.suggest_float(
        "alpha_1", math.pow(10, -7), math.pow(10, -5), log=True
    )
    alpha_2 = trial.suggest_float(
        "alpha_2", math.pow(10, -7), math.pow(10, -5), log=True
    )
    lambda_1 = trial.suggest_float(
        "lambda_1", math.pow(10, -7), math.pow(10, -5), log=True
    )
    lambda_2 = trial.suggest_float(
        "lambda_2", math.pow(10, -7), math.pow(10, -5), log=True
    )
    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_coefficient_tuning_objective(
        trial=trial,
        alpha_1=alpha_1,
        alpha_2=alpha_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        **kwargs,
    )

    return ms
