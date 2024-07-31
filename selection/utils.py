import time
import numpy as np
import pandas as pd
from . import model_selection, tuning
from metrics import utils as metric_utils
from forecasting.models import regression, tree, quantilereg, bayesianreg


def weighted_selection_metric(df: pd.DataFrame):

    val_mean = df.loc[
        (df["data"] == "validation") & (df["sample"] == False), ("agg_metric", "mean")
    ].values[0]
    train = df.loc[
        (df["data"] == "train") & (df["sample"] == False), ("agg_metric", "mean")
    ].values[0]

    train_val = abs(train - val_mean)

    final = val_mean + 0.5 * train_val
    return final


def weighted_selection_metric_with_mean_std(df: pd.DataFrame):
    val_mean = df.loc[
        (df["data"] == "validation") & (df["sample"] == True), ("agg_metric", "mean")
    ].values[0]
    val_std = df.loc[
        (df["data"] == "validation") & (df["sample"] == True), ("agg_metric", "std")
    ].values[0]
    train = df.loc[
        (df["data"] == "train") & (df["sample"] == False), ("agg_metric", "mean")
    ].values[0]

    train_val = abs(train - val_mean)

    final = val_mean + 0.5 * train_val + 0.3 * val_std
    return final


def get_default_model_selection_object(grain, op):

    if op == "tuning":
        return model_selection.BestModelSelection(
            selection_grain=grain,
            metrics=metric_utils.model_comparison_metrics,
            metric_aggregator=lambda x: np.linalg.norm(x, 2),
            model_name_indentifier="model_type",
            final_metric_method=weighted_selection_metric,
            metric_weights=[0, 1, 0, 0],
            scale_metrics=False,
        )
    else:
        return model_selection.BestModelSelection(
            selection_grain=grain,
            metrics=metric_utils.model_comparison_metrics,
            metric_aggregator=lambda x: np.linalg.norm(x, 2),
            model_name_indentifier="model_type",
            final_metric_method=weighted_selection_metric_with_mean_std,
            metric_weights=[0, 1, 0, 0],
            scale_metrics=False,
        )


def tune_ptype_utility(base_path, grain, **kwargs):

    data_path = f"{base_path}/{kwargs['target']}/{grain}"

    train = pd.read_parquet(f"{data_path}/train.parquet")
    validation = pd.read_parquet(f"{data_path}/validation.parquet")

    train = train[train[kwargs["target"]] > 0]

    if not ("log_func" in kwargs.keys()):
        kwargs["log_func"] = print

    best_results = {}
    for name, tune_func, iters, features in [
        ("lasso", regression.lasso_objective, 20, kwargs["coefficient_features"]),
        ("ridge", regression.ridge_objective, 20, kwargs["coefficient_features"]),
        ("elastic", regression.elastic_objective, 20, kwargs["coefficient_features"]),
        # ("randomforest", tree.randomforest_objective, 50, kwargs['tree_features']),
        # ("xgb", tree.xgb_objective, 50, kwargs["tree_features"]),
        # ("quantilereg", quantilereg.quantile_objective, 30, kwargs["coefficient_features"]),
        (
            "bayesianreg",
            bayesianreg.bayesian_objective,
            50,
            kwargs["coefficient_features"],
        ),
    ]:

        # kwargs['log_func'](f"Running tuning objective {name} for {iters} iterations")

        kwargs["log_func"](f"Starting tuning for {name}")
        tic = time.time()

        coeff = model_selection.TuneModels(
            iters,
            12,
            train=(train, train[kwargs["target"]]),
            val=(validation, validation[kwargs["target"]]),
            features=features,
            metric=kwargs["ms"],
            ts_columns=kwargs["ts_columns"],
            ts_values=grain.split("/"),
            log_func=kwargs["log_func"],
            target=kwargs["target"],
            keep_vars=[],
            model_name=name,
        )
        best_result = coeff.tune_model(tune_func)
        best_results[f"{name}_best_result"] = best_result

        kwargs["log_func"](
            f"Finished tuning for {name}. Time Taken: {round((time.time()-tic)/60, 2)} minutes"
        )

        # kwargs['log_func'](f"Best results for {name} after {iters} iterations: {best_results[f'{name}_best_result']}")

    return best_results
