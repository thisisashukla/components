import numpy as np
import pandas as pd
from typing import Union
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.baseml import BaseMLModel


class TreeModel(BaseMLModel):
    """
    A generic class for tree-based models, including Decision Trees, Random Forests, XGBoost, CatBoost, and LightGBM.

    Attributes:
        tree_class (Union[DecisionTreeClassifier, DecisionTreeRegressor, RandomForestClassifier,
                          RandomForestRegressor, XGBClassifier, XGBRegressor, CatBoostClassifier,
                          CatBoostRegressor, LGBMRegressor, LGBMClassifier]): The tree-based model class to use.
        tree_params (dict): Parameters for initializing the tree-based model. Defaults to an empty dict.
        name (str): Optional name for the model.
        date_column (str): The column name representing the date. Defaults to "date".
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        tree_class: Union[
            DecisionTreeClassifier,
            DecisionTreeRegressor,
            RandomForestClassifier,
            RandomForestRegressor,
            XGBClassifier,
            XGBRegressor,
            CatBoostClassifier,
            CatBoostRegressor,
            LGBMRegressor,
            LGBMClassifier,
        ],
        tree_params: dict = {},
        name: str = None,
        date_column: str = "date",
        **kwargs
    ):
        """
        Initialize the TreeModel with the provided data, features, target, and tree class.

        Args:
            data (dict): Dictionary of datasets (e.g., train, validation, test).
            features (list[str]): List of feature column names.
            target (str): The target column name.
            grain_columns (list[str]): List of columns to use for grain-level grouping.
            tree_class (Union): The class of the tree-based model (e.g., DecisionTreeClassifier, XGBRegressor).
            tree_params (dict, optional): Parameters for the tree-based model. Defaults to an empty dict.
            name (str, optional): Optional name for the model. Defaults to None.
            date_column (str, optional): The name of the date column. Defaults to "date".
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            tree_class=tree_class,
            tree_params=tree_params,
            date_column=date_column,
            **kwargs,
        )

    def _initialize(self):
        """
        Initialize the tree-based model using the specified tree class and parameters.

        Returns:
            A tree-based model instance (e.g., DecisionTreeRegressor, XGBRegressor).
        """
        return self.tree_class(**self.tree_params)

    def _train(self, df: pd.DataFrame) -> None:
        """
        Train the tree-based model on the provided DataFrame.

        Args:
            df (pd.DataFrame): The training dataset containing the features and target.

        Returns:
            None
        """
        self.model.fit(df[self.features], df[self.target])

    def _predict(self, data: dict) -> dict:
        """
        Generate predictions for the provided datasets using the trained tree-based model.

        Args:
            data (dict): Dictionary of datasets (e.g., 'validation', 'test') to predict on.

        Returns:
            dict: A dictionary where the keys are dataset names and the values are DataFrames containing predictions.
        """
        predictions = {}
        for k, df in data.items():
            pred = self.model.predict(df[self.features])
            df["prediction"] = pred
            df["model_type"] = self.tree_class.__name__
            predictions[k] = df

        return predictions

    def profile(self) -> pd.DataFrame:
        """
        Generate a feature importance profile for the tree-based model.

        Returns:
            pd.DataFrame: A DataFrame containing the feature names and their importance scores.
        """
        if hasattr(self.model, "get_booster"):
            feature_names = self.model.get_booster().feature_names
        else:
            feature_names = self.model.feature_name_

        importance = self.model.feature_importances_

        profile_df = pd.DataFrame(
            {
                "features": feature_names,
                "importance": importance,
            }
        ).sort_values("importance", ascending=False)
        profile_df["model_type"] = self.tree_class.__name__
        profile_df["model_class"] = self.name

        return profile_df


def generic_tree_tuning_objective(**kwargs):

    model = kwargs["model"](
        **{k: v for k, v in kwargs.items() if not (k in ["trial", "metric", "model"])}
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

    return (
        model_selection["final_metric"].values[0],
        model.final_features,
        model,
        preds,
        kwargs["metric"],
        kwargs["trial"],
    )


def decision_objective(trial, **kwargs):

    train_x = kwargs["data"]["train"]

    max_depth = trial.suggest_int(
        "max_depth", 2, int(-1 + np.log(train_x.shape[0]) / np.log(2))
    )

    min_samples_leaf = trial.suggest_int(
        "min_samples_leaf", 7, max(10, train_x.shape[0] / 4)
    )

    max_features = trial.suggest_int(
        "max_features",
        int(max(3, 0.1 * train_x[kwargs["features"]].shape[1])),
        int(max(10, 0.5 * train_x[kwargs["features"]].shape[1])),
    )

    tree_params = {
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "random_state": 42,
    }

    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_tree_tuning_objective(
        trial=trial, tree_class=DecisionTreeRegressor, tree_params=tree_params, **kwargs
    )

    return ms


def randomforest_objective(trial, **kwargs):

    train_x = kwargs["data"]["train"]

    n_estimators = trial.suggest_int("n_estimators", 50, 1000, log=True)

    max_depth = trial.suggest_int(
        "max_depth", 2, int(-1 + np.log(train_x.shape[0]) / np.log(2))
    )

    min_samples_leaf = trial.suggest_int(
        "min_samples_leaf", 7, max(10, train_x.shape[0] / 4)
    )

    max_features = trial.suggest_int(
        "max_features",
        int(max(3, 0.1 * train_x[kwargs["features"]].shape[1])),
        int(max(10, 0.5 * train_x[kwargs["features"]].shape[1])),
    )

    tree_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
    }

    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_tree_tuning_objective(
        trial=trial, tree_class=RandomForestRegressor, tree_params=tree_params, **kwargs
    )

    return ms


def xgb_objective(trial, **kwargs):

    train_x = kwargs["data"]["train"]

    n_estimators = trial.suggest_int("n_estimators", 50, 1000, log=True)

    max_depth = trial.suggest_int(
        "max_depth", 2, int(-1 + np.log(train_x.shape[0]) / np.log(2))
    )

    min_child_weight = trial.suggest_int(
        "min_child_weight", 7, max(10, train_x.shape[0] / 4)
    )

    subsample = trial.suggest_float("subsample", 0.4, 1.0)

    learning_rate = trial.suggest_float("learning_rate", 0.001, 0.2)

    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)

    tree_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_child_weight": min_child_weight,
        "subsample": subsample,
        "learning_rate": learning_rate,
        "colsample_bytree": colsample_bytree,
    }

    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_tree_tuning_objective(
        trial=trial, tree_class=XGBRegressor, tree_params=tree_params, **kwargs
    )

    return ms


def catboost_objective(trial, **kwargs):

    train_x = kwargs["data"]["train"]

    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 100)
    bagging_temperature = trial.suggest_float("bagging_temperature", 1, 100)

    depth = trial.suggest_int(
        "depth", 2, max(int(-1 + np.log(train_x.shape[0]) / np.log(2)), 16)
    )

    min_data_in_leaf = trial.suggest_int(
        "min_data_in_leaf", 7, max(10, train_x.shape[0] / 4)
    )

    tree_params = {
        "l2_leaf_reg": l2_leaf_reg,
        "bagging_temperature": bagging_temperature,
        "depth": depth,
        "min_data_in_leaf": min_data_in_leaf,
    }

    (
        ms,
        final_feature_set,
        best_model,
        preds,
        metric,
        trial,
    ) = generic_tree_tuning_objective(
        trial=trial, tree_class=CatBoostRegressor, tree_params=tree_params, **kwargs
    )

    return ms
