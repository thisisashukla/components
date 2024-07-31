import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.base import BaseModel


class TreeModel(BaseModel):
    def __init__(
        self,
        tree_class: RandomForestClassifier
        or RandomForestRegressor
        or XGBClassifier
        or XGBRegressor
        or CatBoostClassifier
        or CatBoostRegressor,
        tree_params: dict,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        **kwargs
    ):

        super().__init__(
            data, features, target, grain_columns, name, log_function, **kwargs
        )

        self.tree_class = tree_class
        self.tree_params = tree_params

    def _initialize(self):

        return self.tree_class(**self.tree_params)

    def _train(self, df: pd.DataFrame) -> None:

        self.model.fit(df[self.features], df[self.target])

    def _predict(self, data: dict) -> dict:

        preditions = {}
        for k, df in data.items():
            pred = self.model.predict(df[self.features])
            df["prediction"] = pred
            df["model_type"] = self.tree_class.__name__
            preditions[k] = df

        return preditions
