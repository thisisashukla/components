import pandas as pd
from prophet import Prophet

from models.base import BaseModel


class ProphetModel(BaseModel):
    def __init__(
        self,
        data: dict,
        target: str,
        grain_columns: list[str],
        date_column: str,
        name: str = None,
        log_function: callable = None,
        weekly_seasonality: bool = False,
        yearly_seasonality: bool = False,
        use_holidays: bool = False,
        holidays: pd.DataFrame = None,
        regressors: list[str] = [],
    ) -> None:

        super().__init__(
            data,
            regressors,
            target,
            grain_columns,
            name,
            log_function,
        )

        self.date_column = date_column
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.use_holidays = use_holidays
        self.holidays = holidays

    def _initialize(self):

        model = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )

        for reg in self.features:
            model.add_regressor(reg)

        return model

    def _train(self, df: pd.DataFrame) -> None:

        X = df[self.features + [self.date_column]].rename(
            columns={self.date_column: "ds", self.target: "y"}
        )

        self.model.fit(X)

    def _predict(self, data: dict) -> dict:

        predictions = {}
        for k, df in data.items():
            X = df[self.features + [self.date_column]].rename(
                columns={self.date_column: "ds", self.target: "y"}
            )
            pred = self.model.predict(X)

            df["prediction"] = pred
            df["model_type"] = "Prophet"

            predictions[k] = df

        return predictions
