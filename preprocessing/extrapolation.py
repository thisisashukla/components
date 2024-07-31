import logging
import numpy as np
import pandas as pd
from prophet import Prophet
from datetime import timedelta
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

from common import decorators
from validations import time_series
from projects.forecasting.packaged.common import utils
from common import utils as common_utils


@decorators.time_function
def run_extrapolation(
    df: pd.DataFrame,
    training_start_date: str = None,
    cutoff_date: str = None,
    date_column: str = "date",
    group_columns: list = [],
    extrapolation_kwargs: dict = {},
):
    def extrapolate(actual):
        extrapolated_df = prophet_extrapolation(
            actual, date_column=date_column, **extrapolation_kwargs
        )

        combined = pd.concat([actual, extrapolated_df], axis=0).reset_index(drop=True)

        return combined

    true_df = df.copy()

    if not (training_start_date is None):
        true_df = true_df[true_df[date_column] >= training_start_date]

    if not (cutoff_date is None):
        true_df = true_df[true_df[date_column] < cutoff_date].copy()

    if len(group_columns) == 0:
        combined = extrapolate(true_df)
    else:
        combined = common_utils.run_function_on_groups(
            true_df, extrapolate, group_columns, parallel=False
        )

    for col in extrapolation_kwargs["target_columns"]:
        rolling_median = combined[col].rolling(7, min_periods=1).median()
        combined[col] = np.where(
            (combined[date_column] > true_df[date_column].max())
            & (combined[col] < 0.5 * rolling_median),
            rolling_median,
            combined[col],
        )

    return combined


def prophet_extrapolation(
    df: pd.DataFrame,
    date_column: str = "date",
    target_columns: list = None,
    ratio: bool = False,
    model_kwargs: dict = {},
    regressor_columns: list = [],
    extrapolation_days: int = 365,
    extrapolation_end_data: str = None,
    date_feature_df: pd.DataFrame = None,
):

    input_df = df.copy().rename(columns={date_column: "ds"})

    if target_columns is None:
        target_columns = input_df.select_dtypes(include=["int64", "float64"]).columns
        target_columns = list(set(target_columns) - set(regressor_columns))

    if len(regressor_columns) > 0:
        input_df = input_df[["ds"] + target_columns + regressor_columns].sort_values(
            by=["ds"]
        )
    else:
        input_df = input_df[["ds"] + target_columns].sort_values(by=["ds"])

    df_max_date = input_df["ds"].max()
    if extrapolation_end_data is None:
        extrapolation_dates = pd.date_range(
            pd.to_datetime(df_max_date) + timedelta(days=1),
            pd.to_datetime(df_max_date)
            + timedelta(days=1)
            + timedelta(days=extrapolation_days),
        )
    else:
        extrapolation_dates = pd.date_range(
            pd.to_datetime(df_max_date) + timedelta(days=1), extrapolation_end_data
        )
    extrapolation_dates = list(
        map(lambda x: x.strftime("%Y-%m-%d"), extrapolation_dates)
    )

    prediction_date_df = pd.DataFrame({date_column: extrapolation_dates})
    prediction_date_df = prediction_date_df.merge(date_feature_df).rename(
        columns={date_column: "ds"}
    )

    extrapolation_df = None
    for col in target_columns:
        if ratio:
            model = LogisticRegression(**model_kwargs)
            model.fit(input_df[regressor_columns], input_df[col])
            pred = prediction_date_df.copy()
            pred["yhat"] = model.predict(prediction_date_df[regressor_columns])
        else:
            model = Prophet(**model_kwargs)
            if len(regressor_columns) > 0:
                for reg in regressor_columns:
                    model.add_regressor(reg)
            pr_df = input_df.rename(columns={col: "y"})
            model.fit(pr_df)
            pred = model.predict(prediction_date_df)

            if all(pr_df.dropna(subset=["y"])["y"] >= 0):
                pred["yhat"] = pred["yhat"].apply(lambda x: max(x, 0))

            if pr_df.dropna(subset=["y"])["y"].mod(1).sum() == 0:
                pred["yhat"] = pred["yhat"].apply(np.ceil)

        pred = pred[["ds", "yhat"]].rename(columns={"ds": date_column, "yhat": col})

        if extrapolation_df is None:
            extrapolation_df = pred
        else:
            extrapolation_df = extrapolation_df.merge(pred)

    extrapolation_df[date_column] = utils.convert_date_to_string(
        extrapolation_df[date_column]
    )

    extrapolation_df = extrapolation_df.merge(date_feature_df)

    assert all(
        [t in extrapolation_df.columns for t in target_columns]
    ), f"Target columns missing"
    time_series.validate_timeseries(extrapolation_df, [date_column])

    return extrapolation_df
