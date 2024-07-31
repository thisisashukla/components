import numpy as np
import pandas as pd

from validations import generic


def validate_timeseries(df: pd.DataFrame, ts_grain_columns: list) -> None:

    generic.validate_dataframe(df)
    assert (
        df.drop_duplicates(subset=ts_grain_columns).shape[0] == df.shape[0]
    ), f"Duplicates at time series level"
