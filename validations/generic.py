import numpy as np
import pandas as pd


def finite_values(df: pd.DataFrame):

    assert (
        df.isna().sum().sum() == 0
    ), f"NANs in dataframe. {df.isna().sum()[df.isna().sum()>0]}"
    assert (df == np.inf).sum().sum() == 0, f"Inf in dataframe"
