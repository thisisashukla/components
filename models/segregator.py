import typing
import warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

from common import *


# default criterion function being used currently (2022-09-08)
def segregation_criterion(df: pd.Series) -> pd.DataFrame:

    TS_COLUMNS = ["city", "ptype"]
    TARGET = "atta_sales"
    ZERO_THRESHOLD = 0.4
    CPD_THRESHOLD = 4
    MINIMUM_TFT_TS_LENGTH = 16

    criterion = df[TS_COLUMNS].drop_duplicates()
    criterion["data_split_criterion"] = "True"

    return criterion


class Segregator:
    def __init__(
        self,
        data: dict,
        criterion: typing.Union[typing.Callable, pd.DataFrame],
        group_keys: typing.List,
        fit_on: str = "train",
        env: str = "dev",
        logger=None,
        verbose=1,
    ):

        self.data = data
        self.criterion = criterion
        self.env = env
        if logger is None:
            self.seg_log = utils.get_log_function(self.env == "prod", "SEGREGATOR")
        else:
            self.seg_log = utils.get_log_function(
                self.env == "prod", "SEGREGATOR", logger
            )
        self.verbose = verbose
        if verbose == 0:
            self.verbose = self.dummy_log
        self.handle_fit_on(fit_on)
        self.group_keys = group_keys
        if criterion is None:
            criterion = segregation_criterion
        self.criterion_df = self.convert_criterion_to_df(criterion, to_fit=self.fit_on)

        self.seg_log(f"Data dict has {self.data.keys()} keys.")
        self.seg_log(
            f"Criterion dataframe has {self.criterion_df['data_split_criterion'].nunique()} values ({self.criterion_df['data_split_criterion'].unique()}). This will be the number of parts which will be created."
        )

    def dummy_log(self, msg):
        pass

    def handle_fit_on(self, fit_on):

        self.seg_log(f"Fitting segregator on {fit_on}")

        fiton = fit_on.split("+")

        if len(fiton) == 1:
            pass
        else:
            self.data[fit_on] = pd.concat(
                [self.data[k] for k in fiton], axis=0
            ).reset_index(drop=True)
        self.fit_on = fit_on

    def __inner_join__(self, data: pd.DataFrame, value) -> pd.DataFrame:

        common = data.merge(
            self.criterion_df[self.criterion_df["data_split_criterion"] == value],
            on=self.group_keys,
            how="inner",
        )

        return common.drop(columns=["data_split_criterion"])

    def convert_criterion_to_df(
        self,
        criterion: typing.Union[typing.Callable, pd.DataFrame],
        to_fit="train",
    ) -> pd.DataFrame:

        if callable(criterion):
            criterion_df = criterion(self.data[to_fit].copy())
        elif isinstance(criterion, pd.DataFrame):
            criterion_df = criterion

        assert isinstance(
            criterion_df, pd.DataFrame
        ), f"Criterion function passed should return a dataframe"
        assert all(
            [
                c in criterion_df.columns
                for c in ["data_split_criterion"] + self.group_keys
            ]
        ), f"Dataframe passed or function passed should return a dataframe containing 'data_split_criterion', {', '.join(self.group_keys)} columns"
        assert (
            criterion_df.shape[0]
            == self.data[self.fit_on][self.group_keys].drop_duplicates().shape[0]
        ), f"The time series groups returned by criterion do not match the {self.fit_on} data"

        columns = self.group_keys + ["data_split_criterion"]
        self.seg_log(
            f"Time series counts for segregation values: {criterion_df['data_split_criterion'].value_counts().to_dict()}"
        )

        return criterion_df[columns]

    def get_partial(
        self, criterion_value, key: typing.Union[str, typing.List] = None
    ) -> typing.Union[typing.Dict, pd.DataFrame]:

        assert (
            criterion_value in self.criterion_df["data_split_criterion"].unique()
        ), f"Criterion value passed ({criterion_value}), {type(criterion_value)} is not present in criterion df"

        ret_keys = [key] if isinstance(key, str) else key
        if key is None:
            ret_keys = self.data.keys()

        data = {}
        for k in ret_keys:
            if "+" not in k:
                data[k] = self.__inner_join__(self.data[k], criterion_value)

        return data

    def apply_criterion(self, data: pd.DataFrame):

        result = {}
        for criterion_value in self.criterion_df.data_split_criterion.unique():
            result[criterion_value] = self.__inner_join__(data, criterion_value)

        assert len(result.keys()) == self.criterion_df.data_split_criterion.nunique()

        return result
