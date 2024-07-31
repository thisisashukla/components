import os
import csv
import time
import pytz
import yaml
import logging
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import _pickle as cPickle
import multiprocessing as mp
from collections import namedtuple
from joblib import Parallel, delayed
from functools import partial, reduce
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed


import queries
from common.connection import trino_connection


def read_config(config_path):
    with open(config_path, "r") as f_in:
        config = yaml.safe_load(f_in)
    config["skip_dates"] = list(
        reduce(
            lambda x, y: x.union(y),
            [
                pd.date_range(
                    config["remove_dates_start"][idx], config["remove_dates_end"][idx]
                )
                for idx in range(len(config["remove_dates_start"]))
            ],
        )
    )
    config["skip_dates"] = list(
        map(lambda x: x.strftime("%Y-%m-%d"), config["skip_dates"])
    )
    for key in config.keys():
        if key.endswith("_date"):
            config[key] = config[key].strftime("%Y-%m-%d")

    Config = namedtuple("Config", config)
    config_obj = Config(**config)
    return config_obj


def read_query(query_name: str, module: str = None):

    if module is None:
        file = getattr(queries, query_name)
    else:
        file = getattr(importlib.import_module(f"{module}.queries"), query_name)

    return file.query


def execute_query(
    query_name: str,
    module: str = None,
    query_params: dict = {},
    connection=trino_connection,
):

    query = read_query(query_name, module)

    if len(query_params.keys()) > 0:
        data = pd.read_sql_query(sql=query.format(**query_params), con=connection)
    else:
        data = pd.read_sql_query(sql=query, con=connection)

    return data


def log_msg(msg, pipeline=None, prod=False, channel="bl_forecasting_process_alerts"):
    timestamp = localize_ts(datetime.now())

    if not (pipeline is None):
        msg = f"{timestamp} {pipeline.upper()}: {msg}"
    else:
        msg = f"{timestamp}: {msg}"

    print(msg)
    logger_func(msg)
    if prod:
        try:
            pb.send_slack_message(channel=channel, text=msg)
        except Exception as E:
            print(f"Unable to send slack msg. Error = {E}")
    else:
        print(msg)


def get_logger(logger_name: str, logging_level=logging.DEBUG):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging_level)

    fh = logging.FileHandler(
        f"{logger_name}_{pd.to_datetime(localize_ts(datetime.now())).strftime('%Y-%m-%d_%H')}.txt"
    )
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    return logger


def get_log_function(prod, pipeline=None, logger=None):

    if logger is None:
        logger = get_logger("forecasting_logger", logging.DEBUG)

    log_func = partial(log_msg, logger=logger, pipeline=pipeline.upper(), prod=prod)

    return log_func


def localize_ts(ts, only_date=False):

    timezone = pytz.timezone("Asia/Kolkata")

    if only_date:
        return (
            pytz.utc.localize(ts, is_dst=None).astimezone(timezone).strftime("%Y-%m-%d")
        )

    return (
        pytz.utc.localize(ts, is_dst=None)
        .astimezone(timezone)
        .strftime("%Y-%m-%d %H:%M:%S")
    )


def save_csv(df, local_path):

    df.to_csv(local_path, index=None, quoting=csv.QUOTE_NONNUMERIC)


def run_query_by_parts(query, variable, con, jump=50000):

    variable = list(set(variable))

    results = []
    print(f"Running query in {int(len(variable)/jump)+1} steps of {jump}")
    for i in range(0, len(variable), jump):
        sql = query.format(variable=tuple(variable[i : i + jump]))
        df = pd.read_sql_query(sql=sql, con=con)
        results.append(df)
        print(f"{int(i/jump)+1} steps complete")

    return pd.concat(results, axis=0).reset_index(drop=True)


def parallelize_func(func, input_array):

    pool = ThreadPoolExecutor(max_workers=8)

    outputs = []
    tic = time.time()
    for output in tqdm(pool.map(func, input_array), total=len(input_array)):
        outputs.append(output)
    tac = time.time()

    log_msg(
        f"Parallel process execution for input array of size {len(input_array)} complete in {round(tac-tic, 2)} seconds"
    )

    return outputs


def read_pickle(path: str):
    with open(path, "rb") as input_file:
        object = cPickle.load(input_file)

    return object


def write_pickle(obj: dict, path: str) -> None:
    with open(path, "wb") as output_file:
        cPickle.dump(obj, output_file)


def execute_by_parts(
    query, start_date, end_date, path="./data_cache", n_parts=10, **kwargs
):
    def _execute(query, start_date, end_date, **kwargs):
        conn = trino_connection
        _part = pd.read_sql(
            query.format(start_date=start_date, end_date=end_date, **kwargs), conn
        )
        del conn
        return (_part, start_date, end_date)

    time_range = pd.date_range(
        start=start_date, end=end_date, periods=n_parts + 1
    ).tolist()
    _df = []
    count = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        _future_list = [
            executor.submit(
                _execute,
                query,
                start_date=time_range[i - 1].strftime("%Y-%m-%d"),
                end_date=time_range[i].strftime("%Y-%m-%d"),
                **kwargs,
            )
            for i in range(1, len(time_range))
            if not os.path.isfile(
                f"{path}/part_{time_range[i-1].strftime('%Y-%m-%d')}_{time_range[i].strftime('%Y-%m-%d')}.parquet"
            )
        ]
        for future in as_completed(_future_list):
            print(count, end="\r")
            _tmp, start_date, end_date = future.result()
            _tmp.to_parquet(f"{path}/part_{start_date}_{end_date}.parquet")
            _df.append(_tmp)
            count += 1
    _df = pd.concat(_df)
    return _df


def run_func(inp):

    i, grp, func, func_kwargs, group_columns = inp

    function_result = func(grp, **func_kwargs)
    if len(group_columns) == 1:
        function_result[group_columns[0]] = i
    else:
        for a, b in zip(group_columns, i):
            function_result[a] = b

    return function_result


def run_function_on_groups(
    df: pd.DataFrame,
    func: callable,
    group_columns: list,
    func_kwargs: dict = {},
    parallel=False,
    preprocess: callable = None,
) -> pd.DataFrame:

    if preprocess is None:
        preprocess = lambda x: x

    groups = [
        (i, preprocess(grp), func, func_kwargs, group_columns)
        for i, grp in df.groupby(group_columns)
    ]
    print(f"Number of groups identified for columns {group_columns}: {len(groups)}")

    df_groups = []
    if parallel:
        n_jobs = int(mp.cpu_count() * 0.5)
        try:
            df_groups = Parallel(n_jobs=n_jobs)(
                delayed(run_func)(group) for group in tqdm(groups, total=len(groups))
            )
        except Exception as e:
            print(f"Parallel execution failed. Exception: {e}")
    else:
        df_groups = tqdm(map(run_func, groups), total=len(groups))

    result = pd.concat(df_groups, axis=0).reset_index(drop=True)

    return result


def one_hot_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:

    all_levels = list(df[column].unique())

    if len(all_levels) > 1:
        all_levels.sort()

    df[np.array(["{}_{}".format(column, i) for i in all_levels])] = pd.get_dummies(
        df[column]
    )

    return df


def sanitise_string(s):
    return s.replace(",", "").replace("&", "")


def push_output_to_trino(df, table_kwargs, logger=print):

    df["insert_ts"] = datetime.now()

    prod_columns = [c["name"] for c in table_kwargs["column_dtypes"]]

    try:
        df = df[prod_columns]
        pb.to_trino(df, **table_kwargs)
        logger(
            f"Data pushed to {table_kwargs['table_name']}. Records pushed: {df.shape[0]}"
        )
    except Exception as e:
        raise Exception(f"Results push to table failed. Exception: {e}")


def execute_trino_push_with_kwargs(
    df: pd.DataFrame,
    module: str,
    table_kwarg_name: dict,
    env: str = "dev",
    logger=print,
):

    table_kwargs = getattr(
        importlib.import_module(f"{module}.common.constants"), table_kwarg_name
    )

    logger(f"Table kwargs read for {table_kwargs['table_name']}")

    if env != "prod":
        logger(f"Changing schema to blinkit_staging")
        table_kwargs["schema"] = "blinkit_staging.ds_etls"

    push_output_to_trino(df, table_kwargs, logger)


def to_trino_flat_by_range(sql, start_date, cutoff_date, interval, grain, kwargs):

    i = 1

    while str(start_date) > str(cutoff_date - timedelta(days=interval)):

        end_date = start_date + timedelta(days=interval)
        print(f"Start Date:{start_date}, End Date: {end_date}")

        sql_query1 = sql.replace("start_date", str(start_date))
        sql_query1 = sql_query1.replace("end_date", str(end_date))

        df1 = pd.read_sql_query(sql=sql_query1, con=trino_connection)

        df1 = df1.melt(id_vars=grain)
        df1["value"] = df1["value"].astype(str)
        df1["insert_ts"] = datetime.now()
        df1["insert_ts"] = df1["insert_ts"].astype(str)
        push_output_to_trino(df1, kwargs)
        print(f"Step {i} Completed")

        start_date = start_date - timedelta(days=interval + 1)
        i = i + 1
    return
