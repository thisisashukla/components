import numpy as np
import pandas as pd
from functools import reduce
from datetime import datetime
from dateutil.relativedelta import relativedelta

from common import utils as common_utils


def create_cal(sdate, edate):
    date_df = pd.DataFrame({"date_": pd.date_range(sdate, edate, freq="d")})
    date_df.date_ = date_df.date_.dt.date
    date_df[["weekday", "week_start_date"]] = [
        [datetime.strftime(x, "%a"), x - relativedelta(days=x.weekday())]
        for x in date_df.date_
    ]
    date_df[["year_week", "week_num", "bi_week", "month", "quarter"]] = [
        [
            datetime.strftime(x, "%Y-%V"),
            int(datetime.strftime(x, "%V")),
            int(np.ceil(int(datetime.strftime(x, "%V")) / 2)),
            x.month,
            int(np.ceil(x.month / 3)),
        ]
        for x in date_df.week_start_date
    ]
    date_df["weekend"] = np.where(date_df.weekday.isin(["Sat", "Sun"]), 1, 0)
    date_cols = ["date_", "week_start_date"]
    date_df[date_cols] = date_df[date_cols].apply(to_date)
    one_hot_vars = ["weekday", "bi_week", "month", "quarter"]

    def one_hot_encoder(var_name, df=date_df):
        all_levels = list(df[var_name].unique())
        if len(all_levels) > 1:
            all_levels.sort()
        df[np.array(["{}_{}".format(var_name, i) for i in all_levels])] = (
            pd.get_dummies(date_df[var_name])
        )

    list(map(one_hot_encoder, one_hot_vars))
    return date_df


def get_date_properties(dates, event_calendar):

    ddf = pd.DataFrame({"date": dates}).drop_duplicates()
    ddf["date"] = pd.to_datetime(ddf["date"]).dt.strftime("%Y-%m-%d")
    # event_calendar["event_date"] = pd.to_datetime(event_calendar['event_date']).dt.strftime('%Y-%m-%d')
    ddf = pd.merge(
        ddf,
        event_calendar[["event_date", "calendar_holiday_type"]],
        left_on="date",
        right_on="event_date",
    ).fillna(0)

    ddf["is_event"] = (
        1  # np.where(ddf["date"].isin(event_calendar["event_date"]), 1, 0)
    )
    ddf["is_hfs"] = np.where(pd.to_datetime(ddf["date"]).dt.day <= 7, 1, 0)
    ddf["is_weekend"] = np.where(pd.to_datetime(ddf["date"]).dt.dayofweek >= 5, 1, 0)
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek.isin([0, 4])) & (ddf.calendar_holiday_type=="Gazetted Holiday"), 1, 0)
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek == 5) & ((ddf["is_long_weekend"].shift(1)==1) | (ddf["is_long_weekend"].shift(-2)==1)), 1, ddf["is_long_weekend"])
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek == 6) & ((ddf["is_long_weekend"].shift(2)==1) | ddf["is_long_weekend"].shift(-1)==1), 1, ddf["is_long_weekend"])

    ddf["event_hfs"] = ddf["is_event"] * ddf["is_hfs"]
    ddf["hfs_weekend"] = ddf["is_hfs"] * ddf["is_weekend"]
    ddf["event_hfs_weekend"] = ddf["is_event"] * ddf["is_hfs"] * ddf["is_weekend"]

    ddf.drop(columns=["is_event"], inplace=True)

    ddf.columns = [f"date_{c}" if c != "date" else c for c in ddf.columns]
    ddf.drop_duplicates("date", inplace=True)
    return ddf


def create_events(calendar_df: pd.DataFrame):
    day_events = (
        # pb.from_sheets(
        #     sheetid=config["calendar_sheet_id"],
        #     sheetname=config["calendar_sheet_name"]
        # )
        # pd.read_excel('../prodnotebook/Calendar Events India.xlsx', sheet_name="Final Event Calendar")
        calendar_df.replace(r"^\s*$", np.nan, regex=True).astype(
            {
                "weekend_flag": int,
                "event_count": int,
                "hfs_flag": int,
                "YEAR": int,
                "MONTH": int,
                "DAY OF MONTH": int,
            }
        )
    )
    date_cols = ["date_", "week_start_date"]
    day_events[date_cols] = day_events[date_cols].apply(to_date)
    # Cleaning Up Event Names and creating dummies
    day_events["event_name"] = (
        day_events["event_name"]
        .str.replace(r"[,'./]", "", regex=True)
        .str.lower()
        .str.replace("\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
    )
    one_hot_vars = ["event_name"]

    def one_hot_encoder(var_name, df=day_events):
        all_levels = list(df[var_name].unique())
        if len(all_levels) > 1:
            all_levels.sort()
        df[np.array(["flag_{}".format(i) for i in all_levels])] = pd.get_dummies(
            df[var_name]
        )

    list(map(one_hot_encoder, one_hot_vars))

    # Creating day_events data unique at date level
    flag_vars = [i for i in day_events.columns if "flag_" in i]
    day_events = day_events.groupby(
        ["date_", "week_start_date", "weekend_flag", "event_count"], as_index=False
    )[flag_vars].sum()
    day_events["weekend_event"] = (
        day_events["weekend_flag"].astype(int) * day_events["event_count"]
    )
    # Creating week level events flags
    [flag_vars.append(i) for i in ["event_count", "weekend_event"]]
    week_events = day_events.groupby("week_start_date", as_index=False)[flag_vars].sum()
    day_events.drop(["week_start_date", "weekend_flag"], axis=1, inplace=True)
    return (day_events, week_events, flag_vars)


def to_date(col):
    return pd.to_datetime(col).dt.date


def to_int(col):
    return col.astype(int)


def get_date_properties(dates, event_calendar):

    ddf = pd.DataFrame({"date": dates}).drop_duplicates()
    ddf["date"] = pd.to_datetime(ddf["date"]).dt.strftime("%Y-%m-%d")
    # event_calendar["event_date"] = pd.to_datetime(event_calendar['event_date']).dt.strftime('%Y-%m-%d')
    # ddf = pd.merge(ddf, event_calendar[["event_date"]], left_on="date", right_on="event_date").fillna(0)

    ddf["is_event"] = np.where(ddf["date"].isin(event_calendar["event_date"]), 1, 0)
    ddf["is_hfs"] = np.where(pd.to_datetime(ddf["date"]).dt.day <= 7, 1, 0)
    ddf["is_weekend"] = np.where(pd.to_datetime(ddf["date"]).dt.dayofweek >= 5, 1, 0)
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek.isin([0, 4])) & (ddf.calendar_holiday_type=="Gazetted Holiday"), 1, 0)
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek == 5) & ((ddf["is_long_weekend"].shift(1)==1) | (ddf["is_long_weekend"].shift(-2)==1)), 1, ddf["is_long_weekend"])
    # ddf["is_long_weekend"] = np.where((pd.to_datetime(ddf["date"]).dt.dayofweek == 6) & ((ddf["is_long_weekend"].shift(2)==1) | ddf["is_long_weekend"].shift(-1)==1), 1, ddf["is_long_weekend"])

    ddf["event_hfs"] = ddf["is_event"] * ddf["is_hfs"]
    ddf["hfs_weekend"] = ddf["is_hfs"] * ddf["is_weekend"]
    ddf["event_hfs_weekend"] = ddf["is_event"] * ddf["is_hfs"] * ddf["is_weekend"]

    # ddf.drop(columns=["is_event"], inplace=True)

    ddf.columns = [f"date_{c}" if c != "date" else c for c in ddf.columns]
    ddf.drop_duplicates("date", inplace=True)
    return ddf


def _generate_runup(dt_list, days=14):
    _fragments = []
    for dt in dt_list:
        runup_start_dt = dt - pd.to_timedelta(f"{days-1} days")
        runup_end_dt = dt  # + pd.to_timedelta("1 days")
        _df = pd.DataFrame(
            {"_denominator": list(range(days, 0, -1)), "_numerator": [1] * (days)},
            index=map(
                lambda x: str(x.date()),
                pd.date_range(start=runup_start_dt, end=runup_end_dt),
            ),
        )
        _df[f"event_runup_{dt}"] = _df["_numerator"] / _df["_denominator"]
        _fragments.append(_df[[f"event_runup_{dt}"]])
    df = reduce(
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
        _fragments,
    )
    df["event_runup"] = df.apply(np.nanmax, axis=1)
    return df[["event_runup"]]


def _generate_rundown(dt_list, days=7):
    _fragments = []
    for dt in dt_list:
        rundown_start_dt = dt
        rundown_end_dt = dt + pd.to_timedelta(f"{days-1} days")
        _df = pd.DataFrame(
            {"_denominator": list(range(1, days + 1)), "_numerator": [1] * (days)},
            index=map(
                lambda x: str(x.date()),
                pd.date_range(start=rundown_start_dt, end=rundown_end_dt),
            ),
        )
        _df[f"event_rundown_{dt}"] = _df["_numerator"] / _df["_denominator"]
        _fragments.append(_df[[f"event_rundown_{dt}"]])
    df = reduce(
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
        _fragments,
    )
    df["event_rundown"] = df.apply(np.nanmax, axis=1)
    return df[["event_rundown"]]


def festive_runup_rundown(event_name, dates, runup_days=14, rundown_days=7):
    runup = _generate_runup(dates, runup_days)
    rundown = _generate_rundown(dates, rundown_days)
    return pd.merge(
        runup.rename(columns={"event_runup": f"{event_name}_runup"}),
        rundown.rename(columns={"event_rundown": f"{event_name}_rundown"}),
        left_index=True,
        right_index=True,
        how="outer",
    )


def get_master_date_df(
    sheet_calendar: pd.DataFrame,
    sql_calendar: pd.DataFrame,
    start_date: str,
    end_date: str,
):

    date_df = create_cal("2022-01-01", "2024-12-31")
    day_events, week_events, flag_vars = create_events(sheet_calendar)

    date_features = get_date_properties(
        date_df.date_, event_calendar=sql_calendar
    ).drop(columns=["date_is_weekend"])

    date_features = date_features.rename(
        columns={"date": "snapshot_date_ist"}
    ).set_index("snapshot_date_ist")

    date_df = date_df.merge(day_events, how="left", on="date_").fillna(0)

    date_df[flag_vars] = date_df[flag_vars].fillna(0).apply(to_int)

    rem_cols = [
        "weekday",
        "year_week",
        "week_num",
        "bi_week",
        "month",
        "quarter",
        "bi_week_27",
        "month_12",
        "quarter_4",
        "week_start_date",
    ]
    date_df.drop(rem_cols, axis=1, inplace=True, errors="ignore")
    date_df.columns = [f"date_{c}" if c != "date_" else c for c in date_df.columns]
    # date_df.head()

    _fragments = []
    for col in date_df.columns:
        if col.startswith("date_flag_"):
            _fragments.append(
                festive_runup_rundown(
                    col.split("date_flag_")[1],
                    date_df.loc[date_df[col] == 1, "date_"].values,
                )
            )

    festival_runupdown = reduce(
        lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how="outer"),
        _fragments,
    )

    date_df = pd.merge(
        date_df.astype({"date_": str}),
        festival_runupdown,
        left_on="date_",
        right_index=True,
        how="left",
    ).fillna(0)

    date_features = pd.merge(
        date_features.reset_index()
        .rename(columns={"snapshot_date_ist": "date_"})
        .astype({"date_": str}),
        date_df.rename(columns={"date_": "date_"}).astype({"date_": str}),
        on="date_",
        how="right",
    ).fillna(0)

    to_drop_date = [
        col
        for col in date_features.columns
        if ("bi_week" in col)
        or ("month" in col)
        or ("quarter" in col)
        or ("event_count" in col)
    ]

    date_features.drop(columns=to_drop_date, inplace=True)

    return date_features


def get_date_df(config):

    calendar_df = None
    try:
        sheet_calendar_df = pb.from_sheets(
            sheetid=config.calendar_sheet_id, sheetname=config.calendar_sheet_name
        )
    except Exception as e:
        print(f"Reading calendar from sheet failed. Reading from disk")
        sheet_calendar_df = pd.read_excel(
            f"../assets/Calendar Events India", sheet_name=config.calendar_sheet_name
        ).rename(columns={"date_": "date"})

    sql_calendar = common_utils.execute_query(
        "calendar",
        query_params={
            "min_date": config.data_prep_start_date,
            "max_date": config.test_end_date,
        },
    )

    date_features_df = get_master_date_df(
        sheet_calendar_df.rename(columns={"date": "date_"}),
        sql_calendar.rename(columns={"date": "date_"}),
        start_date=config.data_prep_start_date,
        end_date=config.test_end_date,
    ).rename(columns={"date_": config.date_column})

    date_features_df = date_features_df[
        ["date", "date_is_event", "date_is_hfs", "date_event_hfs", "date_hfs_weekend"]
    ]

    single_date = pd.DataFrame(
        {
            "date": ["2024-02-21"],
            "date_is_event": [0],
            "date_is_hfs": [0],
            "date_event_hfs": [0],
            "date_hfs_weekend": [0],
        }
    )

    date_features_df = pd.concat([date_features_df, single_date])

    date_features_df = date_features_df.reset_index(drop=True).sort_values("date")

    date_features_df["dow"] = pd.to_datetime(date_features_df["date"]).dt.dayofweek

    return date_features_df
