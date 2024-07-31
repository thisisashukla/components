import pandas as pd
import numpy as np
import math


def outlier_treatment(data, columns, cutoff, ratio, impute_by="min_max"):

    df = data.copy()
    list1 = list(np.arange(0.0, 0.01, 0.001))
    list2 = list(np.arange(0.01, 0.99, 0.01))
    list3 = list(np.arange(0.99, 1.0, 0.001))
    lst = list1 + list2 + list3
    lst = [round(x, 3) for x in lst]

    for col in columns:

        a = pd.DataFrame(df[col].describe(percentiles=lst))
        a = a.iloc[4:-1]
        a["jump_prev"] = a[col] - a[col].shift(1)
        a["jump_next"] = a[col].shift(-1) - a[col]
        a = a.reset_index()

        mn = a.iloc[:20]
        minm = mn.loc[
            ((mn["jump_prev"] != 0) & (mn["jump_prev"] > cutoff * mn["jump_next"]))
            | (((mn["jump_prev"] / mn[col]) > ratio)),
            [col, "index"],
        ]
        try:
            minm_perc = minm.iloc[-1, 1]
            minm = minm.iloc[-1, 0]
        except:
            minm_perc = "0%"
            minm = df[col].min()
        minm = df[col].min() if math.isnan(minm) else minm

        mx = a.iloc[100:]
        maxm = mx.loc[
            ((mx["jump_prev"] != 0) & (cutoff * mx["jump_prev"] < mx["jump_next"]))
            | ((mx["jump_next"] / mx[col]) > ratio),
            [col, "index"],
        ]
        try:
            maxm_perc = maxm.iloc[0, 1]
            maxm = maxm.iloc[0, 0]
        except:
            maxm_perc = "100%"
            maxm = df[col].max()
        maxm = df[col].max() if math.isnan(maxm) else maxm

        rows_affected = len(df.loc[df[col] < minm, :].index) + len(
            df.loc[df[col] > maxm, :].index
        )

        if impute_by == "median":
            print(
                f"Column: {col}. {rows_affected} rows affected out of {len(df.index)}. Minimum : {minm} {minm_perc}, Maximum : {maxm} {maxm_perc}. "
            )
            df[col] = np.where(df[col] < minm, df[col].median(), df[col])
            df[col] = np.where(df[col] > maxm, df[col].median(), df[col])

        elif impute_by == "min_max":
            print(
                f"Column: {col}. {rows_affected} rows affected out of {len(df.index)}. Minimum : {minm} {minm_perc}, Maximum : {maxm} {maxm_perc}. "
            )
            df[col] = np.where(df[col] < minm, minm, df[col])
            df[col] = np.where(df[col] > maxm, maxm, df[col])

        elif impute_by == "drop":
            print(
                f"Column: {col}. {rows_affected} rows affected out of {len(df.index)}. Minimum : {minm} {minm_perc}, Maximum : {maxm} {maxm_perc}. "
            )
            df = df[(df[col] >= minm) & (df[col] <= maxm)]

        elif impute_by == "max":
            print(
                f"Column: {col}. {rows_affected} rows affected out of {len(df.index)}. Minimum : {minm} {minm_perc}, Maximum : {maxm} {maxm_perc}. "
            )
            df[col] = np.where(df[col] > maxm, maxm, df[col])
            rows_affected = len(df.loc[df[col] > maxm, :].index)

        elif impute_by == "min":
            print(
                f"Column: {col}. {rows_affected} rows affected out of {len(df.index)}. Minimum : {minm} {minm_perc}, Maximum : {maxm} {maxm_perc}. "
            )
            df[col] = np.where(df[col] < minm, minm, df[col])
            rows_affected = len(df.loc[df[col] < minm, :].index)

    return df
