import pandas as pd
import numpy as np
import pencilbox as pb
import datetime as dt
from tqdm.notebook import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import random
import multiprocessing as mp


def _calculate_r2(exog_x, exog_y):
    model = LinearRegression()
    model.fit(exog_x, exog_y)
    return model.score(exog_x, exog_y), model


def return_r2(df, cols_eliminated, target_col):
    r2, model = _calculate_r2(
        df.drop(cols_eliminated + [target_col], axis=1), df[target_col]
    )
    return r2


def _variable_selection_vif_cutoff(df, config):
    vif_df = []
    cols_eliminated = []

    for col in tqdm(range(df.shape[1]), total=df.shape[1]):
        if (len(cols_eliminated) > 0) and (df.columns[col] in cols_eliminated):
            continue
        r2 = return_r2(df, cols_eliminated, df.columns[col])
        if r2 > config["r2_cutoff"]:
            cols_eliminated.append(df.columns[col])
        vif_df.append((col, r2))

    vif_df = pd.DataFrame(vif_df, columns=["ft", "r2"])

    # for col_idx in tqdm(range(df.shape[1]), total=df.shape[1]):
    # rsquared_val, model = _calculate_r2(df, col_idx)
    # vif_df.at[vif_df.ft==df.columns[col_idx], "r2"] = rsquared_val

    vif_df.sort_values("r2", inplace=True)
    return (
        vif_df.loc[
            (~(vif_df["ft"].isin(cols_eliminated)))
            | (vif_df.ft.isin(config["keep_vars"]))
        ],
        cols_eliminated,
    )


def _variable_selection_random_cutoff(df, config, num_random_ft=4):
    lo, hi = df[config["target"]].min(), df[config["target"]].max()
    for i in range(num_random_ft):
        _rng = np.random.default_rng()
        df[f"random_ft_{i}"] = _rng.uniform(lo, hi, df.shape[0])

    gbm = GradientBoostingRegressor(
        n_estimators=1000,
        max_features=int(np.ceil((df.shape[1] - 1) / 3)) + 1,
        max_depth=10,
        min_samples_leaf=100,
        min_samples_split=100,
        subsample=0.6,
    )

    gbm.fit(df.drop(columns=[config["target"]]), df[config["target"]])

    imp_df = pd.DataFrame(
        {"ft": gbm.feature_names_in_, "importance": gbm.feature_importances_}
    )
    imp_df.sort_values("importance", ascending=False, inplace=True)
    cutoff_imp = imp_df.loc[imp_df.ft.str.startswith("random_ft"), "importance"].max()

    return imp_df.loc[
        (imp_df.importance > cutoff_imp) | (imp_df.ft.isin(config["keep_vars"]))
    ]


def iterative_feature_removal(
    min_pvalue_threshold,
    model_class,
    keep_vars,
    data,
    kwargs,
    fit_kwargs,
    log_msg,
    fit_method=None,
):

    feature_drop_list = []
    step_count = 1
    X = data[0]
    final_features = [c for c in X.columns if not (c in feature_drop_list)]
    while True:
        model = model_class(data[1], X[final_features], **kwargs)
        if fit_method is None:
            if not fit_kwargs:
                result = model.fit()
            else:
                result_fit_regularised = model.fit_regularized(**fit_kwargs)
                result = model.fit(params=result_fit_regularised.params)
        else:
            if fit_method == "fit":
                result = model.fit(**fit_kwargs)
            else:
                result_fit_regularised = model.fit_regularized(**fit_kwargs)
                result = model.fit(params=result_fit_regularised.params)
        pvalue_df = result.pvalues.reset_index(name="pvalue").rename(
            columns={"index": "ft"}
        )
        non_significant_features = pvalue_df.loc[
            pvalue_df.pvalue > min_pvalue_threshold, "ft"
        ].tolist()
        feature_drop_list.extend(non_significant_features)
        final_features = [c for c in X.columns if not (c in feature_drop_list)]
        if (
            len(non_significant_features) == 0
            or len(feature_drop_list) == data[0].shape[1]
        ):
            break
        else:
            step_count += 1

    log_msg(
        f"Iterative feature removal using minimum p-value threhsold of {min_pvalue_threshold} complete at step {step_count}. Features removed: {len(feature_drop_list)} for {model_class.__name__} with kwargs {kwargs} and fit_kwargs {fit_kwargs}"
    )

    final_features = list(set(final_features + keep_vars))
    model = model_class(data[1], X[final_features], **kwargs)
    if fit_method is None:
        if not fit_kwargs:
            result = model.fit()
        else:
            result_fit_regularised = model.fit_regularized(**fit_kwargs)
            result = model.fit(params=result_fit_regularised.params)
    else:
        if fit_method == "fit":
            result = model.fit(**fit_kwargs)
        else:
            result_fit_regularised = model.fit_regularized(**fit_kwargs)
            result = model.fit(params=result_fit_regularised.params)
    pvalue_df = result.pvalues.reset_index(name="pvalue").rename(
        columns={"index": "ft"}
    )

    assert all(
        pvalue_df[pvalue_df["ft"].isin(final_features)]["pvalue"]
        <= min_pvalue_threshold
    ), f"All features not below min threshold after iterative removal"

    return final_features
