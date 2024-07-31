from subprocess import call
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


def pairwise_corr_rejections(
    df: pd.DataFrame, target: str, keep_vars: list = [], threshold: int = 0.5
) -> pd.DataFrame:
    df_corr = df.corr()
    _df_corr = df_corr.drop(columns=[target], index=[target])

    _pairwise_corr = (
        _df_corr.where(np.tril(_df_corr, k=-1).astype(bool), None)
        .stack()
        .reset_index(name="pair_corr")
        .dropna(subset=["pair_corr"])
        .rename(columns={"level_0": "ft_1", "level_1": "ft_2"})
        .merge(
            df_corr[target]
            .reset_index(name="target_corr_ft_1")
            .rename(columns={"index": "ft_1"}),
            on="ft_1",
            how="left",
        )
        .merge(
            df_corr[target]
            .reset_index(name="target_corr_ft_2")
            .rename(columns={"index": "ft_2"}),
            on="ft_2",
            how="left",
        )
    )
    del df_corr, _df_corr

    relevant_set = _pairwise_corr[_pairwise_corr.pair_corr.abs() >= threshold]
    del _pairwise_corr

    relevant_set["rejected_feature"] = np.where(
        relevant_set[["target_corr_ft_1", "target_corr_ft_2"]].abs().idxmin(axis=1)
        == "target_corr_ft_1",
        relevant_set["ft_1"],
        relevant_set["ft_2"],
    )
    return set(relevant_set.rejected_feature.unique()) - set(keep_vars)
    return relevant_set


def variable_selection_random_cutoff(
    df: pd.DataFrame, target: str, keep_vars: list = [], num_random_ft=4
) -> pd.DataFrame:
    lo, hi = df[target].min(), df[target].max()
    for i in range(num_random_ft):
        _rng = np.random.default_rng()
        df[f"random_ft_{i}"] = _rng.uniform(lo, hi, df.shape[0])

    gbm = GradientBoostingRegressor(
        n_estimators=5000,
        max_features=df.shape[1] - 1,
        max_depth=10,
        min_samples_leaf=1,
        min_samples_split=2,
        subsample=0.6,
    )

    gbm.fit(df.drop(columns=[target]), df[target])

    imp_df = pd.DataFrame(
        {"ft": gbm.feature_names_in_, "importance": gbm.feature_importances_}
    )
    imp_df.sort_values("importance", ascending=False, inplace=True)
    cutoff_imp = imp_df.loc[imp_df.ft.str.startswith("random_ft"), "importance"].max()

    selection_df = imp_df.loc[
        (imp_df.importance > cutoff_imp) | (imp_df.ft.isin(keep_vars))
    ]
    selection_df.loc[:, "selected_feature"] = selection_df["ft"].values
    selection_df.loc[:, "method"] = "random cutoff"

    return selection_df[["method", "selected_feature", "importance"]]


def calculate_r2(exog, exog_idx):
    t = exog.columns[exog_idx]
    model = LinearRegression()
    model.fit(exog.drop(columns=[t]), exog[t])
    return model.score(exog.drop(columns=[t]), exog[t]), model


def variable_selection_vif_cutoff(df, r2_cutoff, keep_vars):
    vif_df = pd.DataFrame({"ft": df.columns, "r2": None})

    for col_idx in tqdm(range(df.shape[1]), total=df.shape[1]):
        rsquared_val, model = calculate_r2(df, col_idx)
        vif_df.at[vif_df.ft == df.columns[col_idx], "r2"] = rsquared_val
    vif_df.sort_values("r2", inplace=True)
    return vif_df.loc[(vif_df.r2 <= r2_cutoff) | (vif_df.ft.isin(keep_vars))]


def pvalue_based_backward_elimination(
    min_pvalue_threshold: float,
    stats_model_class,
    X: pd.DataFrame,
    y: pd.Series,
    keep_vars: list[str],
    model_kwargs: dict,
    fit_kwargs: dict,
    log_msg: callable,
    fit_method: callable = None,
):

    feature_drop_list = []
    step_count = 1
    final_features = [c for c in X.columns if not (c in feature_drop_list)]
    while True:
        model = stats_model_class(y, X[final_features], **model_kwargs)
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
        if len(non_significant_features) == 0 or len(feature_drop_list) == X.shape[1]:
            break
        else:
            step_count += 1

    log_msg(
        f"Iterative feature removal using minimum p-value threhsold of {min_pvalue_threshold} complete at step {step_count}. Features removed: {len(feature_drop_list)} for {model_class.__name__} with kwargs {kwargs} and fit_kwargs {fit_kwargs}"
    )

    final_features = list(set(final_features + keep_vars))
    model = stats_model_class(y, X[final_features], **model_kwargs)
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
