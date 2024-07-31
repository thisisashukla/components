import gc
import optuna
import numpy as np
import pandas as pd
from functools import partial
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


class BestModelSelection(BaseEstimator):
    def __init__(
        self,
        selection_grain,
        metrics,
        metric_aggregator,
        final_metric_method,
        model_name_indentifier,
        metric_weights=None,
        scale_metrics=False,
    ):

        self.selection_grain = selection_grain
        self.metrics = metrics
        self.metric_aggregator = metric_aggregator
        self.model_name_indentifier = model_name_indentifier
        self.final_metric_method = final_metric_method
        self.metric_weights = metric_weights
        self.scale_metrics = scale_metrics

    def fit(self, x, y=None):
        def weighted_agg(agg_func, x, w):

            if w is None:
                return agg_func(x)

            assert len(x) == len(w)

            return agg_func(x * w)

        assert all([k in x.keys() for k in ["train", "validation"]])

        predmetricdf = []
        for k, df in x.items():
            if k == "test":
                continue
            combined = df
            metricdf = pd.DataFrame()
            for metric in self.metrics:
                temp = (
                    combined.groupby(
                        self.selection_grain + [self.model_name_indentifier],
                        as_index=False,
                    )
                    .apply(lambda x: metric[1](x["true"], x["prediction"]))
                    .rename(columns={None: metric[0]})
                )
                if metricdf.shape[0] == 0:
                    metricdf = temp
                else:
                    metricdf = metricdf.merge(temp)
            metricdf["data"] = k.split("_")[0]
            metricdf["sample"] = "sample" in k
            predmetricdf.append(metricdf)

        predmetricdf = pd.concat(predmetricdf, axis=0)

        if self.scale_metrics:
            scalled = []
            metric_names = [m for m, f in self.metrics]
            for i, grp in predmetricdf.groupby(self.selection_grain):
                mm = MinMaxScaler()
                grp[metric_names] = mm.fit_transform(grp[metric_names])
                scalled.append(grp)

            scalled_df = pd.concat(scalled)
        else:
            scalled_df = predmetricdf

        scalled_df = scalled_df.set_index(
            self.selection_grain + [self.model_name_indentifier, "data", "sample"]
        )

        scalled_df["agg_metric"] = scalled_df.apply(
            lambda x: weighted_agg(self.metric_aggregator, x, self.metric_weights),
            axis=1,
        )

        scalled_df = scalled_df.groupby(
            [
                # 'ptype',
                "model_type",
                "data",
                "sample",
            ]
        ).agg({"agg_metric": ["mean", "std", "count"]})

        scalled_df = scalled_df.groupby(
            self.selection_grain + ["model_type", "data", "sample"]
        ).agg({"agg_metric": ["mean", "std", "count"]})

        self.predmetricdf = scalled_df.reset_index()

        model_selection = (
            self.predmetricdf.groupby(
                self.selection_grain + [self.model_name_indentifier]
            )
            .apply(self.final_metric_method)
            .reset_index()
            .rename(columns={0: "final_metric"})
        )

        model_selection = model_selection.groupby(self.selection_grain).apply(
            lambda x: x[x["final_metric"] == x["final_metric"].min()]
        )

        self.model_selection = model_selection["model_type"].values[0]

    def transform(self, x):

        best_preds = []
        for k, df in x.items():
            df["best_model"] = np.where(
                df["model_type"] == self.model_selection, True, False
            )
            df["data_type"] = k
            best_preds.append(df)

        return pd.concat(best_preds)


class TuneModels:
    def __init__(self, ntrial, njobs, **kwargs):

        self.ntrial = ntrial
        self.njobs = njobs
        if "log_func" in kwargs.keys():
            self.log_func = kwargs["log_func"]
        else:
            self.log_func = print
        self.kwargs = kwargs

    def tune_model(self, objective_func, direction="minimize"):

        study = optuna.create_study(direction=direction)
        objective = partial(objective_func, **self.kwargs)
        study.optimize(
            objective,
            n_trials=self.ntrial,
            timeout=600,
            n_jobs=self.njobs,
            callbacks=[lambda study, trial: gc.collect()],
        )

        best_trial = study.best_trial

        best_trial_results = {
            "best_metric_value": best_trial.value,
            "best_params": best_trial.params,
            "tuning_grain": dict(
                zip(self.kwargs["ts_columns"], self.kwargs["ts_values"])
            ),
        }
        for k, v in best_trial.user_attrs.items():
            best_trial_results[k] = v

        return best_trial_results
