import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from typing import Callable, List, Optional, Dict, Any

from validations.generic import enforce_types


class MultiMetricSelection(BaseEstimator):
    """
    A class to compare results from two or more ML algorithms and select the best model
    based on a given list of metrics and weights to aggregate these metrics.
    """

    def __init__(
        self,
        metrics: List[Callable[[Any, Any], float]],
        metric_aggregator: Callable[[np.ndarray], float],
        final_metric_method: Callable[[pd.DataFrame], float],
        model_name_identifier: str,
        metric_weights: Optional[List[float]] = None,
        selection_grain: Optional[List[str]] = None,
        scale_metrics: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        """
        Initialize the MultiMetricSelection class.

        Parameters:
        - metrics: List of tuples where each tuple contains the metric name and its corresponding function.
        - metric_aggregator: Function to aggregate metrics.
        - metric_weights: Weights for each metric. Default is None.
        - selection_grain: List of columns to group by for metric calculation.
        - final_metric_method: Function to select the best model based on aggregated metrics.
        - model_name_identifier: Column name that identifies the model type.
        - scale_metrics: Callable for scaling metrics. Default is MinMaxScaler.
        """
        self.metrics = metrics
        self.metric_aggregator = metric_aggregator
        self.metric_weights = metric_weights
        self.selection_grain = selection_grain or []
        self.final_metric_method = final_metric_method
        self.model_name_identifier = model_name_identifier
        self.scale_metrics = scale_metrics

    def weighted_agg(
        self, agg_func: Callable, x: np.ndarray, w: Optional[List[float]]
    ) -> float:
        """
        Perform weighted aggregation of a list using a specified aggregation function.

        Parameters:
        - agg_func: Aggregation function.
        - x: List or array of values to aggregate.
        - w: Weights for each value in x.

        Returns:
        - Aggregated value.
        """
        if w is None:
            return agg_func(x)

        assert len(x) == len(w), "Length of weights must match length of values."
        return agg_func(x * np.array(w))

    def fit(self, X: Dict[str, pd.DataFrame], y=None) -> None:
        """
        Fit the model to the data and select the best model.

        Parameters:
        - X: Dictionary with data splits ('train', 'validation', 'test') as keys and DataFrames as values.
        - y: Ignored, for compatibility with sklearn.
        """
        assert all(
            key in X for key in ["train", "validation"]
        ), "Input data must have 'train' and 'validation' keys."

        predmetricdf = []
        for key, df in X.items():
            if key == "test":
                continue
            combined = df.copy()
            metricdf = pd.DataFrame()

            # Calculate metrics for each model
            for metric_name, metric_func in self.metrics:
                temp = (
                    combined.groupby(
                        self.selection_grain + [self.model_name_identifier],
                        as_index=False,
                    )
                    .apply(lambda x: metric_func(x["true"], x["prediction"]))
                    .rename(columns={None: metric_name})
                )
                metricdf = (
                    temp
                    if metricdf.empty
                    else metricdf.merge(
                        temp, on=self.selection_grain + [self.model_name_identifier]
                    )
                )

            metricdf["data"] = key.split("_")[0]
            metricdf["sample"] = "sample" in key

            predmetricdf.append(metricdf)

        predmetricdf = pd.concat(predmetricdf, axis=0)

        # Scale the metrics
        if self.scale_metrics:
            scaled = []
            metric_names = [m for m, _ in self.metrics]
            for _, grp in predmetricdf.groupby(self.selection_grain):
                scaler = self.scale_metrics()
                grp[metric_names] = scaler.fit_transform(grp[metric_names])
                scaled.append(grp)
            scaled_df = pd.concat(scaled)
        else:
            scaled_df = predmetricdf

        scaled_df = scaled_df.set_index(
            self.selection_grain + [self.model_name_identifier, "data", "sample"]
        )

        # Aggregate metrics using weighted function
        scaled_df["agg_metric"] = scaled_df.apply(
            lambda x: self.weighted_agg(self.metric_aggregator, x, self.metric_weights),
            axis=1,
        )

        # Group by model and aggregate metric
        scaled_df = (
            scaled_df.groupby(self.selection_grain + ["model_type", "data", "sample"])
            .agg({"agg_metric": ["mean", "std", "count"]})
            .reset_index()
        )
        scaled_df.columns = (
            self.selection_grain
            + ["model_type", "data", "sample"]
            + ["agg_metric_mean", "agg_metric_std", "agg_metric_count"]
        )

        self.predmetricdf = scaled_df

        # Select the best model based on the final metric method
        model_selection = (
            self.predmetricdf.groupby(
                self.selection_grain + [self.model_name_identifier]
            )
            .apply(self.final_metric_method)
            .reset_index()
            .rename(columns={0: "final_metric"})
        )

        self.model_selection = model_selection.loc[
            model_selection.groupby(self.selection_grain)["final_metric"].idxmin()
        ][self.model_name_identifier].values[0]

    def transform(self, X: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Transform the data to retain only the best model's predictions.

        Parameters:
        - X: Dictionary with data splits ('train', 'validation', 'test') as keys and DataFrames as values.

        Returns:
        - DataFrame with predictions from the best model.
        """
        best_preds = []
        for key, df in X.items():
            df["best_model"] = df[self.model_name_identifier] == self.model_selection
            df["data_type"] = key
            best_preds.append(df)

        return pd.concat(best_preds)
