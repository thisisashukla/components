import os
import re
import inspect
import binascii
import traceback
import pandas as pd
from typing import Union
from datetime import datetime
from collections import defaultdict
from typing import Callable, List, Dict, Optional

from common import *


class MLHandler:
    """
    A handler class to manage machine learning model initialization, training, and prediction.

    Attributes:
        model_classes (List): A list of machine learning model classes to be instantiated.
        data (Dict[str, pd.DataFrame]): Dictionary containing datasets (train, validation, test) as pandas DataFrames.
        features (List[str]): List of feature column names.
        target (str): The target column name for prediction.
        grain_columns (List[str]): List of columns to group data by (e.g., customer_id, product_id).
        date_column (str): The date column in the dataset. Defaults to 'date'.
        preprocessor (Optional[Callable]): Function for preprocessing the data. Defaults to None.
        postprocessor (Optional[Callable]): Function for postprocessing predictions. Defaults to None.
        preprocess_on (Optional[List[str]]): List of dataset splits to preprocess (e.g., train, validation, test). Defaults to ['train', 'validation', 'test'].
        postprocess_on (Optional[List[str]]): List of dataset splits to postprocess after predictions. Defaults to ['train', 'validation', 'test'].
        log_function (Callable): A function for logging information. Defaults to the print function.
        model_kwargs (Optional[Dict]): Optional dictionary of model-specific keyword arguments. Defaults to an empty dict.
        tuning_kwargs (Optional[Dict]): Optional dictionary of model-specific tuning keyword arguments. Defaults to an empty dict.
        verbose (int): Controls the level of verbosity. 1 for verbose, 0 for silent. Defaults to 1.
    """

    def __init__(
        self,
        model_classes: List,
        data: Dict[str, pd.DataFrame],
        features: List[str],
        target: str,
        grain_columns: List[str],
        date_column: str = "date",
        preprocessor: Optional[Callable] = None,
        postprocessor: Optional[Callable] = None,
        preprocess_on: Optional[List[str]] = ["train", "validation", "test"],
        postprocess_on: Optional[List[str]] = ["train", "validation", "test"],
        log_function: Callable = print,
        model_kwargs: Optional[Dict] = {},
        tuning_kwargs: Optional[Dict] = {},
        verbose: int = 1,
    ) -> None:
        """
        Initialize the MLHandler class with model classes, data, and configuration parameters.

        Args:
            model_classes (List): List of machine learning model classes.
            data (Dict[str, pd.DataFrame]): Datasets, where the keys are 'train', 'validation', 'test'.
            features (List[str]): List of feature column names.
            target (str): The target column name.
            grain_columns (List[str]): Columns to group by.
            date_column (str, optional): Date column name. Defaults to "date".
            preprocessor (Optional[Callable], optional): Function for preprocessing the data. Defaults to None.
            postprocessor (Optional[Callable], optional): Function for postprocessing predictions. Defaults to None.
            preprocess_on (Optional[List[str]], optional): Datasets to preprocess. Defaults to ["train", "validation", "test"].
            postprocess_on (Optional[List[str]], optional): Datasets to postprocess. Defaults to ["train", "validation", "test"].
            log_function (Callable, optional): Function to handle logging. Defaults to print.
            model_kwargs (Optional[Dict], optional): Model-specific keyword arguments. Defaults to an empty dict.
            tuning_kwargs (Optional[Dict], optional): Tuning-specific keyword arguments. Defaults to an empty dict.
            verbose (int, optional): Verbosity level. Defaults to 1.
        """

        self.model_classes = model_classes
        self.data = data
        self.features = features
        self.target = target
        self.grain_columns = grain_columns
        self.date_column = date_column
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.preprocess_on = preprocess_on
        self.postprocess_on = postprocess_on
        self.ml_log = log_function
        self.verbose = verbose
        self.model_kwargs = model_kwargs
        self.tuning_kwargs = {
            re.sub("'|>", "", str(k)).split(".")[-1]: v
            for k, v in tuning_kwargs.items()
        }

        if self.verbose == 0:
            self.ml_log = self.dummy_log

        self.dataset = {}
        for k, df in data.items():
            if self.preprocessor and k in self.preprocess_on:
                self.dataset[k] = self.preprocessor(df)
                assert all(self.dataset[k][self.target]) > 0
            else:
                self.dataset[k] = df

        self.models = []
        self.best_model = None
        self.failures = defaultdict(list)

    def dummy_log(self, msg: str) -> None:
        """
        A dummy log function to replace logging when verbose is set to 0.

        Args:
            msg (str): Message to log. Ignored.
        """

    def get_model_attribute(self, attr: str) -> Dict[str, Union[str, float, int]]:
        """
        Retrieve a specific attribute from all models.

        Args:
            attr (str): The name of the attribute to retrieve.

        Returns:
            Dict[str, Union[str, float, int]]: A dictionary where the keys are model names and the values are the attribute values.
        """

        attributes = {}
        for m in self.models:
            if attr in dict(m).keys():
                attributes[m.name] = getattr(m, attr)

        return attributes

    def initialize_models(self) -> None:
        """
        Initialize model instances using the provided model classes and keyword arguments.
        """

        for i, model_class in enumerate(self.model_classes):
            model_type = re.sub("'|>", "", str(model_class).split(".")[-1])
            model_name = f"{model_type}_{utils.localize_ts(datetime.now())}_{binascii.b2a_hex(os.urandom(15)).decode('utf-8')[:5]}"

            model_kwarg = (
                {}
                if model_type not in self.model_kwargs.keys()
                else self.model_kwargs[model_type]
            )

            model_variants = []
            if isinstance(model_kwarg, list):
                for j, kwarg in enumerate(model_kwarg):
                    try:
                        model_variant = model_class(
                            data=self.dataset,
                            features=self.features,
                            target=self.target,
                            grain_columns=self.grain_columns,
                            name=f"{model_name}_{j}",
                            **kwarg,
                        )

                        # instantiating model object
                        model_variant.initialize()
                        model_variants.append(model_variant)
                    except Exception as e:
                        self.failures["init"].append((e, traceback.format_exc()))
            else:
                try:
                    model_variant = model_class(
                        data=self.dataset,
                        features=self.features,
                        target=self.target,
                        grain_columns=self.grain_columns,
                        name=model_name,
                        **model_kwarg,
                    )
                    model_variant.initialize()
                    model_variants.append(model_variant)
                except Exception as e:
                    self.failures["init"].append((e, traceback.format_exc()))

            self.models.extend(model_variants)

    def train(self) -> None:
        """
        Train all initialized models using the provided training dataset.
        """

        # if self.models is empty then there are not models to be trained
        assert len(self.models) > 0, f"No models initialized"

        for model in self.models:
            model_class = re.sub("'|>", "", str(model.__class__)).split(".")[-1]
            try:
                if model_class in self.tuning_kwargs:
                    model.tune(**self.tuning_kwargs[model_class])
                else:
                    model.train()
            except Exception as e:
                self.failures["train"].append((e, traceback.format_exc()))

    def predict(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Generate predictions for the dataset using all models.

        Args:
            data (Optional[Dict[str, pd.DataFrame]], optional): The dataset to predict on. Defaults to None (uses internal dataset).

        Returns:
            pd.DataFrame: A DataFrame with predictions and associated metadata.
        """

        if data is None:
            data = self.dataset

        predictions = []
        for model in self.models:
            try:
                model_predictions = model.predict(data=data)
                for k, df in model_predictions.items():
                    if k in self.postprocess_on and self.postprocessor:
                        df = self.postprocessor(df)

                    df.loc[:, "data_type"] = k
                    df.loc[:, "model_name"] = str(model.name)
                    df.loc[:, "df_model_class"] = str(model.__class__)
                    df.loc[:, "model_type"] = model.model_type
                    predictions.append(df)
            except Exception as e:
                self.failures["prediction"].append((e, traceback.format_exc()))

        if len(predictions) == 0:
            raise "All model predictions failed"

        pred_df = pd.concat(predictions, axis=0).reset_index(drop=True)

        return pred_df

    def read_data_samples(
        self,
        data_type: str,
        sample_size: Union[int, float],
        nsamples: int = 1,
    ) -> List[pd.DataFrame]:
        """
        Generate random samples from the dataset for a given data type.

        Args:
            data_type (str): The dataset split ('train', 'validation', or 'test') to sample from.
            sample_size (Union[int, float]): The number or proportion of samples to draw.
            nsamples (int, optional): The number of samples to draw. Defaults to 1.

        Returns:
            List[pd.DataFrame]: A list of sampled DataFrames.
        """

        sample_rows = 0
        if isinstance(sample_size, int):
            sample_rows = sample_size
        else:
            sample_rows = int(self.dataset[data_type].shape[0] * sample_size)

        samples = []
        for i in range(nsamples):
            samples.append(self.dataset[data_type].sample(sample_rows))

        return samples

    def select_best_model(
        self, selector: Callable, use_samples: bool = False
    ) -> object:
        """
        Select the best model based on a custom selection function.

        Args:
            selector (Callable): A selection function to choose the best model.
            use_samples (bool, optional): Whether to use samples for validation. Defaults to False.

        Returns:
            Optional: The best model instance, if found.
        """

        assert all(
            [m.trained for m in self.models]
        ), f"All models should be trained for best model selection. Run train method first"

        pred_df = self.predict()

        train = (
            self.dataset["train"]
            .assign(data_type="train")
            .rename(columns={self.target: "true"})
        )
        validation = (
            self.dataset["validation"]
            .assign(data_type="validation")
            .rename(columns={self.target: "true"})
        )

        train_val = pd.concat([train, validation], axis=0).reset_index(drop=True)

        concats = [train_val]
        if use_samples:
            samples = self.read_data_samples("validation", 0.8, 10)
            sample_preds = self.predict(
                data={
                    f"validation_sample_{k+1}": v.assign(
                        data_type=f"validation_sample_{k+1}"
                    )
                    for k, v in enumerate(samples)
                }
            )
            for i, sample in enumerate(samples):
                concats.append(
                    sample.assign(data_type=f"validation_sample_{i+1}").rename(
                        columns={self.target: "true"}
                    )
                )

            train_val = pd.concat(concats, axis=0).reset_index(drop=True)
            pred_df = pd.concat([pred_df, sample_preds], axis=0).reset_index(drop=True)

        combined = train_val.merge(pred_df)

        selector.fit({k: v for k, v in combined.groupby("data_type")})

        for model in self.models:
            if selector.model_selection in str(model.__class__):
                model_args = inspect.signature(model.__class__.__init__).parameters
                best_model = model.__class__(
                    **{
                        **{
                            k: v
                            for k, v in model.__dict__.items()
                            if k in model_args.keys()
                        },
                        "data": {
                            "train": pd.concat(
                                [self.dataset["train"], self.dataset["validation"]],
                                axis=0,
                            ).reset_index(drop=True),
                            "validation": self.dataset["validation"],
                            "test": self.dataset["test"],
                        },
                    }
                )
                self.best_model = best_model
                self.best_model.initialize()
                self.best_model.train()

        return self.best_model

    def best_prediction(self) -> pd.DataFrame:
        """
        Generate predictions from the best model.

        Returns:
            pd.DataFrame: The predictions generated by the best model.

        Raises:
            AssertionError: If no best model has been selected.
        """

        assert not (self.best_model is None), "Run select_best_model method first"

        pred = self.best_model.predict()
        for k, df in pred.items():
            if k in self.postprocess_on and self.postprocessor:
                pred[k] = self.postprocessor(df)

        return pred
