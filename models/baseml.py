import os
import re
import inspect
import binascii
import pandas as pd
from common import utils
from typing import Optional
from selection import tuning
from datetime import datetime
from typing import Optional, Callable
from abc import ABCMeta, abstractmethod


class BaseMLModel(metaclass=ABCMeta):
    """
    Abstract base class for a machine learning model.

    Attributes:
        data (dict): Dictionary containing datasets (train, validation, test).
        features (list[str]): List of feature columns to be used in the model.
        target (str): The target column to be predicted.
        grain_columns (list[str]): List of columns to group by.
        name (str): The name of the model instance.
        model (Optional): The model object after initialization.
        trained (Optional[bool]): Boolean indicating whether the model is trained.
        log_function (Callable): Function to handle logging, default is print.
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the BaseMLModel with the provided data, features, target, and other settings.

        Args:
            data (dict): Dictionary containing datasets (train, validation, test).
            features (list[str]): List of feature columns.
            target (str): The target column.
            grain_columns (list[str]): Columns for grain-level grouping (e.g., customer_id, product_id).
            name (str, optional): Custom name for the model. Defaults to None (auto-generated).
            **kwargs: Additional keyword arguments to be passed and set as attributes.
        """
        self.data = data
        self.features = features
        self.target = target
        self.grain_columns = grain_columns
        self.model_type = re.sub("'|>", "", str(self.__class__).split(".")[-1])

        if name is None:
            self.name = f"{self.model_type}_{utils.localize_ts(datetime.now())}_{binascii.b2a_hex(os.urandom(15)).decode('utf-8')[:5]}"
        else:
            self.name = name

        self.model = None
        self.trained = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        if not "log_function" in kwargs.keys():
            self.log_function = print

        # Extract grain-level information from the training dataset
        self.grain_df = self.data["train"][self.grain_columns].drop_duplicates().values

        # Remove features that have a single unique value to avoid leakage
        single_value_features = [
            c for c in self.features if self.data["train"][c].nunique() == 1
        ]
        self.features = [c for c in self.features if not (c in single_value_features)]

        # Ensure target column is not part of the feature columns
        assert not (self.target in self.features), "Target Leak found in features"

    @abstractmethod
    def _initialize(self):
        """
        Abstract method that must be implemented by subclasses to return a valid model object.
        """
        pass

    def _read_model(self, path: str):
        """
        Read and return the model from a file.

        Args:
            path (str): The path to the file where the model is saved.

        Returns:
            object: The model object after being read from the file.
        """
        return utils.read_pickle(path)

    @abstractmethod
    def _train(self, df: pd.DataFrame) -> None:
        """
        Abstract method to implement training logic.

        Args:
            df (pd.DataFrame): The training dataset.
        """
        pass

    @abstractmethod
    def _predict(self, data: dict) -> dict:
        """
        Abstract method to implement prediction logic.

        Args:
            data (dict): Dictionary containing data splits as keys (e.g., 'train', 'test') and DataFrames as values.

        Returns:
            dict: Dictionary with keys matching the input data and values being the prediction DataFrames.
        """
        pass

    def _save_model(self, path: str) -> None:
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.
        """
        utils.write_pickle(self.model, path)

    def initialize(self) -> None:
        """
        Initialize the model by calling the subclass's _initialize method.
        Changes the 'trained' attribute to False once the model is initialized.
        """
        model = self._initialize()

        assert (
            model is not None
        ), "Initialization function did not return a model object"

        self.model = model
        self.trained = False

    def read_model(self, path: str) -> None:
        """
        Read the model from a file and set it to the model attribute.

        Args:
            path (str): The path to the model file.

        Raises:
            AssertionError: If the model is not initialized or reading the model fails.
        """
        assert (
            self.model is not None
        ), "Model has not been initialized. Call the initialize method first"

        model = self._read_model(path)

        assert model is not None, "Read model function did not return a model object"

        self.model = model
        self.trained = True

    def train(self) -> None:
        """
        Train the model using the training dataset.

        Raises:
            AssertionError: If the model is not initialized or read.
        """
        assert self.trained is not None, "Model is either not initialized or not read"

        self._train(self.data["train"])
        self.trained = True

    def predict(self, key: str = None, data: dict = None) -> dict:
        """
        Generate predictions using the trained model.

        Args:
            key (str, optional): Specific key for a data split (e.g., 'test'). Defaults to None.
            data (dict, optional): A dictionary of datasets to predict on. Defaults to None.

        Returns:
            dict: A dictionary with predictions, where the keys match the input keys and values are DataFrames.

        Raises:
            AssertionError: If the model is not trained or key and data arguments are incorrectly provided.
        """
        assert (
            self.trained
        ), "Predictions can only be taken when the model has been trained or read from disk"
        assert (
            sum([key is None, data is None]) >= 1
        ), "Only one out of key and data should be provided"

        predictions = {}
        if key:
            predictions = self._predict({key: self.data[key]})
            assert set(predictions.keys()) - set([key]) == set(
                []
            ), "Prediction method did not return the same keys as the data"
        else:
            if data is None:
                data = {k: v for k, v in self.data.items()}
            predictions = self._predict(data)

            assert set(predictions.keys()) - set(data.keys()) == set(
                []
            ), "Prediction method did not return the same keys as the data"

        # Add metadata to prediction DataFrames
        for k, df in predictions.items():
            df.loc[:, "model_name"] = self.name
            if "model_type" not in df.columns:
                df["model_type"] = self.name

        return predictions

    def save_model(self, path: str) -> Optional[str]:
        """
        Save the trained model to disk.

        Args:
            path (str): Directory path where the model should be saved.

        Returns:
            Optional[str]: The full path where the model was saved, or None if saving fails.
        """
        model_path = f"{path}/{self.name}.pkl"

        try:
            self._save_model(model_path)
        except Exception as e:
            self.log_function(f"Model saving failed. Exception: {e}")
            return None

        return model_path

    def tune(
        self,
        objective: callable,
        metric: str,
        ntrials: int = 1,
        njobs: int = 1,
        **kwargs,
    ) -> dict:
        """
        Tune model hyperparameters using a tuning library.

        Args:
            objective (callable): The objective function to optimize during tuning.
            metric (str): Metric to evaluate performance.
            ntrials (int, optional): Number of tuning trials. Defaults to 1.
            njobs (int, optional): Number of parallel jobs during tuning. Defaults to 1.
            **kwargs: Additional tuning arguments.

        Returns:
            dict: Results of the tuning process including best parameters and best model.
        """
        tuner = tuning.TuneModels(
            ntrials,
            njobs,
            model=self.__class__,
            data=self.data,
            features=self.features,
            target=self.target,
            grain_columns=self.grain_columns,
            **kwargs,
        )

        self.tuning_results = tuner.tune_model(objective, metric=metric)

        # Update model with best parameters from tuning
        for param, best_value in self.tuning_results["best_params"].items():
            setattr(self, param, best_value)

        # Set the best model from tuning
        setattr(self, "model", self.tuning_results["best_model"].model)
        self.trained = True

        return self.tuning_results

    def train_val(self) -> object:
        """
        Combines the 'train' and 'validation' datasets to create a new model trained on the combined data.

        This method replicates the current model by retrieving its initialization parameters via introspection.
        It combines the 'train' and 'validation' datasets into a single training set, re-initializes the model
        with these datasets, and trains the new model. The 'test' set remains unchanged.

        Returns:
            object: The newly trained model, initialized with the combined 'train' and 'validation' data.
        """

        model_args = inspect.signature(self.__class__.__init__).parameters
        train_val_model = self.__class__(
            **{
                **{k: v for k, v in self.__dict__.items() if k in model_args.keys()},
                "data": {
                    "train": pd.concat(
                        [self.data["train"], self.data["validation"]],
                        axis=0,
                    ).reset_index(drop=True),
                    "validation": self.data["validation"],
                    "test": self.data["test"],
                },
            }
        )
        train_val_model.initialize()
        train_val_model.train()

        return train_val_model
