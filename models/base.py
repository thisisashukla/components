from subprocess import call
import numpy as np
import pandas as pd
from abc import ABCMeta, abstractmethod

from common import utils
from selection import model_selection


class BaseModel(metaclass=ABCMeta):
    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        name: str = None,
        log_function: callable = None,
        **kwargs,
    ):

        self.data = data
        self.features = features
        self.target = target
        self.grain_columns = grain_columns
        self.name = name
        self.log_function = log_function
        self.model = None
        self.trained = None

        for k, v in kwargs.items():
            setattr(self, k, v)

        for k, v in self.data.items():
            self.log_function(
                f"{k.upper()} data initiazed with {v.shape} shape and {v.drop_duplicates(subset = self.grain_columns).shape[0]} grain entities"
            )

        self.grain_df = self.data["train"][self.grain_columns].drop_duplicates().values

        single_value_features = [
            c for c in self.features if self.data["train"][c].nunique() == 1
        ]
        self.log_function(
            f"Removing {len(single_value_features)} features having only 1 unique value"
        )
        self.features = [c for c in self.features if not (c in single_value_features)]

        assert not (self.target in self.features), "Target Leak found in features"

    @abstractmethod
    def _initialize(self):
        """
        Mandatory method to be implemented by the developer.
        This method should return a valid object of the model.
        """

        pass

    # @abstractmethod
    def _read_model(self, path: str):
        """
        Optional method to be implemented by the developer.
        This method should return a valid object of the model after reading the file from storage.
        """

        return utils.read_pickle(path)

    @abstractmethod
    def _train(self, df: pd.DataFrame) -> None:
        """
        Mandatory method to be implemented by the developer.
        It takes one parameter which is the train df.
        This method implements the training logic for the model and saves the trained model in the self.model attribute
        """

        pass

    @abstractmethod
    def _predict(self, data: dict) -> dict:
        """
        Mandatory method to be implemented by the developer.
        This method implements the prediction logicand returns a dictionary
        with the same keys as passed in the data argument and prediction dataframes as the values
        """

        pass

    def _save_model(self, path: str) -> None:
        """
        Optional method to be implemented by the developer.
        This method implements the logic to save the model to disk at a given path.
        """

        utils.write_pickle(self.model, path)

    def initialize(self) -> None:
        """
        Method to initialize the model and add it to the class attributes.
        Trained state changes from None to False if model is initialized successfully.
        """

        model = self._initialize()

        assert (
            model is not None
        ), f"Initialization function did not return a model object"

        self.model = model
        self.trained = False

    def read_model(self, path: str) -> None:

        assert (
            self.model is not None
        ), f"Model has not been initialized. Call the intialize method first"

        model = self._read_model(path)

        assert model is not None, f"Read model function did not return a model object"

        self.model = model
        self.trained = True

    def train(self):

        assert self.trained is not None, f"Model is either not initialized or not read"

        self.log_function(f"Starting training")
        self._train(self.data["train"])
        self.trained = True

    def train_val(self):

        train_val = pd.concat(
            [self.data["train"], self.data["validation"]], axis=0
        ).reset_index(drop=True)
        self.log_function(f"Starting training")
        self._train(train_val)
        self.trained = True

    def predict(self, key: str = None, data: dict = None) -> dict:

        assert (
            self.trained
        ), f"Predictions can only be taken when the model has been trained or read from disk"
        assert (
            sum([key is None, data is None]) >= 1
        ), f"Only one out of key and data should be provided"

        predictions = {}
        if key:
            predictions = self._predict({key: self.data[key]})
            assert set(predictions.keys()) - set([key]) == set(
                []
            ), f"Prediction method did not return the same keys as the data"
        else:
            if data is None:
                data = {k: v for k, v in self.data.items()}
            predictions = self._predict(data)

            assert set(predictions.keys()) - set(data.keys()) == set(
                []
            ), f"Prediction method did not return the same keys as the data"

        for k, df in predictions.items():
            df.loc[:, "model_name"] = self.name
            if not ("model_type" in df.columns):
                df["model_type"] = self.name

        return predictions

    def save_model(self, path) -> None:

        model_path = f"{path}/{self.name}.pkl"

        try:
            self._save_model(model_path)
        except Exception as e:
            self.log_function(f"model saving failed. Exception: {e}")
            return False

        return model_path

    def tune(
        self, objective: callable, ntrials: int = 1, njobs: int = 1, **kwargs
    ) -> dict:

        tuner = model_selection.TuneModels(ntrials, njobs, **kwargs)

        tuning_results = tuner.tune_model(objective)

        return tuning_results
