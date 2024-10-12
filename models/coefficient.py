import os
import re
import binascii
import pandas as pd
import statsmodels.api as sm
from datetime import datetime
from abc import ABCMeta, abstractmethod

pd.options.mode.chained_assignment = None

from common import utils
from models.baseml import BaseMLModel
from selection.variable_selection import PValueBackwardEliminationSelector


class CoefficientModel(BaseMLModel):
    """
    A base class for coefficient-based models that inherit from BaseMLModel.
    It handles features, target variables, grain columns, and model initialization logic.

    Attributes:
        model_class (ABCMeta): The class of the model to be used.
        model_params (dict): Parameters for initializing the model.
        force_features (bool): Whether to force the use of the provided features.
        keep_vars (list[str]): List of variables to keep in the final model.
        fit_params (dict): Parameters for fitting the model.
        min_pvalue_threshold (float): Threshold for removing variables based on p-values.
        add_constant (bool): Whether to add a constant to the features.
        final_features (list[str]): List of final selected features after variable selection.
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        model_class: ABCMeta,
        model_params: dict,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = False,
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the CoefficientModel with the necessary data and configuration.

        Args:
            data (dict): A dictionary of datasets (train, validation, test).
            features (list[str]): List of feature column names.
            target (str): The target column name.
            grain_columns (list[str]): Columns that identify unique grain-level entities.
            model_class (ABCMeta): Class of the model (e.g., from statsmodels or sklearn).
            model_params (dict): Parameters for the model class.
            force_features (bool, optional): Whether to force using the provided features. Defaults to False.
            keep_vars (list[str], optional): Variables to always keep in the model. Defaults to an empty list.
            fit_params (dict, optional): Parameters for fitting the model. Defaults to an empty dict.
            min_pvalue_threshold (float, optional): Minimum p-value threshold for feature selection. Defaults to 0.1.
            add_constant (bool, optional): Whether to add a constant to the features. Defaults to False.
            name (str, optional): Custom name for the model. Defaults to None (auto-generated).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            **kwargs,
        )

        self.model_class = model_class
        self.model_params = model_params
        self.force_features = force_features
        self.keep_vars = keep_vars
        self.fit_params = fit_params
        self.min_pvalue_threshold = min_pvalue_threshold
        self.add_constant = add_constant
        self.final_features = self.features

        self.name = f"CoefficientModel_{utils.localize_ts(datetime.now())}_{binascii.b2a_hex(os.urandom(15)).decode('utf-8')[:5]}"

        self.dataset = {}
        for k, df in self.data.items():
            self.dataset[k] = self.get_dataset(df, k != "test")

    def get_dataset(
        self, df: pd.DataFrame, target: bool = False, features: list[str] = None
    ) -> tuple:
        """
        Prepares the dataset by extracting features and optionally the target variable.

        Args:
            df (pd.DataFrame): The input DataFrame.
            target (bool, optional): Whether to include the target column. Defaults to False.
            features (list[str], optional): List of feature columns. Defaults to None.

        Returns:
            tuple: A tuple containing the feature DataFrame and optionally the target series.
        """

        if features is None:
            features = self.final_features

        if self.add_constant:
            X = sm.add_constant(df, has_constant="add")
        else:
            X = df

        X = X[features]

        if target:
            return X, df[self.target]
        else:
            return X, None

    @abstractmethod
    def _initialize(self) -> None:

        pass

    @abstractmethod
    def _train(self, df: pd.DataFrame) -> None:

        pass

    @abstractmethod
    def _predict(self, data: dict) -> pd.DataFrame:

        pass


class CoefficientStatsModel(CoefficientModel):
    """
    A model class based on statsmodels to handle coefficient-based models and feature selection.

    Attributes:
        variable_selector (PValueBackwardEliminationSelector): Selector for removing variables based on p-values.
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        model_class: ABCMeta,
        model_params: dict,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        min_pvalue_threshold: float = 0.1,
        add_constant: bool = True,
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the CoefficientStatsModel with the necessary data and parameters.

        Args:
            Same as CoefficientModel, with additional statsmodel-related parameters.
        """
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=model_class,
            model_params=model_params,
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            min_pvalue_threshold=min_pvalue_threshold,
            add_constant=add_constant,
            **kwargs,
        )

        self.variable_selector = PValueBackwardEliminationSelector(
            self.target,
            self.keep_vars,
            self.model_class,
            "statsmodels",
            0.1,
            self.model_params,
            self.fit_params,
            "results_regularized" if self.fit_params else "fit",
        )

    def _initialize(self) -> None:
        """
        Initializes the statsmodel-based model and performs backward elimination if features aren't forced.
        """

        if self.force_features:
            self.final_features = self.features
        else:
            self.variable_selector.fit(
                self.dataset["train"][0], self.dataset["train"][1]
            )
            self.final_features = self.variable_selector.transform(
                self.dataset["train"][0]
            )

        assert all([x in self.final_features for x in self.keep_vars])

        X = self.dataset["train"][0][self.final_features]

        model = self.model_class(self.dataset["train"][1], X, **self.model_params)

        return model

    def _read_model(self, path: str) -> None:

        model_object = utils.read_pickle(path)
        model = model_object["model"]
        self.results = model_object["results"]
        self.results_regularized = model_object["results_regularized"]

        return model

    def _train(self, df: pd.DataFrame) -> None:

        if self.fit_params:
            results_regularized = self.model.fit_regularized(**self.fit_params)
            results = self.model.fit(params=results_regularized.params)
        else:
            results = self.model.fit()
            results_regularized = None

        self.results = results
        self.results_regularized = results_regularized

    def _predict(self, data: dict) -> dict:

        dataset = {}
        for k, df in data.items():
            data_x, _ = self.get_dataset(df, features=self.final_features)
            dataset[k] = data_x

        predictions = {}
        for k, df in dataset.items():
            if self.fit_params:
                pred = self.model.predict(self.results_regularized.params, df)
            else:
                pred = self.model.predict(self.results.params, df)

            preddf = data[k][self.grain_columns + [self.date_column]]
            preddf.loc[:, "prediction"] = pred
            preddf.loc[:, "model_type"] = [
                re.sub("'|>", "", str(self.__class__).split(".")[-1])
            ] * preddf.shape[0]
            predictions[k] = preddf

        return predictions

    def _save_model(self, path):

        utils.write_pickle(
            {
                "model": self.model,
                "results": self.results,
                "results_regularized": self.results_regularized,
            },
            path,
        )

    def tune(
        self,
        objective: callable,
        metric: callable,
        ntrials: int = 1,
        njobs: int = 1,
        **kwargs,
    ) -> dict:

        super().tune(objective, metric, ntrials, njobs, **kwargs)

        setattr(self, "results", self.tuning_results["best_model"].results)
        setattr(
            self,
            "results_regularized",
            self.tuning_results["best_model"].results_regularized,
        )
        setattr(
            self,
            "final_features",
            self.tuning_results["final_features"],
        )

        return self.tuning_results

    def profile(self):

        profile_df = (
            pd.DataFrame(
                {
                    "pvalue": self.results.pvalues,
                    "coefficient": (
                        self.results.params
                        if self.results_regularized is None
                        else self.results_regularized.params
                    ),
                }
            )
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        profile_df["model_type"] = [self.model_type] * profile_df.shape[0]
        profile_df["model_name"] = self.name

        return profile_df


class CoefficientSklearnModel(CoefficientModel):
    """
    A model class based on scikit-learn for handling coefficient-based models.

    This class inherits from the CoefficientModel class and provides implementations
    for initializing, training, predicting, and tuning models using scikit-learn.

    Attributes:
        model_class (ABCMeta): The class of the scikit-learn model to be used.
        model_params (dict): Parameters for initializing the scikit-learn model.
        force_features (bool): Whether to force the use of the provided features.
        keep_vars (list[str]): List of variables to keep in the final model.
        fit_params (dict): Parameters for fitting the model.
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        model_class: ABCMeta,
        model_params: dict,
        force_features: bool = False,
        keep_vars: list[str] = [],
        fit_params: dict = {},
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the CoefficientSklearnModel with the necessary data and parameters.

        Args:
            data (dict): A dictionary of datasets (train, validation, test).
            features (list[str]): List of feature column names.
            target (str): The target column name.
            grain_columns (list[str]): Columns that identify unique grain-level entities.
            model_class (ABCMeta): Class of the scikit-learn model (e.g., LinearRegression).
            model_params (dict): Parameters for the scikit-learn model.
            force_features (bool, optional): Whether to force using the provided features. Defaults to False.
            keep_vars (list[str], optional): Variables to always keep in the model. Defaults to an empty list.
            fit_params (dict, optional): Parameters for fitting the model. Defaults to an empty dict.
            name (str, optional): Custom name for the model. Defaults to None (auto-generated).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            name=name,
            model_class=model_class,
            model_params=model_params,
            force_features=force_features,
            keep_vars=keep_vars,
            fit_params=fit_params,
            **kwargs,
        )

    def _initialize(self) -> None:
        """
        Initializes the scikit-learn model and sets the final features for the model.

        Returns:
            None
        """
        self.final_features = self.features

        # Initialize the model class with provided parameters
        model = self.model_class(**self.model_params)

        return model

    def _train(self, df: pd.DataFrame) -> None:
        """
        Train the scikit-learn model using the provided training dataset.

        Args:
            df (pd.DataFrame): The training dataset.

        Returns:
            None
        """
        # Get feature matrix (X) and target vector (y) from the dataset
        X, y = self.get_dataset(df, True)

        # Train the model with the training data
        self.model.fit(X, y, **self.fit_params)

    def _predict(self, data: dict) -> pd.DataFrame:
        """
        Generate predictions for the provided data using the trained scikit-learn model.

        Args:
            data (dict): Dictionary of datasets to predict on, with dataset names as keys.

        Returns:
            pd.DataFrame: A DataFrame containing the predictions and model metadata.
        """
        dataset = {}
        for k, df in data.items():
            data_x, _ = self.get_dataset(df, features=self.final_features)
            dataset[k] = data_x

        predictions = {}
        for k, df in dataset.items():
            # Predict using the model for the provided features
            pred = self.model.predict(df[self.model.feature_names_in_])

            # Create a DataFrame with grain columns, date, and predictions
            preddf = data[k][self.grain_columns + [self.date_column]]
            preddf.loc[:, "prediction"] = pred
            preddf.loc[:, "model_type"] = [self.model_type] * preddf.shape[0]
            predictions[k] = preddf

        return predictions

    def tune(
        self,
        objective: callable,
        metric: callable,
        ntrials: int = 1,
        njobs: int = 1,
        **kwargs,
    ) -> dict:
        """
        Tune the scikit-learn model using a hyperparameter tuning framework.

        Args:
            objective (callable): The objective function for the tuning process.
            metric (callable): The evaluation metric for tuning.
            ntrials (int, optional): The number of trials for tuning. Defaults to 1.
            njobs (int, optional): The number of parallel jobs to use during tuning. Defaults to 1.
            **kwargs: Additional keyword arguments for tuning.

        Returns:
            dict: Dictionary containing the results of the tuning process.
        """
        super().tune(objective, metric, ntrials, njobs, **kwargs)

        # Set final features after tuning
        setattr(
            self,
            "final_features",
            self.tuning_results["final_features"],
        )

        return self.tuning_results

    def profile(self) -> pd.DataFrame:
        """
        Generate a DataFrame containing the coefficients and feature names for the model.

        Returns:
            pd.DataFrame: A DataFrame with the feature names, coefficients, model type, and model name.
        """
        profile_df = pd.DataFrame(
            {
                "feature": self.model.feature_names_in_,
                "coefficient": self.model.coef_,
                "model_type": [self.model_type] * len(self.model.coef_),
                "model_name": [self.name] * len(self.model.coef_),
            }
        )

        return profile_df
