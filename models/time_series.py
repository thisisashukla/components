import pandas as pd
from prophet import Prophet
from models.baseml import BaseMLModel


class ProphetModel(BaseMLModel):
    """
    A class that integrates Facebook Prophet with BaseMLModel for time series forecasting.

    This class allows using additional regressors and custom seasonality settings
    for making predictions based on time series data.

    Attributes:
        weekly_seasonality (bool): Whether to include weekly seasonality. Defaults to False.
        yearly_seasonality (bool): Whether to include yearly seasonality. Defaults to False.
        use_holidays (bool): Whether to include holiday effects in the model. Defaults to False.
        holidays (pd.DataFrame, optional): DataFrame containing holiday information. Defaults to None.
        date_column (str): The name of the date column in the dataset. Defaults to 'date'.
    """

    def __init__(
        self,
        data: dict,
        features: list[str],
        target: str,
        grain_columns: list[str],
        weekly_seasonality: bool = False,
        yearly_seasonality: bool = False,
        use_holidays: bool = False,
        holidays: pd.DataFrame = None,
        date_column: str = "date",
        **kwargs,
    ) -> None:
        """
        Initialize the ProphetModel with the provided data, features, target, and seasonality settings.

        Args:
            data (dict): Dictionary of datasets, typically including 'train', 'validation', 'test'.
            features (list[str]): List of feature column names to be used as regressors.
            target (str): The target column name for predictions.
            grain_columns (list[str]): List of columns for grain-level grouping.
            weekly_seasonality (bool, optional): Whether to include weekly seasonality in the model. Defaults to False.
            yearly_seasonality (bool, optional): Whether to include yearly seasonality in the model. Defaults to False.
            use_holidays (bool, optional): Whether to include holiday effects. Defaults to False.
            holidays (pd.DataFrame, optional): DataFrame of holidays for holiday effects. Defaults to None.
            date_column (str, optional): The name of the date column. Defaults to "date".
            **kwargs: Additional keyword arguments passed to the parent class (BaseMLModel).
        """
        super().__init__(
            data=data,
            features=features,
            target=target,
            grain_columns=grain_columns,
            date_column=date_column,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            use_holidays=use_holidays,
            holidays=holidays,
            **kwargs,
        )

    def _initialize(self):
        """
        Initializes the Prophet model and adds any regressors specified in the features list.

        Returns:
            Prophet: An instance of the initialized Prophet model.
        """
        # Initialize the Prophet model with specified seasonality settings
        model = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )

        # Add regressors for each feature specified
        for reg in self.features:
            model.add_regressor(reg)

        return model

    def _train(self, df: pd.DataFrame) -> None:
        """
        Trains the Prophet model using the provided DataFrame.

        Args:
            df (pd.DataFrame): The training dataset containing the date, target, and regressor columns.

        Returns:
            None
        """
        # Prepare the data for Prophet by renaming columns as required ('ds' for date and 'y' for target)
        X = df[self.features + [self.date_column, self.target]].rename(
            columns={self.date_column: "ds", self.target: "y"}
        )

        # Fit the Prophet model
        self.model.fit(X)

    def _predict(self, data: dict) -> dict:
        """
        Generates predictions using the trained Prophet model for the provided datasets.

        Args:
            data (dict): Dictionary containing datasets (e.g., 'validation', 'test') with features and date.

        Returns:
            dict: A dictionary where keys are dataset names (e.g., 'validation') and values are DataFrames containing predictions.
        """
        predictions = {}

        for k, df in data.items():
            # Prepare the data for prediction by renaming columns as required by Prophet ('ds' for date)
            X = df[self.features + [self.date_column]].rename(
                columns={self.date_column: "ds", self.target: "y"}
            )

            # Generate predictions
            pred = self.model.predict(X)

            # Extract relevant columns ('ds' as date and 'yhat' as prediction)
            pred = pred[["ds", "yhat"]].rename(
                columns={"ds": self.date_column, "yhat": "prediction"}
            )

            # Format the date column
            pred[self.date_column] = pd.to_datetime(pred[self.date_column]).dt.strftime(
                "%Y-%m-%d"
            )

            # Add model type metadata
            pred["model_type"] = self.model_type

            # Merge predictions with grain columns and date column from the original dataset
            pred = pred.merge(df[self.grain_columns + [self.date_column]])

            predictions[k] = pred

        return predictions
