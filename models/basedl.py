import torch
import pandas as pd
from torch import nn
from common import utils
from selection import tuning
from datetime import datetime
from typing import Callable
from abc import ABCMeta, abstractmethod


class BaseDLArchitecture(nn.Module):
    """
    A base deep learning architecture class that includes a backbone and a head for custom model architectures.

    Attributes:
        head (nn.Module): The head module that defines the final layers of the architecture.
        backbone (nn.Module, optional): The backbone module, typically a pre-trained model. Defaults to None.
        freeze_backbone_weights (bool): If True, the backbone weights are frozen. Defaults to True.
    """

    def __init__(
        self,
        head: nn.Module,
        backbone: nn.Module = None,
        freeze_backbone_weights: bool = True,
    ):
        """
        Initialize the BaseDLArchitecture with a head and an optional backbone.

        Args:
            head (nn.Module): The head of the architecture (final layers).
            backbone (nn.Module, optional): The backbone of the architecture (e.g., pre-trained layers). Defaults to None.
            freeze_backbone_weights (bool, optional): Whether to freeze the backbone weights. Defaults to True.
        """
        super(BaseDLArchitecture, self).__init__()

        self.head = head
        self.backbone = backbone
        self.freeze_backbone_weights = freeze_backbone_weights

        if self.freeze_backbone_weights and self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model, first through the backbone (if present), then through the head.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output of the model after passing through the head.
        """
        if self.backbone is not None:
            x = self.backbone(x)

        x = self.head(x)
        return x


class BaseDLModel(metaclass=ABCMeta):
    """
    Abstract base class for a deep learning model that manages data loading, model definition, training, and prediction.

    Attributes:
        data_loaders (dict): A dictionary containing data loaders for training, validation, and test datasets.
        head (nn.Module): The head module of the model.
        backbone (nn.Module, optional): The backbone module, typically a pre-trained model. Defaults to None.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        loss (Callable): Loss function for training the model.
        postprocess (Callable, optional): A function to post-process model outputs during prediction. Defaults to None.
        callbacks (list, optional): List of callback functions to be applied during training. Defaults to an empty list.
        name (str, optional): The name of the model. If None, a name will be auto-generated.
        device (torch.device): The device on which the model will be trained/predicted (CPU or GPU).
        log_function (Callable, optional): Function to handle logging during training. Defaults to print.
    """

    def __init__(
        self,
        data_loaders: dict,
        head: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: Callable,
        backbone: nn.Module = None,
        postprocess: Callable = None,
        callbacks: list = [],
        name: str = None,
        **kwargs,
    ):
        """
        Initialize the BaseDLModel with data loaders, architecture components, and training configurations.

        Args:
            data_loaders (dict): Dictionary containing data loaders for training, validation, and test datasets.
            head (nn.Module): Head module that defines the final layers of the model.
            optimizer (torch.optim.Optimizer): Optimizer for training the model.
            loss (Callable): Loss function for training.
            backbone (nn.Module, optional): Backbone model, typically pre-trained. Defaults to None.
            postprocess (Callable, optional): Post-processing function for model outputs during prediction. Defaults to None.
            callbacks (list, optional): List of callback functions for training. Defaults to an empty list.
            name (str, optional): Custom name for the model. Defaults to None (auto-generated).
            **kwargs: Additional keyword arguments for setting other attributes (e.g., log_function).
        """
        self.data_loaders = data_loaders
        self.head = head
        self.backbone = backbone
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.loss = loss
        self.postprocess = postprocess
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k, v in kwargs.items():
            setattr(self, k, v)

        if not "log_function" in kwargs:
            self.log_function = print

        if name is None:
            self.name = f"dlmodel_{utils.localize_ts(datetime.now())}_{binascii.b2a_hex(os.urandom(15)).decode('utf-8')[:5]}"
        else:
            self.name = name

    def _initialize(self) -> BaseDLArchitecture:
        """
        Initialize the model by creating an instance of the BaseDLArchitecture with the provided head and backbone.

        Returns:
            BaseDLArchitecture: An instance of the deep learning architecture with the given head and backbone.
        """
        model = BaseDLArchitecture(self.head, self.backbone)
        return model

    def _train(self, data_loader: torch.utils.data.DataLoader) -> None:
        """
        Train the model on the provided data loader.

        Args:
            data_loader (torch.utils.data.DataLoader): Data loader for training data.
        """
        self.model.train()

        for batch_idx, (x, y) in enumerate(data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)
            y = y.float()

            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()

    def _predict(self, data_loaders: dict) -> dict:
        """
        Generate predictions for each dataset in the data_loaders dictionary.

        Args:
            data_loaders (dict): Dictionary of data loaders for different datasets (e.g., 'train', 'test').

        Returns:
            dict: A dictionary where keys are the dataset names and values are DataFrames with predictions.
        """
        predictions = {}
        self.model.eval()

        for k, data_loader in data_loaders.items():
            indexes = []
            batch_predictions = []

            with torch.no_grad():
                for batch_idx, data in enumerate(data_loader):
                    if isinstance(data, tuple):
                        x = data[0]
                    else:
                        x = data

                    x = x.to(self.device)
                    output = self.model(x)

                    # Apply post-processing function if provided
                    batch_preds = (
                        self.postprocess(output) if self.postprocess else output
                    )

                    indexes.extend([(batch_idx, i) for i in range(len(batch_preds))])
                    batch_predictions.extend(batch_preds.cpu().numpy())

            preddf = pd.DataFrame(indexes, columns=["batch_index", "sample_index"])
            preddf.loc[:, "prediction"] = batch_predictions
            predictions[k] = preddf

        return predictions
