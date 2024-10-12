import gc
import optuna
import logging
from functools import partial
from typing import Callable, Dict, List, Any, Optional

from validations.generic import enforce_types

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TuneModels:
    """
    A class for hyperparameter tuning of machine learning models using Optuna.

    Attributes:
    - ntrial: Number of trials for the Optuna study.
    - njobs: Number of parallel jobs for Optuna optimization.
    - log_func: Function to handle logging, defaults to print.
    - kwargs: Additional keyword arguments for the objective function.
    - callbacks: List of callbacks to be executed after each trial.
    """

    def __init__(
        self,
        ntrial: int,
        njobs: int,
        log_func: Optional[Callable[[str], None]] = print,
        enable_gc: bool = True,
        custom_callbacks: Optional[List[Callable]] = None,
        **kwargs,
    ) -> None:
        """
        Initialize the TuneModels class.

        Parameters:
        - ntrial: Number of trials for the Optuna study.
        - njobs: Number of parallel jobs for Optuna optimization.
        - log_func: Function to handle logging, defaults to Python's logging.info.
        - enable_gc: Boolean flag to enable or disable garbage collection.
        - custom_callbacks: List of custom callbacks to add to the study.
        - kwargs: Additional keyword arguments for the objective function.
        """
        self.ntrial = ntrial
        self.njobs = njobs
        self.log_func = log_func or logging.info
        self.kwargs = kwargs
        self.enable_gc = enable_gc

        # Setup callbacks
        self.callbacks = custom_callbacks if custom_callbacks else []
        if self.enable_gc:
            self.callbacks.append(lambda study, trial: gc.collect())

    def tune_model(
        self, objective_func: Callable[..., float], metric, direction: str = "minimize"
    ) -> Dict[str, Any]:
        """
        Perform model tuning using the provided objective function.

        Parameters:
        - objective_func: The objective function to optimize.
        - direction: Direction for optimization ("minimize" or "maximize").

        Returns:
        - Dictionary containing the best metric value, best parameters, and tuning grain.
        """
        # Validate direction input
        if direction not in ["minimize", "maximize"]:
            raise ValueError("Invalid direction. Must be 'minimize' or 'maximize'.")

        # Create study and optimize
        self.log_func(
            f"Starting optimization with {self.ntrial} trials and {self.njobs} jobs."
        )
        study = optuna.create_study(direction=direction)
        objective = partial(objective_func, metric=metric, **self.kwargs)

        study.optimize(
            objective,
            n_trials=self.ntrial,
            timeout=600,
            n_jobs=self.njobs,
            callbacks=self.callbacks,
        )

        best_trial = study.best_trial

        # Prepare best trial results
        best_trial_results = {
            "best_metric_value": best_trial.value,
            "best_params": best_trial.params,
            "tuning_grain": dict(
                zip(self.kwargs.get("ts_columns", []), self.kwargs.get("ts_values", []))
            ),
        }

        # Include any user attributes in the results
        for k, v in best_trial.user_attrs.items():
            best_trial_results[k] = v

        return best_trial_results
