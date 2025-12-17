# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Module defining parameter tuning models for OpenSTEF Beam.

SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project
"""

from typing import Any, Literal, Self

from optuna.trial import Trial
from pydantic import Field, model_validator

from openstef_beam.evaluation.metric_providers import MetricProvider
from openstef_core.base_model import BaseModel
from openstef_models.models.forecasting.forecaster import HyperParams
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearHyperParams
from openstef_models.models.forecasting.lgbm_forecaster import LGBMHyperParams
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearHyperParams
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams


class OptimizationMetric(BaseModel):
    """Defines the metric used for optimization during parameter tuning."""

    metric: MetricProvider = Field(description="Name of the optimization metric.")
    direction_minimize: bool = Field(description="Indicates if the metric should be minimized.")

    @property
    def name(self) -> str:
        """Get the name of the optimization metric.

        Returns:
            The name of the metric.
        """
        name = self.metric.__class__.__name__
        name_map = {
            "RCRPSProvider": "rCRPS",
            "RCRPSSampleWeightedProvider": "rCRPS_sample_weighted",
            "RMAEProvider": "rMAE",
        }

        return name_map[name]


class Distribution(BaseModel):
    """Base class for different types of parameter distributions."""


class CategoricalDistribution(Distribution):
    """Categorical distribution for hyperparameter tuning."""

    type: Literal["categorical"] = Field(default="categorical", description="Type of distribution (categorical).")
    choices: list[Any] = Field(description="List of categorical values.")


class FloatDistribution(Distribution):
    """Float distribution for hyperparameter tuning."""

    type: Literal["float"] = Field(default="float", description="Type of distribution (float).")
    low: float = Field(description="Lower bound of the distribution.")
    high: float = Field(description="Upper bound of the distribution.")
    log: bool = Field(default=False, description="Whether to sample on a logarithmic scale.")
    step: float | None = Field(default=None, description="Step size for the distribution.")


class IntDistribution(Distribution):
    """Integer distribution for hyperparameter tuning."""

    type: Literal["int"] = Field(default="int", description="Type of distribution (int).")
    low: int = Field(description="Lower bound of the distribution.")
    high: int = Field(description="Upper bound of the distribution.")
    log: bool = Field(default=False, description="Whether to sample on a logarithmic scale.")
    step: int = Field(default=1, description="Step size for the distribution.")


DistributionOrParameter = Distribution | int | float | str | bool
FloatOrFloatDistribution = FloatDistribution | float
IntOrIntDistribution = IntDistribution | int
BoolOrCategoricalDistribution = CategoricalDistribution | bool
StrOrCategoricalDistribution = CategoricalDistribution | str


class ParameterSpace(BaseModel):
    """Defines a hyperparameter search space for tuning forecasting models."""

    def items(self) -> list[tuple[str, DistributionOrParameter]]:
        """Get an iterator over the parameter space items.

        Returns:
            An iterator over (parameter name, Distribution) tuples.
        """
        return list(self.__dict__.items())

    @model_validator(mode="before")
    @staticmethod
    def check_distribution(value: dict[str, Any]) -> dict[str, DistributionOrParameter]:
        """Validate that the value is an instance of Distribution.

        Args:
            value: The value to validate.

        Returns:
            The validated values.

        Raises:
            TypeError: If the value is not an instance of Distribution.
        """
        if not all(isinstance(v, DistributionOrParameter) for v in value.values()):
            raise TypeError("All values in ParameterSpace must be instances of Distribution.")

        return value

    @model_validator(mode="after")
    def check_parameters(self) -> Self:
        """Validate that the value is an instance of Distribution.

        Returns:
            Self: The validated ParameterSpace instance.

        Raises:
            ValueError: If a parameter is not valid for the associated forecasting model.
        """
        object_cls = self.__class__
        # Only run validate for subclasses
        if object_cls == ParameterSpace:
            return self

        default_params: HyperParams = object_cls.default_hyperparams()
        hp_class = default_params.__class__

        allowed_params: list[str] = list(hp_class.model_fields.keys())

        for param, _val in iter(self):
            if param not in allowed_params:
                msg = (
                    f"Parameter '{param}' is not valid for '{default_params.__name__}'. "
                    f"Allowed parameters are: {allowed_params}."
                )
                raise ValueError(msg)

        return self

    @classmethod
    def default_hyperparams(cls) -> HyperParams:
        """Get default hyperparameters for the associated forecaster.

        Returns:
            HyperParams instance.
        """
        raise NotImplementedError("Subclasses must implement the forecasting_model property.")

    @classmethod
    def get_preset(cls, model: str) -> "ParameterSpace":
        """Get a preset parameter space for a given model.

        Args:
            model: The name of the model to get the preset for.

        Returns:
            A ParameterSpace instance with preset hyperparameter distributions.

        Raises:
            ValueError: If the model name is not recognized.
        """
        if model not in {"lgbm", "xgboost", "gblinear", "lgbmlinear"}:
            message = """
            Parameter space model must be one of 'lgbm', 'xgboost', 'gblinear', or 'lgbmlinear'."""
            raise ValueError(message)

        if model == "lgbm":
            return LGBMParameterSpace()
        if model == "xgboost":
            return XGBoostParameterSpace()
        if model == "gblinear":
            return GBLinearParameterSpace()
        if model == "lgbmlinear":
            return LGBMLinearParameterSpace()
        raise ValueError("Unreachable code reached.")

    def make_optuna(self, trial: Trial) -> dict[str, Any]:
        """Convert the parameter space to an Optuna-compatible search space.

        Args:
            trial: An Optuna trial object.

        Returns:
            A dictionary representing the Optuna search space.

        Raises:
            TypeError: If an unsupported distribution type is encountered.
        """
        optuna_space: dict[str, str | float | int | None] = {}
        for param_name, distribution_or_value in self.items():
            if isinstance(distribution_or_value, FloatDistribution):
                optuna_space[param_name] = trial.suggest_float(
                    param_name,
                    distribution_or_value.low,
                    distribution_or_value.high,
                    log=distribution_or_value.log,
                    step=distribution_or_value.step,
                )
            elif isinstance(distribution_or_value, IntDistribution):
                optuna_space[param_name] = trial.suggest_int(
                    param_name,
                    distribution_or_value.low,
                    distribution_or_value.high,
                    log=distribution_or_value.log,
                    step=distribution_or_value.step,
                )
            elif isinstance(distribution_or_value, CategoricalDistribution):
                optuna_space[param_name] = trial.suggest_categorical(param_name, distribution_or_value.choices)
            elif isinstance(distribution_or_value, (float, int, str)):
                optuna_space[param_name] = distribution_or_value
            else:
                message = f"Unsupported type: {type(distribution_or_value)}"
                raise TypeError(message)
        return optuna_space


class LGBMParameterSpace(ParameterSpace):
    """Preset parameter space for LGBM model."""

    learning_rate: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=0.3, log=True))
    num_leaves: IntOrIntDistribution = Field(default=IntDistribution(low=20, high=150))
    max_depth: IntOrIntDistribution = Field(default=IntDistribution(low=3, high=15))
    reg_lambda: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=10.0, log=True))

    @classmethod
    def default_hyperparams(cls) -> HyperParams:
        """Get default hyperparameters for the associated forecaster.

        Returns:
            LGBMHyperParams instance.
        """
        return LGBMHyperParams()


class LGBMLinearParameterSpace(ParameterSpace):
    """Preset parameter space for LGBM Linear model."""

    n_estimators: IntOrIntDistribution = Field(default=IntDistribution(low=3, high=500))
    learning_rate: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=0.3, log=True))
    num_leaves: IntOrIntDistribution = Field(default=IntDistribution(low=3, high=150))
    max_depth: IntOrIntDistribution = Field(default=IntDistribution(low=1, high=5))
    reg_lambda: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=10.0, log=True))
    colsample_bytree: FloatOrFloatDistribution = Field(default=FloatDistribution(low=0.5, high=1.0))
    max_bin: IntOrIntDistribution = Field(default=IntDistribution(low=10, high=256))
    min_data_in_leaf: IntOrIntDistribution = Field(default=IntDistribution(low=1, high=50))
    min_data_in_bin: IntOrIntDistribution = Field(default=IntDistribution(low=1, high=50))
    min_child_weight: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=100.0, log=True))

    @classmethod
    def default_hyperparams(cls) -> HyperParams:
        """Get default hyperparameters for the associated forecaster.

        Returns:
            LGBMLinearHyperParams instance.
        """
        return LGBMLinearHyperParams()


class XGBoostParameterSpace(ParameterSpace):
    """Preset parameter space for XGBoost model."""

    learning_rate: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=0.3, log=True))
    max_depth: IntOrIntDistribution = Field(default=IntDistribution(low=2, high=15))
    n_estimators: IntOrIntDistribution = Field(default=IntDistribution(low=50, high=500))
    subsample: FloatOrFloatDistribution = Field(default=FloatDistribution(low=0.5, high=1.0))
    reg_lambda: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=10.0, log=True))

    @classmethod
    def default_hyperparams(cls) -> HyperParams:
        """Get default hyperparameters for the associated forecaster.

        Returns:
            XGBoostHyperParams instance.
        """
        return XGBoostHyperParams()


class GBLinearParameterSpace(ParameterSpace):
    """Preset parameter space for GBLinear model."""

    learning_rate: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-3, high=0.3, log=True))
    n_steps: IntOrIntDistribution = Field(default=IntDistribution(low=50, high=500))
    reg_alpha: FloatOrFloatDistribution = Field(default=FloatDistribution(low=1e-5, high=10.0, log=True))

    @classmethod
    def default_hyperparams(cls) -> HyperParams:
        """Get default hyperparameters for the associated forecaster.

        Returns:
            GBLinearHyperParams instance.
        """
        return GBLinearHyperParams()
