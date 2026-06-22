# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""LightGBM Linear-based forecasting models for probabilistic energy forecasting.

Provides gradient boosting tree models with linear leaves using LightGBM for
multi-quantile energy forecasting.
"""

from typing import ClassVar, override

from pydantic import Field

from openstef_models.models.forecasting.lgbm_forecaster import (
    LGBMBaseForecaster,
    LGBMHyperParamsBase,
)


class LGBMLinearHyperParams(LGBMHyperParamsBase):
    """LGBMLinear hyperparameters for gradient boosting tree models with linear leaves.

    Example:
        Creating custom hyperparameters for deep trees with regularization

        >>> hyperparams = LGBMLinearHyperParams(
        ...     n_estimators=200,
        ...     max_depth=8,
        ...     learning_rate=0.1,
        ...     reg_alpha=0.1,
        ...     reg_lambda=1.0,
        ... )

    Note:
        These parameters are optimized for probabilistic forecasting with
        quantile regression. The default objective function is specialized
        for magnitude-weighted pinball loss.
    """

    learning_rate: float = Field(
        default=0.07,
        alias="eta",
        description="Step size shrinkage used to prevent overfitting. Range: [0,1]. Lower values require "
        "more boosting rounds.",
    )
    max_depth: int = Field(
        default=6,
        description="Maximum depth of trees. Higher values capture more complex patterns but risk "
        "overfitting. Range: [1,∞]",
    )
    min_child_weight: float = Field(
        default=0.06,
        description="Minimum sum of instance weight (hessian) needed in a child. Higher values prevent "
        "overfitting. Range: [0,∞]",
    )
    min_data_in_leaf: int = Field(
        default=500,
        description="Minimum number of data points in a leaf. Higher values prevent overfitting. Range: [1,∞]",
    )
    min_data_in_bin: int = Field(
        default=500,
        description="Minimum number of data points in a bin. Higher values prevent overfitting. Range: [1,∞]",
    )
    num_leaves: int = Field(
        default=30,
        description="Maximum number of leaves. 0 means no limit. Only relevant when grow_policy='lossguide'.",
    )
    max_bin: int = Field(
        default=256,
        description="Maximum number of discrete bins for continuous features. Higher values may improve accuracy but "
        "increase memory.",
    )

    @classmethod
    def forecaster_class(cls) -> "type[LGBMLinearForecaster]":
        """Get forecaster class for these hyperparams.

        Returns:
            Forecaster class associated with this configuration.
        """
        return LGBMLinearForecaster


MODEL_CODE_VERSION = 1


class LGBMLinearForecaster(LGBMBaseForecaster):
    """LGBMLinear-based forecaster for probabilistic energy forecasting.

    Implements gradient boosting trees with linear leaves using LightGBM for
    multi-quantile forecasting. Optimized for time series prediction with specialized
    loss functions and comprehensive hyperparameter control suitable for production
    energy forecasting.

    The forecaster uses a multi-output strategy where each quantile is predicted
    by separate trees within the same boosting ensemble. This approach provides
    well-calibrated uncertainty estimates while maintaining computational efficiency.

    Invariants:
        - fit() must be called before predict() to train the model
        - Configuration quantiles determine the number of prediction outputs
        - Model state is preserved across predict() calls after fitting
        - Input features must match training data structure during prediction

    Example:
        Basic forecasting workflow

        >>> from datetime import timedelta
        >>> from openstef_core.types import LeadTime, Quantile
        >>> forecaster = LGBMLinearForecaster(
        ...     quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        ...     horizons=[LeadTime(timedelta(hours=1))],
        ...     hyperparams=LGBMLinearHyperParams(n_estimators=100, max_depth=6),
        ... )
        >>> forecaster.fit(training_data)  # doctest: +SKIP
        >>> predictions = forecaster.predict(test_data)  # doctest: +SKIP

    Note:
        LightGBM dependency is optional and must be installed separately.
        The model automatically handles multi-quantile output and uses
        magnitude-weighted pinball loss by default for better forecasting performance.

    See Also:
        LGBMLinearHyperParams: Detailed hyperparameter configuration options.
        Forecaster: Base interface for all forecasting models.
        GBLinearForecaster: Alternative linear model using XGBoost.
    """

    _use_linear_tree: ClassVar[bool] = True

    HyperParams: ClassVar[type[LGBMLinearHyperParams]] = LGBMLinearHyperParams

    hyperparams: LGBMLinearHyperParams = Field(default_factory=LGBMLinearHyperParams)

    @property
    @override
    def hparams(self) -> LGBMLinearHyperParams:
        return self.hyperparams


__all__ = ["LGBMLinearForecaster", "LGBMLinearHyperParams"]
