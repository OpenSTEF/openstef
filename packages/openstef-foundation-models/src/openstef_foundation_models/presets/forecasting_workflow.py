# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasters from declarative config.

This module is intentionally **import-light**: importing it pulls in only
dependency-free pydantic config (checkpoint refs, execution-provider configs,
forecaster configs). The heavy runtime — ONNX Runtime, Torch — is imported
lazily inside the selected backend's :meth:`build` method, so a caller that only
inspects or serialises a config never pays for a backend it will not use.

Typical use::

    from openstef_foundation_models.presets.forecasting_workflow import (
        FoundationForecasterConfig,
        OnnxBackendConfig,
        create_foundation_forecaster,
    )
    from openstef_foundation_models.models.checkpoint import LocalCheckpoint

    config = FoundationForecasterConfig(
        model="chronos2",
        backend=OnnxBackendConfig(checkpoint=LocalCheckpoint(path="chronos-2.onnx")),
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime.from_string("PT24H")],
    )
    forecaster = create_foundation_forecaster(config)
"""

from typing import Literal

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.types import LeadTime, Quantile
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.inference.providers import ExecutionProvider, SessionOptionsConfig
from openstef_foundation_models.models.checkpoint import CheckpointRef
from openstef_foundation_models.models.forecasting.chronos2_forecaster import (
    Chronos2Forecaster,
    Chronos2HyperParams,
)
from openstef_models.models.forecasting.forecaster import Forecaster


class OnnxBackendConfig(BaseConfig):
    """Configuration for an ONNX Runtime inference backend.

    :meth:`build` resolves the checkpoint and constructs an
    :class:`~openstef_foundation_models.inference.onnx_backend.OnnxBackend`. The
    ONNX Runtime dependency is imported lazily there, so this config stays
    importable without the ``[cpu]``/``[gpu]`` extra installed.
    """

    kind: Literal["onnx"] = Field(default="onnx", description="Discriminator tag for backend type.")
    checkpoint: CheckpointRef = Field(description="Local or Hub checkpoint to load.")
    providers: list[ExecutionProvider] | None = Field(
        default=None,
        description="Ordered execution providers to try. Defaults to CPU only.",
    )
    session_options: SessionOptionsConfig | None = Field(
        default=None,
        description="Optional ONNX Runtime session options.",
    )
    strict_providers: bool = Field(
        default=False,
        description="Raise (rather than warn) if the realized provider chain silently falls back to CPU.",
    )

    def build(self) -> InferenceBackend:
        """Resolve the checkpoint and build the ONNX Runtime backend.

        Importing the backend raises ``MissingExtraError`` if ONNX Runtime is not installed.

        Returns:
            A ready-to-run backend wrapping the resolved checkpoint.
        """
        from openstef_foundation_models.inference.onnx_backend import OnnxBackend  # noqa: PLC0415

        resolved = self.checkpoint.resolve()
        return OnnxBackend.from_checkpoint(
            resolved,
            providers=self.providers,
            session_options=self.session_options,
            strict_providers=self.strict_providers,
        )


#: A backend configuration. Currently ONNX-only; kept as a named type so the
#: forecaster config and factory grow to a discriminated union of backends
#: (e.g. a Torch backend) without changing their public signatures.
BackendConfig = OnnxBackendConfig


class FoundationForecasterConfig(BaseConfig):
    """Declarative configuration for a foundation-model forecaster.

    Selects a model family and the backend that executes it, plus the requested
    quantiles and horizons. Designed to grow to more model families by extending
    the ``model`` selector and adding per-model hyperparameter fields.
    """

    model: Literal["chronos2"] = Field(description="Foundation model family to use.")
    backend: BackendConfig = Field(description="Backend that executes the model checkpoint.")
    quantiles: list[Quantile] = Field(
        default=[Quantile(0.5)],
        min_length=1,
        description="Quantile levels to predict.",
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT48H")],
        min_length=1,
        description="Forecast horizons to predict.",
    )
    chronos2_hyperparams: Chronos2HyperParams = Field(
        default_factory=Chronos2HyperParams,
        description="Hyperparameters for the Chronos-2 forecaster.",
    )


def create_foundation_forecaster(config: FoundationForecasterConfig) -> Forecaster:
    """Build a foundation-model forecaster from a declarative config.

    Constructs the backend (lazily importing its runtime) and composes it into
    the forecaster for the selected model family.

    Args:
        config: The forecaster configuration.

    Returns:
        A ready-to-use forecaster composing the configured backend.
    """
    backend = config.backend.build()

    match config.model:
        case "chronos2":
            return Chronos2Forecaster(
                backend=backend,
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.chronos2_hyperparams,
            )


__all__ = [
    "BackendConfig",
    "FoundationForecasterConfig",
    "OnnxBackendConfig",
    "create_foundation_forecaster",
]
