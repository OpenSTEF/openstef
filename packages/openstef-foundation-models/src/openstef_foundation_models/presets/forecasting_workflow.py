# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasting workflows from config.

A :class:`ForecastingWorkflowConfig` declares the model family, the checkpoint
that backs it, the requested quantiles/horizons, the target column, and the
columns to keep; :func:`create_forecasting_workflow` turns it into a
:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`
with feature-selection preprocessing and quantile-sorting postprocessing. Every
selected non-target column is forwarded to the model as a known covariate.

The checkpoint defaults to the published OpenSTEF Chronos-2 model on the
HuggingFace Hub, so the minimal config is just::

    workflow = create_forecasting_workflow(ForecastingWorkflowConfig())

Pick a different size or variant through :class:`~openstef_foundation_models.models.catalog.Chronos2`,
or pass a :class:`~openstef_foundation_models.models.checkpoint.LocalCheckpoint`
to run a file already on disk.
"""

from collections.abc import Mapping
from typing import Annotated, Literal, Protocol, assert_never, runtime_checkable

from pydantic import BeforeValidator, ConfigDict, Field, WithJsonSchema

from openstef_core.base_model import BaseConfig
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.inference.provider_selection import DefaultProviderPolicy
from openstef_foundation_models.inference.providers import ExecutionProvider, SessionOptionsConfig
from openstef_foundation_models.models.catalog import Chronos2
from openstef_foundation_models.models.checkpoint import CheckpointRef
from openstef_foundation_models.models.forecasting.chronos2_forecaster import (
    Chronos2Forecaster,
    Chronos2HyperParams,
)
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.transforms.general import Selector
from openstef_models.transforms.postprocessing import QuantileSorter
from openstef_models.utils.feature_selection import FeatureSelection
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


@runtime_checkable
class BackendConfigProtocol(Protocol):
    """Configuration that can build an inference backend for a checkpoint.

    The workflow only relies on this small contract, so third-party backends do
    not need to inherit from an OpenSTEF class or be added to a registry. A
    backend config owns *how* to run a checkpoint; the checkpoint itself is
    passed to :meth:`build` by the workflow factory.
    """

    def build(self, checkpoint: CheckpointRef) -> InferenceBackend:
        """Build a ready-to-run backend for *checkpoint*."""
        ...


class _ExistingBackendConfig:
    """Adapt an already constructed backend to the config protocol."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend

    def build(self, checkpoint: CheckpointRef) -> InferenceBackend:
        _ = checkpoint
        return self._backend


class OnnxBackendConfig(BaseConfig):
    """Compute configuration for an ONNX Runtime inference backend.

    Holds only *how* to run the model (execution providers, session options), not
    *which* weights: the checkpoint is supplied to :meth:`build` by the caller, so
    the same compute settings can run different checkpoints.

    This is the default backend config. Other backend configs only need to
    provide the same ``build(checkpoint)`` method.
    """

    kind: Literal["onnx"] = Field(default="onnx", description="Discriminator tag for backend type.")
    providers: list[ExecutionProvider] | None = Field(
        default=None,
        description="Ordered execution providers to try. ``None`` lets :attr:`policy` pick a host-appropriate "
        "chain from the checkpoint metadata (graceful). An explicit list is used exactly as given (strict: a "
        "missing accelerator raises).",
    )
    policy: DefaultProviderPolicy = Field(
        default=DefaultProviderPolicy(),
        description="Selection policy used when :attr:`providers` is None. Maps the checkpoint's precision and "
        "static-shape-ness plus the host to an ordered provider chain. Replace it (e.g. a subclass overriding "
        "select) to target hardware the default does not cover.",
    )
    session_options: SessionOptionsConfig | None = Field(
        default=None,
        description="Optional ONNX Runtime session options.",
    )

    def build(self, checkpoint: CheckpointRef) -> InferenceBackend:
        """Resolve *checkpoint* and build the ONNX Runtime backend.

        Importing the backend raises ``MissingExtraError`` if ONNX Runtime is not installed.

        Args:
            checkpoint: The checkpoint (weights + metadata) to load and run.

        Returns:
            A ready-to-run backend wrapping the resolved checkpoint.
        """
        from openstef_foundation_models.inference.onnx_backend import OnnxBackend  # noqa: PLC0415

        resolved = checkpoint.resolve()
        return OnnxBackend.from_checkpoint(
            resolved,
            providers=self.providers,
            session_options=self.session_options,
            policy=self.policy,
        )


#: Backwards-compatible alias for the default backend configuration.
BackendConfig = OnnxBackendConfig


def _coerce_backend_config(value: object) -> object:
    """Accept a ready backend or restore the built-in ONNX config.

    Third-party backend configs are intentionally passed as Python objects: they
    may contain runtime handles that are not YAML/JSON serializable. The built-in
    ONNX config remains round-trippable through the usual ``kind`` tag.
    """
    if isinstance(value, InferenceBackend):
        return _ExistingBackendConfig(value)
    if isinstance(value, Mapping) and value.get("kind") == "onnx":
        return OnnxBackendConfig.model_validate(value)
    return value


#: A backend config accepted by :class:`ForecastingWorkflowConfig`. The
#: ``BeforeValidator`` keeps the default ONNX config deserializable while the
#: protocol leaves room for out-of-tree backends. The explicit JSON Schema keeps
#: arbitrary backend objects representable in generated configuration docs.
BackendConfigField = Annotated[
    BackendConfigProtocol,
    BeforeValidator(_coerce_backend_config),
    WithJsonSchema(
        {
            "type": "object",
            "description": "A backend configuration with build(checkpoint), or a ready InferenceBackend. "
            "Defaults to OnnxBackendConfig.",
        }
    ),
]


class ForecastingWorkflowConfig(BaseConfig):
    """Declarative configuration for a foundation-model forecasting workflow.

    Selects a model family and the checkpoint that backs it, the requested
    quantiles and horizons, the target column, and the columns to keep before
    forecasting. Every kept non-target column is treated as a known covariate, so
    weather forecasts condition the prediction. Compute settings (execution
    providers, session options) live on the nested :attr:`backend` config.

    ``backend`` accepts any object implementing the small
    :class:`BackendConfigProtocol`, or an already constructed
    :class:`~openstef_foundation_models.inference.backend.InferenceBackend`, so a
    backend hosted behind KServe, a custom in-process runtime, or any other
    ``build(checkpoint)`` factory can be used without changing this config class.
    The default remains :class:`OnnxBackendConfig`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Literal["chronos2"] = Field(default="chronos2", description="Foundation model family to use.")
    checkpoint: CheckpointRef = Field(
        default=Chronos2.BASE.checkpoint(),
        description="Checkpoint (weights + metadata) to load and run. Defaults to the published OpenSTEF "
        "Chronos-2 dynamic ONNX checkpoint on the HuggingFace Hub. Pick a size and variant with "
        "`Chronos2.<SIZE>.checkpoint(...)`, or pass a LocalCheckpoint to run a file already on disk.",
    )

    quantiles: list[Quantile] = Field(
        default=[Q(0.5)],
        min_length=1,
        description="Quantile levels to predict.",
    )
    horizons: list[LeadTime] = Field(
        default=[LeadTime.from_string("PT48H")],
        min_length=1,
        description="Forecast horizons to predict.",
    )

    target_column: str = Field(default="load", description="Name of the target column to forecast.")
    selected_features: FeatureSelection = Field(
        default_factory=lambda: FeatureSelection.ALL,
        description="Columns to keep before forecasting. Defaults to all columns. Every kept non-target column "
        "is forwarded to the model as a known covariate. Must include the target column.",
    )

    backend: BackendConfigField = Field(
        default_factory=OnnxBackendConfig,
        description="Inference backend (compute) configuration. Any object implementing the "
        "`BackendConfigProtocol.build(checkpoint)` protocol is accepted; the default is the local ONNX "
        "Runtime backend.",
    )

    chronos2_hyperparams: Chronos2HyperParams = Field(
        default_factory=Chronos2HyperParams,
        description="Hyperparameters for the Chronos-2 forecaster.",
    )

    model_id: ModelIdentifier = Field(default="chronos2", description="Unique identifier for the workflow model.")
    run_name: str | None = Field(default=None, description="Optional name for this workflow run.")
    experiment_tags: dict[str, str] = Field(
        default_factory=dict,
        description="Optional metadata tags for experiment tracking.",
    )


def create_forecasting_workflow(config: ForecastingWorkflowConfig) -> CustomForecastingWorkflow:
    """Build a foundation-model forecasting workflow from a declarative config.

    Resolves the checkpoint (lazily importing the inference runtime), composes
    the forecaster for the selected model family on the configured backend, and
    wraps it in a workflow with feature-selection preprocessing and
    quantile-sorting postprocessing. There is no training step: the model is
    zero-shot, so :meth:`CustomForecastingWorkflow.fit` only fits the feature
    selector.

    Args:
        config: The workflow configuration.

    Returns:
        A ready-to-use workflow composing the configured backend.
    """
    match config.model:
        case "chronos2":
            model_backend = config.backend.build(config.checkpoint)
            forecaster = Chronos2Forecaster(
                backend=model_backend,
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.chronos2_hyperparams,
            )
        case _:
            assert_never(config.model)

    callbacks: list[ForecastingCallback] = []
    return CustomForecastingWorkflow(
        model=ForecastingModel(
            preprocessing=TransformPipeline(transforms=[Selector(selection=config.selected_features)]),
            forecaster=forecaster,
            postprocessing=TransformPipeline(transforms=[QuantileSorter()]),
            target_column=config.target_column,
        ),
        model_id=config.model_id,
        run_name=config.run_name,
        callbacks=callbacks,
        experiment_tags=config.experiment_tags,
    )


__all__ = [
    "BackendConfig",
    "BackendConfigProtocol",
    "ForecastingWorkflowConfig",
    "OnnxBackendConfig",
    "create_forecasting_workflow",
]
