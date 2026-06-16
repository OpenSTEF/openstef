# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasting workflows from config.

This module is **runtime-light**: importing it pulls in dependency-free pydantic
config (checkpoint refs, execution-provider configs) and the pure-Python
transform/workflow classes from ``openstef_models``. The heavy inference runtime
— ONNX Runtime, Torch — is imported lazily inside the selected backend's
:meth:`build` method, so a caller that only inspects or serialises a config never
pays for a backend it will not use.

The factory returns a
:class:`~openstef_models.workflows.custom_forecasting_workflow.CustomForecastingWorkflow`
rather than a bare forecaster, so the same preprocessing (feature selection) and
postprocessing (quantile sorting) wraps the model as for any other OpenSTEF
model. Chronos-2 conditions on covariates, so every selected non-target feature
column is forwarded to the model as a known covariate.

Typical use::

    from openstef_foundation_models.presets.forecasting_workflow import (
        ForecastingWorkflowConfig,
        create_forecasting_workflow,
    )
    from openstef_foundation_models.models.checkpoint import LocalCheckpoint

    config = ForecastingWorkflowConfig(
        model="chronos2",
        checkpoint=LocalCheckpoint(path="chronos-2.onnx"),
        quantiles=[Quantile(0.1), Quantile(0.5), Quantile(0.9)],
        horizons=[LeadTime.from_string("PT48H")],
    )
    workflow = create_forecasting_workflow(config)
"""

from typing import Literal

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.mixins import TransformPipeline
from openstef_core.types import LeadTime, Q, Quantile
from openstef_foundation_models.inference.backend import InferenceBackend
from openstef_foundation_models.inference.providers import ExecutionProvider, SessionOptionsConfig
from openstef_foundation_models.models.checkpoint import CheckpointRef, HubCheckpoint
from openstef_foundation_models.models.forecasting.chronos2_forecaster import (
    Chronos2Forecaster,
    Chronos2HyperParams,
)
from openstef_models.mixins import ModelIdentifier
from openstef_models.models import ForecastingModel
from openstef_models.transforms.general import Selector
from openstef_models.transforms.postprocessing import QuantileSorter
from openstef_models.utils.feature_selection import FeatureSelection, Include
from openstef_models.workflows.custom_forecasting_workflow import (
    CustomForecastingWorkflow,
    ForecastingCallback,
)


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
        description="Ordered execution providers to try. Defaults to CUDA when available, otherwise CPU.",
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
#: workflow config and factory grow to a discriminated union of backends
#: (e.g. a Torch backend) without changing their public signatures.
BackendConfig = OnnxBackendConfig


class ForecastingWorkflowConfig(BaseConfig):
    """Declarative configuration for a foundation-model forecasting workflow.

    Selects a model family and the checkpoint that backs it, the requested
    quantiles and horizons, and the feature columns Chronos-2 conditions on.
    Every selected non-target column is treated as a known covariate, so the
    weather forecasts (radiation, wind speed, temperature) condition the
    forecast. Column names default to the Liander 2024 dataset conventions and
    are configurable.
    """

    model: Literal["chronos2"] = Field(description="Foundation model family to use.")
    checkpoint: CheckpointRef | None = Field(
        default=None,
        description="Checkpoint to load. When omitted, a sensible default is inferred from `model`.",
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
    radiation_column: str = Field(
        default="shortwave_radiation",
        description="Name of the (known-future) radiation covariate column.",
    )
    wind_speed_column: str = Field(
        default="wind_speed_80m",
        description="Name of the (known-future) wind-speed covariate column.",
    )
    temperature_column: str = Field(
        default="temperature_2m",
        description="Name of the (known-future) temperature covariate column.",
    )
    selected_features: FeatureSelection | None = Field(
        default=None,
        description="Features to keep before forecasting. When omitted, the target plus the radiation, "
        "wind-speed and temperature covariate columns are selected.",
    )

    providers: list[ExecutionProvider] | None = Field(
        default=None,
        description="Ordered ONNX Runtime execution providers to try. Defaults to CUDA when available, otherwise CPU.",
    )
    session_options: SessionOptionsConfig | None = Field(
        default=None,
        description="Optional ONNX Runtime session options.",
    )
    strict_providers: bool = Field(
        default=False,
        description="Raise (rather than warn) if the realized provider chain silently falls back to CPU.",
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


def create_forecasting_workflow(
    config: ForecastingWorkflowConfig,
    *,
    backend: InferenceBackend | None = None,
) -> CustomForecastingWorkflow:
    """Build a foundation-model forecasting workflow from a declarative config.

    Resolves the checkpoint (lazily importing the inference runtime), composes
    the forecaster for the selected model family, and wraps it in a workflow with
    feature-selection preprocessing and quantile-sorting postprocessing. There is
    no training step: the model is zero-shot, so
    :meth:`CustomForecastingWorkflow.fit` only fits the feature selector.

    Args:
        config: The workflow configuration.
        backend: A pre-built inference backend to run the model on. When omitted,
            an ONNX backend is built from the config's checkpoint and provider
            settings (defaulting to CUDA when available, otherwise CPU). Pass an
            explicit backend to control the device or runtime (e.g. a GPU ONNX
            session, or a Torch backend).

    Returns:
        A ready-to-use workflow composing the configured backend.
    """
    selected_features = config.selected_features or Include(
        config.target_column,
        config.radiation_column,
        config.wind_speed_column,
        config.temperature_column,
    )

    match config.model:
        case "chronos2":
            # Inferred default while the ONNX checkpoint is unpublished; resolves
            # to the Hub repo once it is published. Pass `checkpoint` explicitly
            # (e.g. a LocalCheckpoint) until then.
            checkpoint = config.checkpoint or HubCheckpoint(
                repo_id="openstef/chronos-2-onnx", filename="chronos-2.onnx"
            )
            model_backend = backend or OnnxBackendConfig(
                checkpoint=checkpoint,
                providers=config.providers,
                session_options=config.session_options,
                strict_providers=config.strict_providers,
            ).build()
            forecaster = Chronos2Forecaster(
                backend=model_backend,
                quantiles=config.quantiles,
                horizons=config.horizons,
                hyperparams=config.chronos2_hyperparams,
            )

    callbacks: list[ForecastingCallback] = []
    return CustomForecastingWorkflow(
        model=ForecastingModel(
            preprocessing=TransformPipeline(transforms=[Selector(selection=selected_features)]),
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
    "ForecastingWorkflowConfig",
    "OnnxBackendConfig",
    "create_forecasting_workflow",
]
