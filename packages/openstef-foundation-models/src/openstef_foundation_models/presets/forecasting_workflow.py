# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Presets for building foundation-model forecasting workflows from config.

This module is **runtime-light**: importing it pulls in dependency-free pydantic
config (checkpoint refs, execution-provider configs) and the pure-Python
transform/workflow classes from ``openstef_models``. The heavy inference runtime
— ONNX Runtime — is imported lazily inside the selected backend's
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


class OnnxBackendConfig(BaseConfig):
    """Compute configuration for an ONNX Runtime inference backend.

    Holds only *how* to run the model (execution providers, session options), not
    *which* weights: the checkpoint is supplied to :meth:`build` by the caller, so
    the same compute settings can run different checkpoints. The ONNX Runtime
    dependency is imported lazily in :meth:`build`, so this config stays importable
    without the ``[cpu]``/``[gpu]`` extra installed.
    """

    kind: Literal["onnx"] = Field(default="onnx", description="Discriminator tag for backend type.")
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
            strict_providers=self.strict_providers,
        )


#: A backend configuration. Currently ONNX-only; kept as a named type so the
#: workflow config and factory grow to a discriminated union of backends
#: (e.g. a Torch backend) without changing their public signatures.
BackendConfig = OnnxBackendConfig


class ForecastingWorkflowConfig(BaseConfig):
    """Declarative configuration for a foundation-model forecasting workflow.

    Selects a model family and the checkpoint that backs it, the requested
    quantiles and horizons, the target column, and the columns to keep before
    forecasting. Every kept non-target column is treated as a known covariate, so
    weather forecasts condition the prediction. Compute settings (execution
    providers, session options) live on the nested :attr:`backend` config.
    """

    model: Literal["chronos2"] = Field(description="Foundation model family to use.")
    checkpoint: CheckpointRef = Field(description="Checkpoint (weights + metadata) to load and run.")

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

    backend: BackendConfig = Field(
        default_factory=OnnxBackendConfig,
        description="Inference backend (compute) configuration: execution providers and session options.",
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
    "ForecastingWorkflowConfig",
    "OnnxBackendConfig",
    "create_forecasting_workflow",
]
