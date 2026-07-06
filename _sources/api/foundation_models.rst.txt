.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _foundation-models-api:

Foundation Models Package (:mod:`openstef_foundation_models`)
=============================================================

.. currentmodule:: openstef_foundation_models

Pretrained foundation-model forecasters that run on an ONNX inference runtime and plug
into openstef-models workflows and openstef-beam backtesting. Chronos-2 is the first model
family the package ships; the inference backend, provider selection, checkpoint catalog,
and workflow machinery are model-agnostic.

The package is split into four subpackages:

- ``models`` holds the checkpoint catalog (:class:`~openstef_foundation_models.models.catalog.Chronos2`,
  :class:`~openstef_foundation_models.models.catalog.CheckpointVariant`), the checkpoint
  resolution types, and the :class:`~openstef_foundation_models.models.forecasting.chronos2_forecaster.Chronos2Forecaster`.
- ``presets`` holds :class:`~openstef_foundation_models.presets.forecasting_workflow.ForecastingWorkflowConfig`
  and :func:`~openstef_foundation_models.presets.forecasting_workflow.create_forecasting_workflow`,
  the one-call path from config to a runnable workflow.
- ``inference`` holds the :class:`~openstef_foundation_models.inference.backend.InferenceBackend`
  protocol, the ONNX backend, the execution-provider configs, and the host-aware
  provider selection policy.
- ``integrations`` holds :class:`~openstef_foundation_models.integrations.beam.FoundationModelBacktestForecaster`,
  the beam backtesting adapter.

.. autosummary::
   :toctree: generated/
   :template: package_overview.rst
   :recursive:

   models
   presets
   inference
   integrations
