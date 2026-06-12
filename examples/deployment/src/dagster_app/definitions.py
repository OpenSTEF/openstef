# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


# Dagster's asset decorators and the generic MaterializeResult are not fully typed; silence that
# noise. Real type checking stays on for everything else in this file.

"""Dagster deployment example for OpenSTEF (asset-based, DAG orchestration).

Three software-defined assets model the pipeline, with one **partition per forecast target**
(so each target is an independent, retriable materialization that Dagster fans out in parallel):

- ``input_data`` — fetch + combine measurements and weather for the target.
- ``trained_model`` — train on the training slice and log the model to MLflow.
- ``forecast`` — predict on the prediction slice and publish the result.

Dagster's built-in IO manager passes ``input_data`` between the assets; the model itself is
handed over through the shared MLflow store (the prediction asset builds a fresh workflow that
loads what training persisted). Run training before forecasting.

Launch the UI with ``uv run poe deploy-dagster-ui`` (``dagster dev``); run the CLI with
``uv run poe deploy-dagster-train`` / ``deploy-dagster-forecast`` (see :mod:`dagster_app.run`).
"""

import dagster as dg
from common import pipeline, services
from common.config import Settings, model_id_for

from openstef_core.datasets import TimeSeriesDataset

settings = Settings()

# One partition per target. Partition keys must be filesystem/URL safe, so we key on the model
# id (a slug) and map back to the dataset target for fetching.
targets_by_key = {model_id_for(target): target for target in settings.targets}
target_partitions = dg.StaticPartitionsDefinition(list(targets_by_key))


@dg.asset(partitions_def=target_partitions, description="Fetch + combine measurements and weather for one target.")
def input_data(context: dg.AssetExecutionContext) -> TimeSeriesDataset:
    return pipeline.input_data(targets_by_key[context.partition_key], settings=settings)


@dg.asset(partitions_def=target_partitions, description="Train a model for one target and log it to MLflow.")
def trained_model(context: dg.AssetExecutionContext, input_data: TimeSeriesDataset) -> dg.MaterializeResult:  # ty: ignore[missing-type-argument]
    target = targets_by_key[context.partition_key]
    pipeline.build_workflow(target, settings=settings).fit(pipeline.training_view(input_data, settings=settings))
    return dg.MaterializeResult(metadata={"target": target, "model_id": context.partition_key})


@dg.asset(
    partitions_def=target_partitions,
    deps=[trained_model],
    description="Forecast one target and publish the result.",
)
def forecast(context: dg.AssetExecutionContext, input_data: TimeSeriesDataset) -> dg.MaterializeResult:  # ty: ignore[missing-type-argument]
    target = targets_by_key[context.partition_key]
    # A fresh workflow loads the model the trained_model asset persisted to MLflow.
    workflow = pipeline.build_workflow(target, settings=settings)
    result = workflow.predict(
        pipeline.prediction_view(input_data, settings=settings), forecast_start=settings.reference_time
    )
    path = services.publish_forecast(result, target, settings=settings)
    return dg.MaterializeResult(metadata={"output_path": str(path), "rows": len(result.data)})


all_assets = [input_data, trained_model, forecast]
defs = dg.Definitions(assets=all_assets)
