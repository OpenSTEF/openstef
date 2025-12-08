# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta
from typing import cast

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import TimeSeriesDataset
from openstef_core.types import LeadTime, Q
from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.presets import EnsembleWorkflowConfig, create_ensemble_workflow
from openstef_models.models.forecasting_model import ForecastingModel
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow


@pytest.fixture
def sample_timeseries_dataset() -> TimeSeriesDataset:
    """Create sample time series data with typical energy forecasting features."""
    n_samples = 25
    rng = np.random.default_rng(seed=42)

    data = pd.DataFrame(
        {
            "load": 100.0 + rng.normal(10.0, 5.0, n_samples),
            "temperature": 20.0 + rng.normal(1.0, 0.5, n_samples),
            "radiation": rng.uniform(0.0, 500.0, n_samples),
        },
        index=pd.date_range("2025-01-01 10:00", periods=n_samples, freq="h", tz="UTC"),
    )

    return TimeSeriesDataset(data, timedelta(hours=1))


@pytest.fixture
def config() -> EnsembleWorkflowConfig:
    return EnsembleWorkflowConfig(
        model_id="ensemble_model_",
        ensemble_type="learned_weights",
        base_models=["gblinear", "lgbm"],
        combiner_model="lgbm",
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime.from_string("PT36H")],
        forecaster_sample_weight_exponent={"gblinear": 1, "lgbm": 0},
    )


@pytest.fixture
def create_models(
    config: EnsembleWorkflowConfig,
) -> tuple[EnsembleForecastingModel, dict[str, ForecastingModel]]:

    ensemble_model = cast(EnsembleForecastingModel, create_ensemble_workflow(config=config).model)

    base_models: dict[str, ForecastingModel] = {}
    for forecaster_name in config.base_models:
        model_config = ForecastingWorkflowConfig(
            model_id=f"{forecaster_name}_model_",
            model=forecaster_name,  # type: ignore
            quantiles=config.quantiles,
            horizons=config.horizons,
            sample_weight_exponent=config.forecaster_sample_weight_exponent[forecaster_name],
        )
        base_model = create_forecasting_workflow(config=model_config).model
        base_models[forecaster_name] = cast(ForecastingModel, base_model)

    return ensemble_model, base_models


def test_preprocessing(
    sample_timeseries_dataset: TimeSeriesDataset,
    create_models: tuple[EnsembleForecastingModel, dict[str, ForecastingModel]],
) -> None:

    ensemble_model, base_models = create_models

    ensemble_model.common_preprocessing.fit(data=sample_timeseries_dataset)

    #  Check all base models
    for name, model in base_models.items():
        # Ensemble model
        common_ensemble = ensemble_model.common_preprocessing.transform(data=sample_timeseries_dataset)
        ensemble_model.model_specific_preprocessing[name].fit(data=common_ensemble)
        transformed_ensemble = ensemble_model.model_specific_preprocessing[name].transform(data=common_ensemble)
        # Base model
        model.preprocessing.fit(data=sample_timeseries_dataset)
        transformed_base = model.preprocessing.transform(data=sample_timeseries_dataset)
        # Compare
        pd.testing.assert_frame_equal(
            transformed_ensemble.data,
            transformed_base.data,
            check_dtype=False,
            check_index_type=False,
            check_column_type=False,
        )
