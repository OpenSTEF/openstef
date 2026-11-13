# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from openstef_meta.models.ensemble_forecasting_model import EnsembleForecastingModel
from openstef_meta.presets.forecasting_workflow import (
    EnsembleForecastingWorkflowConfig,
    create_ensemble_forecasting_workflow,
)
from openstef_models.transforms.general import Selector
from openstef_models.transforms.time_domain import SchoolHolidayConfig, SchoolHolidayFeatureAdder

_FEATURES = [
    "load",
    "temperature",
    "load_lag_P1D",
    "load_lag_P7D",
    "is_school_holiday",
    "is_school_holiday_christmas",
]


def _gblinear_features_after_selectors(config: EnsembleForecastingWorkflowConfig, features: list[str]) -> list[str]:
    model = create_ensemble_forecasting_workflow(config).model
    assert isinstance(model, EnsembleForecastingModel)
    resolved = features
    for transform in model.model_specific_preprocessing["gblinear"].transforms:
        if isinstance(transform, Selector):
            resolved = transform.selection.resolve(resolved)
    return resolved


def test_school_holidays_are_not_added_by_default() -> None:
    config = EnsembleForecastingWorkflowConfig(model_id="test", mlflow_storage=None)

    workflow = create_ensemble_forecasting_workflow(config)

    assert not any(isinstance(t, SchoolHolidayFeatureAdder) for t in workflow.model.preprocessing.transforms)


def test_school_holidays_are_added_to_common_preprocessing() -> None:
    config = EnsembleForecastingWorkflowConfig(
        model_id="test", school_holidays=SchoolHolidayConfig(country_code="NL"), mlflow_storage=None
    )

    workflow = create_ensemble_forecasting_workflow(config)

    assert sum(isinstance(t, SchoolHolidayFeatureAdder) for t in workflow.model.preprocessing.transforms) == 1


def test_gblinear_excludes_school_holiday_features_when_enabled() -> None:
    config = EnsembleForecastingWorkflowConfig(
        model_id="test", school_holidays=SchoolHolidayConfig(country_code="NL"), mlflow_storage=None
    )

    resolved = _gblinear_features_after_selectors(config, _FEATURES)

    assert "temperature" in resolved
    assert not any(feature.startswith("is_school_holiday") for feature in resolved)


def test_gblinear_keeps_school_holiday_features_when_disabled() -> None:
    config = EnsembleForecastingWorkflowConfig(model_id="test", mlflow_storage=None)

    resolved = _gblinear_features_after_selectors(config, _FEATURES)

    assert "is_school_holiday" in resolved
    assert "is_school_holiday_christmas" in resolved
