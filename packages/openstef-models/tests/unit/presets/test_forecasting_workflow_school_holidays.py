# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Literal

import pytest

from openstef_models.presets.forecasting_workflow import (
    ForecastingWorkflowConfig,
    create_forecasting_workflow,
)
from openstef_models.transforms.time_domain import SchoolHolidayConfig, SchoolHolidayFeatureAdder


def _school_holiday_transforms(
    model_name: Literal["xgboost", "gblinear", "lgbm", "lgbmlinear"], school_holidays: SchoolHolidayConfig | None
) -> list[SchoolHolidayFeatureAdder]:
    config = ForecastingWorkflowConfig(
        model_id="test", model=model_name, school_holidays=school_holidays, mlflow_storage=None
    )
    workflow = create_forecasting_workflow(config)
    return [t for t in workflow.model.preprocessing.transforms if isinstance(t, SchoolHolidayFeatureAdder)]


def test_school_holidays_are_disabled_by_default() -> None:
    config = ForecastingWorkflowConfig(model_id="test", model="gblinear", mlflow_storage=None)

    assert config.school_holidays is None
    assert _school_holiday_transforms("gblinear", None) == []


@pytest.mark.parametrize("model_name", ["xgboost", "lgbm", "lgbmlinear", "gblinear"])
def test_school_holidays_add_configured_transform(
    model_name: Literal["xgboost", "gblinear", "lgbm", "lgbmlinear"],
) -> None:
    school_holidays = SchoolHolidayConfig(
        country_code="NL",
        subdivision_code="NL-NH",
        years_ahead=1,
        one_hot_by_name=True,
        csv_path=Path("holidays.csv"),
    )

    (transform,) = _school_holiday_transforms(model_name, school_holidays)

    assert transform.country_code == "NL"
    assert transform.subdivision_code == "NL-NH"
    assert transform.years_ahead == 1
    assert transform.one_hot_by_name is True
    assert transform.csv_path == Path("holidays.csv")


def test_create_transform_returns_new_instance_each_call() -> None:
    school_holidays = SchoolHolidayConfig(country_code="NL")

    assert school_holidays.create_transform() is not school_holidays.create_transform()
