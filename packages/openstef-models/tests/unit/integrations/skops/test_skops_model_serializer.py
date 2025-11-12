# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

import pytest

from openstef_core.mixins import Stateful
from openstef_core.types import LeadTime, Q
from openstef_models.integrations.skops.skops_model_serializer import SkopsModelSerializer
from openstef_models.models.forecasting.forecaster import ForecasterConfig
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster
from openstef_models.models.forecasting.lgbmlinear_forecaster import LGBMLinearForecaster
from openstef_models.models.forecasting.xgboost_forecaster import XGBoostForecaster

if TYPE_CHECKING:
    from openstef_models.models.forecasting.forecaster import Forecaster


class SimpleSerializableModel(Stateful):
    """A simple model class that can be pickled for testing."""

    def __init__(self) -> None:
        self.target_column = "load"
        self.is_fitted = True


def test_skops_model_serializer__roundtrip__preserves_model_integrity():
    """Test complete serialize/deserialize roundtrip preserves model state."""
    # Arrange
    buffer = BytesIO()
    serializer = SkopsModelSerializer()
    model = SimpleSerializableModel()

    # Act - Serialize then deserialize
    serializer.serialize(model, buffer)
    buffer.seek(0)
    restored_model = serializer.deserialize(buffer)

    # Assert - Model state should be identical
    assert isinstance(restored_model, SimpleSerializableModel)
    assert restored_model.target_column == model.target_column
    assert restored_model.is_fitted == model.is_fitted


@pytest.mark.parametrize(
    "forecaster_class",
    [
        XGBoostForecaster,
        LGBMForecaster,
        LGBMLinearForecaster,
    ],
)
def test_skops_works_with_different_forecasters(forecaster_class: type[Forecaster]):
    buffer = BytesIO()
    serializer = SkopsModelSerializer()

    config: ForecasterConfig = forecaster_class.Config(horizons=[LeadTime.from_string("PT12H")], quantiles=[Q(0.5)])  # type: ignore
    assert isinstance(config, ForecasterConfig)
    forecaster = forecaster_class(config=config)

    # Act - Serialize then deserialize
    serializer.serialize(forecaster, buffer)
    buffer.seek(0)
    restored_model = serializer.deserialize(buffer)

    # Assert - Model state should be identical
    assert isinstance(restored_model, forecaster.__class__)
