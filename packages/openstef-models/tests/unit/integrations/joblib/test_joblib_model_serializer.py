# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from io import BytesIO

from openstef_core.mixins import Stateful
from openstef_models.integrations.joblib.joblib_model_serializer import JoblibModelSerializer


class SimpleSerializableModel(Stateful):
    """A simple model class that can be pickled for testing."""

    def __init__(self) -> None:
        self.target_column = "load"
        self.is_fitted = True


def test_joblib_model_serializer__roundtrip__preserves_model_integrity():
    """Test complete serialize/deserialize roundtrip preserves model state."""
    # Arrange
    buffer = BytesIO()
    serializer = JoblibModelSerializer()
    model = SimpleSerializableModel()

    # Act - Serialize then deserialize
    serializer.serialize(model, buffer)
    buffer.seek(0)
    restored_model = serializer.deserialize(SimpleSerializableModel(), buffer)

    # Assert - Model state should be identical
    assert restored_model.target_column == model.target_column
    assert restored_model.is_fitted == model.is_fitted
