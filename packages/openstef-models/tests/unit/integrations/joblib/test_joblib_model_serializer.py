# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from io import BytesIO
from typing import Self, cast, override

from openstef_core.mixins import Stateful
from openstef_models.integrations.joblib.joblib_model_serializer import JoblibModelSerializer


class SimpleSerializableModel(Stateful):
    """A simple model class that can be pickled for testing."""

    def __init__(self) -> None:
        self.target_column = "load"
        self.is_fitted = True

    @override
    def to_state(self) -> object:
        return {"target_column": self.target_column, "is_fitted": self.is_fitted}

    @override
    def from_state(self, state: object) -> Self:
        state_dict = cast(dict[str, object], state)
        self.target_column = state_dict.get("target_column", "load")
        self.is_fitted = state_dict.get("is_fitted", True)
        return self


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
