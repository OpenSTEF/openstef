# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import pytest

from openstef_core.exceptions import ModelNotFoundError
from openstef_models.integrations.joblib.storage import LocalModelStorage

if TYPE_CHECKING:
    from collections.abc import Generator

    from openstef_models.models.forecasting_model import ForecastingModel


class SimpleSerializableModel:
    """A simple model class that can be pickled for testing."""

    def __init__(self) -> None:
        self.target_column = "load"
        self.is_fitted = True


@pytest.fixture
def temp_storage_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for storage testing."""
    with TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_forecasting_model() -> SimpleSerializableModel:
    """Create a simple serializable mock object for testing."""
    return SimpleSerializableModel()


@pytest.fixture
def storage(temp_storage_dir: Path) -> LocalModelStorage:
    """Create a LocalModelStorage instance with temporary directory."""
    return LocalModelStorage(storage_dir=temp_storage_dir)


@pytest.mark.parametrize(
    ("model_id", "expected_filename"),
    [
        pytest.param("simple_model", "simple_model.pkl", id="simple_name"),
        pytest.param("model-with_special.chars", "model-with_special.chars.pkl", id="special_characters"),
        pytest.param("123_numeric_model", "123_numeric_model.pkl", id="numeric_prefix"),
    ],
)
def test_local_model_storage__save_and_load_integration(
    storage: LocalModelStorage,
    mock_forecasting_model: SimpleSerializableModel,
    model_id: str,
    expected_filename: str,
):
    """Test save and load integration across different model ID formats."""
    # Arrange & Act - Save the model
    storage.save_model(model_id, cast("ForecastingModel", mock_forecasting_model))

    # Assert - File should exist with correct name
    model_path = storage.storage_dir / expected_filename
    assert model_path.exists()

    # Act - Load the model
    loaded_model = storage.load_model(model_id)

    # Assert - Should get back equivalent model object
    assert loaded_model is not None
    assert hasattr(loaded_model, "target_column")
    assert hasattr(loaded_model, "is_fitted")


def test_local_model_storage__load_model__raises_error_when_file_not_exists(storage: LocalModelStorage):
    """Test that load_model raises ModelNotFoundError when file doesn't exist."""
    # Arrange
    model_id = "nonexistent_model"

    # Act & Assert
    with pytest.raises(ModelNotFoundError, match="nonexistent_model"):
        storage.load_model(model_id)
