# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import yaml
from pydantic import TypeAdapter

from openstef_beam.core.config import BaseConfig, read_yaml_config, write_yaml_config


class SampleConfig(BaseConfig):
    foo: int
    bar: str


def test_write_yaml_basic(tmp_path: Path):
    """Basic write: YAML matches model_dump."""
    # Arrange
    cfg = SampleConfig(foo=1, bar="abc")
    path = tmp_path / "config.yaml"
    # Act
    write_yaml_config(cfg, path)
    # Assert
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    expected = cfg.model_dump(mode="json")
    assert data == expected


def test_read_yaml_basic(tmp_path: Path):
    """Basic read via helper returns model instance."""
    # Arrange
    path = tmp_path / "config.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump({"foo": 10, "bar": "value"}, f)
    # Act
    result = read_yaml_config(path, class_type=SampleConfig)
    # Assert
    assert result == SampleConfig(foo=10, bar="value")


def test_read_yaml_type_adapter(tmp_path: Path):
    """TypeAdapter branch returns raw validated value."""
    # Arrange
    path = tmp_path / "list.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump([1, 2, 3], f)
    adapter = TypeAdapter(list[int])
    # Act
    result = read_yaml_config(path, class_type=adapter)
    # Assert
    assert result == [1, 2, 3]


def test_roundtrip(tmp_path: Path):
    """Instance write + classmethod read roundtrip."""
    # Arrange
    original = SampleConfig(foo=42, bar="rt")
    path = tmp_path / "rt.yaml"
    # Act
    original.write_yaml(path)
    loaded = SampleConfig.read_yaml(path)
    # Assert
    assert loaded == original
