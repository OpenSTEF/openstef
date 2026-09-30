# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import yaml
from pydantic import BaseModel as PydanticBaseModel
from pydantic import TypeAdapter

from openstef_core.base_model import BaseConfig, read_yaml_config, write_yaml_config


class SampleConfig(BaseConfig):
    foo: int
    bar: str


class PydanticConfig(PydanticBaseModel):
    foo: int


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


def test_write_yaml_pydantic_model(tmp_path: Path):
    """Writing a plain Pydantic model remains supported."""
    config = PydanticConfig(foo=1)
    path = tmp_path / "pydantic_config.yaml"

    write_yaml_config(config, path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data == config.model_dump(mode="json")


def test_write_yaml_sequence(tmp_path: Path):
    """Writing a sequence preserves a top-level YAML list."""
    configs = [SampleConfig(foo=1, bar="abc"), SampleConfig(foo=2, bar="def")]
    path = tmp_path / "configs.yaml"

    write_yaml_config(configs, path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    assert data == [config.model_dump(mode="json") for config in configs]


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
