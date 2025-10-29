# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pickle  # noqa: S403 - controlled test
import warnings
from typing import ClassVar, override

from openstef_core.mixins.stateful import Stateful
from openstef_core.types import Any


class SimpleStateful(Stateful):
    """Simple stateful object for testing basic serialization."""

    def __init__(self, value: int, name: str):
        self.value = value
        self.name = name


class VersionedStateful(Stateful):
    """Stateful object with custom version for testing migrations."""

    _VERSION: ClassVar[int] = 2

    def __init__(self, value: int, label: str):
        self.value = value
        self.label = label

    @classmethod
    @override
    def _migrate_state(cls, state: dict[str, Any], from_version: int, to_version: int) -> dict[str, Any]:
        """Migrate state from v1 to v2, renaming 'name' to 'label'."""
        if from_version == 1 and to_version == 2 and "name" in state:
            state["label"] = state.pop("name")
        return state


def test_stateful__basic_pickle_roundtrip():
    """Test that a simple Stateful object can be pickled and unpickled."""
    # Arrange
    original = SimpleStateful(value=42, name="test")

    # Act - pickle and unpickle
    pickled = pickle.dumps(original)
    restored = pickle.loads(pickled)  # noqa: S301 - Controlled test

    # Assert - verify all attributes are preserved
    assert restored.value == original.value
    assert restored.name == original.name


def test_stateful__state_migration_from_v1_to_v2():
    """Test that state is correctly migrated from version 1 to version 2."""
    # Arrange - create a v1 state manually
    v1_state = {
        "__version__": 1,
        "__class_name__": "VersionedStateful",
        "state": {"value": 100, "name": "old_field"},
    }

    # Act - restore object from v1 state
    obj = VersionedStateful.__new__(VersionedStateful)
    obj.__setstate__(v1_state)

    # Assert - verify migration renamed 'name' to 'label'
    assert obj.value == 100
    assert obj.label == "old_field"
    assert not hasattr(obj, "name")


def test_stateful__legacy_state_without_version_warns():
    """Test that loading legacy state without version metadata issues a warning."""
    # Arrange - create a legacy state (just a dict without __version__)
    legacy_state = {"value": 50, "name": "legacy"}

    # Act & Assert - verify warning is issued
    obj = SimpleStateful.__new__(SimpleStateful)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.__setstate__(legacy_state)

        # Verify warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "legacy" in str(w[0].message).lower()
        assert "version metadata" in str(w[0].message).lower()

    # Assert - verify state was still restored
    assert obj.value == 50
    assert obj.name == "legacy"


def test_stateful__forward_compatibility_warns():
    """Test that loading a newer version state issues a forward compatibility warning."""
    # Arrange - create a state with version higher than current
    future_state = {
        "__version__": 999,
        "__class_name__": "SimpleStateful",
        "state": {"value": 75, "name": "future"},
    }

    # Act & Assert - verify warning is issued
    obj = SimpleStateful.__new__(SimpleStateful)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        obj.__setstate__(future_state)

        # Verify warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "v999" in str(w[0].message)
        assert "v1" in str(w[0].message)
        assert "forward compatibility" in str(w[0].message).lower()

    # Assert - verify state was restored despite version mismatch
    assert obj.value == 75
    assert obj.name == "future"
