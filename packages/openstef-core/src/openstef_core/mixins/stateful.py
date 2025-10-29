# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""State management for serializable objects.

Enables objects to save their state and restore it later. Supports model
deployment, caching, and distributed processing by providing versioned
serialization with automatic state migration.
"""

import warnings
from typing import ClassVar, TypedDict, cast

from openstef_core.types import Any


class VersionedState(TypedDict):
    """Versioned state structure for object serialization.

    Contains version metadata and the actual object state, enabling
    backward compatibility through state migration.
    """

    __version__: int
    __class_name__: str
    state: dict[str, Any]


class Stateful:
    """Mixin for objects that can save and restore their internal state.

    Provides versioned serialization with automatic state migration. Objects can be
    pickled, saved to disk, transmitted over networks, or stored in databases, then
    restored to their previous state. Version tracking ensures backward compatibility
    when object structure changes.

    Subclasses can override `_migrate_state` to handle state migrations between versions.
    Increment `_VERSION` when making incompatible changes to object structure.
    """

    _VERSION: ClassVar[int] = 1

    def __getstate__(self) -> VersionedState:
        """Serialize object state with version metadata.

        Returns:
            Versioned state dictionary containing version number, class name,
            and the object's internal state.
        """
        if hasattr(super(), "__getstate__"):
            # In case of pydantic or other base classes implementing __getstate__
            base_state = super().__getstate__()  # type: ignore[misc]
            # Pydantic returns None for models with no fields
            if base_state is None:
                base_state = {}
        else:
            base_state = self.__dict__.copy()

        return VersionedState(
            __version__=self._VERSION,
            __class_name__=self.__class__.__name__,
            state=cast(dict[str, Any], base_state),
        )

    def __setstate__(self, state: Any) -> None:
        """Restore object from serialized state.

        Handles both versioned and legacy state formats. Automatically migrates
        state from older versions using `_migrate_state`. Warns when loading
        legacy objects or when current version is older than saved version.

        Args:
            state: Serialized state, either VersionedState dict or legacy format.
        """
        # Handle legacy objects without versioning
        if not isinstance(state, dict) or "__version__" not in state:  # pyright: ignore[reportUnnecessaryIsInstance]
            warnings.warn(
                f"Loading legacy {self.__class__.__name__} without version metadata.", UserWarning, stacklevel=2
            )
            self._restore_state(state)
            return

        state = cast(VersionedState, state)
        saved_version: int = state["__version__"]
        actual_state: dict[str, Any] = state["state"]

        if saved_version < self._VERSION:
            actual_state = self._migrate_state(state=actual_state, from_version=saved_version, to_version=self._VERSION)
        elif saved_version > self._VERSION:
            warnings.warn(
                f"{self.__class__.__name__} saved with v{saved_version}, "
                f"current is v{self._VERSION}. Forward compatibility not guaranteed.",
                UserWarning,
                stacklevel=2,
            )

        self._restore_state(actual_state)

    def _restore_state(self, state: Any) -> None:
        """Restore object's internal state from a dictionary.

        Delegates to parent class `__setstate__` if available, otherwise
        updates `__dict__` directly.

        Args:
            state: State dictionary to restore.
        """
        # Check if any parent class has __setstate__
        if hasattr(super(), "__setstate__"):
            super().__setstate__(state)  # type: ignore[misc]
        elif state:  # Only update if state is not empty
            self.__dict__.update(state)

    @classmethod
    def _migrate_state(cls, state: dict[str, Any], from_version: int, to_version: int) -> dict[str, Any]:
        """Migrate state from an older version to the current version.

        Override this method in subclasses to handle state transformations when
        the object structure changes. Called automatically during deserialization
        when saved_version < current_version.

        Args:
            state: State dictionary from the older version.
            from_version: Version of the saved state.
            to_version: Target version (current `_VERSION`).

        Returns:
            Migrated state dictionary compatible with current version.
        """
        _ = from_version, to_version  # Improtant arguments, but unused in base implementation
        return state
