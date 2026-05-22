# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Pickleable helpers for autosummary templates.

Defined in a separate importable module (rather than in conf.py) so that
Sphinx can pickle configuration values and enable incremental builds.
Classes defined directly in conf.py cannot be pickled because conf.py is
loaded via exec() and is not an importable module.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings

__all__ = ["DiscoverSubmodules"]


class DiscoverSubmodules:
    """Discover child module/package names of a given Python package.

    Used in autosummary Jinja2 templates to dynamically list submodules
    of packages that have empty ``__init__.py`` files.
    """

    def __call__(self, fullname: str) -> list[str]:
        """Return sorted child module names, or [] if *fullname* is not a package."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                mod = importlib.import_module(fullname)
            except Exception:  # noqa: BLE001
                return []
        pkg_path = getattr(mod, "__path__", None)
        if pkg_path is None:
            return []
        return sorted(name for _, name, _ispkg in pkgutil.iter_modules(pkg_path))
