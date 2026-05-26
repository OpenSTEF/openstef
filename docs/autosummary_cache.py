# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Sphinx extension: cache autosummary stub generation between builds.

Problem
-------
Autosummary renders a Jinja2 template for every API page on every build
(~600 templates), importing each Python module to inspect its members.
This takes ~2 minutes even when no Python source has changed.

Solution
--------
Track when stubs were last generated via a timestamp file. On subsequent
builds, skip generation entirely if no Python source file in the monitored
package directories has been modified since the stamp was written.

Usage
-----
Add ``"autosummary_cache"`` to the ``extensions`` list in conf.py (before
``sphinx.ext.autosummary``). The extension patches the generation function
at build time via a ``builder-inited`` event handler.

Cache invalidation
------------------
Generation runs when any of these conditions hold:

- The stamp file does not exist (first build or after ``rm -rf build/``)
- Any ``.py`` file under ``packages/*/src/`` is newer than the stamp
- The generated stubs directory is empty or missing
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sphinx.application import Sphinx
    from sphinx.util.typing import ExtensionMetadata

__all__: list[str] = []

logger = logging.getLogger(__name__)

_STAMP_FILENAME = ".autosummary_stamp"
_PACKAGES_DIR = "packages"
_GENERATED_DIR = Path("api") / "generated"


def _find_package_sources(repo_root: Path) -> list[Path]:
    """Discover all ``packages/*/src/`` directories in the repository."""
    packages_dir = repo_root / _PACKAGES_DIR
    if not packages_dir.is_dir():
        return []
    return sorted(
        src_dir
        for pkg in packages_dir.iterdir()
        if pkg.is_dir() and (src_dir := pkg / "src").is_dir()
    )


def _sources_changed(app: Sphinx) -> bool:
    """Check whether any Python source is newer than the generation stamp."""
    build_dir = Path(app.outdir).parent
    stamp = build_dir / _STAMP_FILENAME

    if not stamp.exists():
        logger.debug("autosummary_cache: no stamp file, regenerating stubs")
        return True

    stamp_mtime = stamp.stat().st_mtime

    # Derive repo root: confdir is docs/source, repo root is two levels up
    repo_root = Path(app.confdir).parent.parent
    pkg_sources = _find_package_sources(repo_root)

    if not pkg_sources:
        logger.warning("autosummary_cache: no package source dirs found under %s", repo_root / _PACKAGES_DIR)
        return True

    for pkg_dir in pkg_sources:
        for py_file in pkg_dir.rglob("*.py"):
            if py_file.stat().st_mtime > stamp_mtime:
                logger.info("autosummary_cache: %s modified, regenerating stubs", py_file.relative_to(repo_root))
                return True

    # Verify generated stubs directory exists and is non-empty
    generated_dir = Path(app.srcdir) / _GENERATED_DIR
    if not generated_dir.is_dir() or not any(generated_dir.iterdir()):
        logger.debug("autosummary_cache: generated dir missing/empty, regenerating stubs")
        return True

    return False


def _update_stamp(app: Sphinx) -> None:
    """Write the current timestamp to the stamp file."""
    build_dir = Path(app.outdir).parent
    stamp = build_dir / _STAMP_FILENAME
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text(str(time.time()))


def _patch_autosummary(app: Sphinx) -> None:
    """Wrap autosummary generation to skip it when sources are unchanged."""
    if not app.config.autosummary_generate:
        return

    from sphinx.ext.autosummary import generate as gen_module

    original_generate = gen_module.generate_autosummary_docs

    def cached_generate(*args, **kwargs):
        if not _sources_changed(app):
            logger.info("autosummary_cache: sources unchanged, skipping stub generation")
            return []
        result = original_generate(*args, **kwargs)
        _update_stamp(app)
        return result

    gen_module.generate_autosummary_docs = cached_generate


def setup(app: Sphinx) -> ExtensionMetadata:
    """Register the caching extension with Sphinx."""
    # Priority 400 runs before autosummary's own builder-inited handler (priority 500)
    app.connect("builder-inited", _patch_autosummary, priority=400)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
