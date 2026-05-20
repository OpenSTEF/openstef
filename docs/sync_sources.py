# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Sync tutorial and benchmark sources into docs/source for Sphinx.

This script copies example notebooks and benchmarks from their canonical
locations into docs/source/ where Sphinx can find them. It also embeds
specific tutorials into user guide sections so they appear in both the
Examples sidebar and the User Guide sidebar (as separate Sphinx documents).

Run via: ``poe _docs_sync`` (called automatically before ``poe docs``).
"""

import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_SOURCE = ROOT / "docs" / "source"
EXAMPLES = ROOT / "examples"

# Primary copies: full directories
SYNC_DIRS = {
    EXAMPLES / "tutorials": DOCS_SOURCE / "tutorials",
    EXAMPLES / "benchmarks": DOCS_SOURCE / "benchmarks",
}

# Embedded copies: specific tutorials placed into user guide sections
# so they show under the correct sidebar context.
EMBED_MAP = {
    EXAMPLES / "tutorials" / "forecasting_quickstart.py": (
        DOCS_SOURCE / "user_guide" / "getting_started" / "quick_start_tutorial.py"
    ),
    EXAMPLES / "tutorials" / "feature_engineering.py": (
        DOCS_SOURCE / "user_guide" / "guides" / "feature_engineering_tutorial.py"
    ),
    EXAMPLES / "tutorials" / "datasets.py": (
        DOCS_SOURCE / "user_guide" / "guides" / "datasets_tutorial.py"
    ),
}


def sync() -> None:
    """Run the full sync."""
    # Sync full directories
    for src, dst in SYNC_DIRS.items():
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # Embed tutorials into user guide
    for src, dst in EMBED_MAP.items():
        os.makedirs(dst.parent, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)} (embed)")


if __name__ == "__main__":
    print("Syncing example sources into docs/source/...")
    sync()
    print("Done.")
