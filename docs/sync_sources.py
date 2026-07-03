# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Sync tutorial and benchmark sources into docs/source for Sphinx.

This script copies example notebooks and benchmarks from their canonical
locations into docs/source/ where Sphinx can find them. It also embeds
specific tutorials into user guide sections so they appear in both the
Examples sidebar and the User Guide sidebar (as separate Sphinx documents).

It also materializes the community health files (CONTRIBUTING, CODE_OF_CONDUCT,
SECURITY, SUPPORT, PROJECT_GOVERNANCE). These live canonically in the org-level
``OpenSTEF/.github`` repository — the single source of truth — as plain,
GitHub-flavored Markdown (so they render correctly on GitHub). They are always
downloaded via the GitHub REST API at build time and written verbatim; the docs
render them through the MyST parser (which already handles GitHub constructs
such as ``<details>`` blocks and blockquotes) and reference their headings via
``sphinx.ext.autosectionlabel`` — so there is no bespoke Markdown rewriting to
maintain. See ``sync_community_files``.

Run via: ``poe _docs_sync`` (called automatically before ``poe docs``).
"""

import os
import shutil
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS_SOURCE = ROOT / "docs" / "source"
EXAMPLES = ROOT / "examples"

# -- Community health files (single source of truth: OpenSTEF/.github) --------

# Canonical org-level repo holding the markdown community/governance files.
# The repo name is, by GitHub convention, literally ``.github``.
COMMUNITY_REPO_NAME = ".github"
COMMUNITY_REPO = f"OpenSTEF/{COMMUNITY_REPO_NAME}"

# GitHub REST API contents endpoint. Requesting the ``raw`` media type returns
# the file content verbatim (no base64 wrapping). Files are fetched one by one.
COMMUNITY_API_URL = f"https://api.github.com/repos/{COMMUNITY_REPO}/contents/{{name}}"

# Where the materialized markdown lands. Gitignored; included (never committed)
# by the contribute pages via MyST. Keep in sync with .gitignore.
COMMUNITY_DEST = DOCS_SOURCE / "contribute" / "_community"

# Files pulled from the org .github repo into COMMUNITY_DEST.
COMMUNITY_FILES = (
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "PROJECT_GOVERNANCE.md",
)

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
    EXAMPLES / "tutorials" / "backtesting_quickstart.py": (
        DOCS_SOURCE / "user_guide" / "guides" / "backtesting_tutorial.py"
    ),
    EXAMPLES / "tutorials" / "foundation_model_forecasting_quickstart.py": (
        DOCS_SOURCE / "user_guide" / "getting_started" / "foundation_model_forecasting_quickstart.py"
    ),
    EXAMPLES / "tutorials" / "foundation_model_forecasting.py": (
        DOCS_SOURCE / "user_guide" / "guides" / "foundation_model_forecasting.py"
    ),
}


def _download_community_file(name: str) -> str:
    """Download a single community file via the GitHub REST API.

    Args:
        name: File name in the ``OpenSTEF/.github`` repo (e.g. ``CONTRIBUTING.md``).

    Returns:
        The raw file content as text.

    Raises:
        RuntimeError: If the file cannot be fetched (network error, missing file,
            rate limiting, ...). The docs embed these files and cannot build
            without them, so failure must abort the build with a clear message.
    """
    url = COMMUNITY_API_URL.format(name=name)
    request = urllib.request.Request(  # noqa: S310 (https URL, hardcoded host)
        url,
        headers={
            # ``raw`` media type returns the file content verbatim.
            "Accept": "application/vnd.github.raw",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "openstef-docs-build",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        msg = (
            f"Could not download '{name}' from the GitHub API ({url}): {exc}. "
            "The docs embed the OpenSTEF/.github community files "
            "(CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, SUPPORT.md, "
            "PROJECT_GOVERNANCE.md) and cannot build without them. Ensure "
            "network access to GitHub, then rebuild."
        )
        raise RuntimeError(msg) from exc


def sync_community_files() -> None:
    """Materialize community health files from the org ``.github`` repo.

    The canonical copies live in ``OpenSTEF/.github`` (a single source of truth
    inherited by every repo on GitHub) as plain, GitHub-flavored Markdown. They
    are downloaded via the GitHub REST API and written into
    ``COMMUNITY_DEST``. The contribute pages pull them in with ``.. include::``
    and let the MyST parser render them, so no Markdown rewriting happens here.

    Raises:
        RuntimeError: If any required community file cannot be obtained. The docs
            would otherwise build incompletely, so the build must fail with a
            clear message.
    """
    COMMUNITY_DEST.mkdir(parents=True, exist_ok=True)

    for name in COMMUNITY_FILES:
        print(f"  GitHub API:{COMMUNITY_REPO}:{name}")

        if (COMMUNITY_DEST / name).is_file():
            print(f"    -> {COMMUNITY_DEST.relative_to(ROOT) / name} (already present; skipping)")
            continue

        content = _download_community_file(name)
        dest = COMMUNITY_DEST / name
        dest.write_text(content, encoding="utf-8")
        print(f"    -> {dest.relative_to(ROOT)}")


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

    # Materialize community health files from the org .github repo
    sync_community_files()


if __name__ == "__main__":
    print("Syncing example sources into docs/source/...")
    sync()
    print("Done.")
