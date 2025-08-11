#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""
reuse-fix: Ensure (and optionally repair) REUSE compliance metadata.

Modernised for Python >=3.12 with:
- Full static typing
- Dataclass based configuration
- Non-interactive operation (no y/N prompts)
- Required --copyright and --license options
- Clear exit codes (0 compliant / fixed, 1 error, 2 still non-compliant)
- Separation of concerns & small focused functions

Dependencies:
    pip install reuse

Exit codes:
    0 - Success (already compliant or successfully annotated & now compliant)
    1 - Operational error (invalid path, misuse, unexpected exception)
    2 - Non-compliant after attempted fix (or check-only found issues)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Iterable, Sequence, Tuple
import argparse
import subprocess
import sys

# Third-party (reuse)
from reuse.project import Project
from reuse.report import ProjectReport

# ---------------------------------------------------------------------------
# Exit codes
# ---------------------------------------------------------------------------
class ExitCode(IntEnum):
    SUCCESS = 0
    ERROR = 1
    NON_COMPLIANT = 2


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class CLIConfig:
    project_root: Path
    copyright: str
    license_id: str
    year: str | None
    dry_run: bool
    check_only: bool


# ---------------------------------------------------------------------------
# REUSE helpers
# ---------------------------------------------------------------------------

def run_reuse_annotate(
    file_path: Path,
    *,
    copyright_holder: str,
    license_id: str,
    year: str | None,
) -> bool:
    """Annotate a single file using the 'reuse annotate' CLI.

    Returns True on success, False on failure.
    """
    cmd: list[str] = [
        "reuse",
        "annotate",
        f"--copyright={copyright_holder}",
        f"--license={license_id}",
        "--fallback-dot-license",
        "--merge-copyrights",
        str(file_path),
    ]
    if year:
        # Insert after command name but before other options (position 2)
        cmd.insert(2, f"--year={year}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Annotated: {file_path.relative_to(Path.cwd())}")
        return True
    except FileNotFoundError:
        print("Error: 'reuse' command not found. Install with 'pip install reuse'.")
        return False
    except subprocess.CalledProcessError as e:  # pragma: no cover - external tool variability
        stderr = e.stderr.strip() if e.stderr else "(no stderr)"
        print(f"✗ Failed to annotate {file_path}: {stderr}")
        return False


def generate_project_report(project_root: Path) -> ProjectReport:
    project = Project.from_directory(str(project_root))
    return ProjectReport.generate(project)


def get_files_missing_annotations(project_root: Path) -> tuple[set[Path], set[Path]]:
    """Return sets of files missing license and/or copyright."""
    try:
        report = generate_project_report(project_root)
        root = project_root
        files_without_licenses = {root / p for p in report.files_without_licenses}
        files_without_copyright = {root / p for p in report.files_without_copyright}
        return files_without_licenses, files_without_copyright
    except Exception as e:  # pragma: no cover - defensive
        print(f"Error analyzing project: {e}")
        return set(), set()


def check_reuse_compliance(project_root: Path) -> bool:
    try:
        report = generate_project_report(project_root)
        return bool(report.is_compliant)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Error checking compliance: {e}")
        return False


def validate_project_setup(project_root: Path) -> bool:
    """Validate the presence of LICENSES directory & at least one license file."""
    licenses_dir = project_root / "LICENSES"
    if not licenses_dir.is_dir():
        print(f"Error: No LICENSES directory found at {licenses_dir}")
        return False
    if not any(licenses_dir.glob("*.txt")):
        print(f"Error: No license files (*.txt) found inside {licenses_dir}")
        return False
    return True


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def annotate_files(
    files: Iterable[Path],
    *,
    copyright_holder: str,
    license_id: str,
    year: str | None,
    dry_run: bool,
) -> Tuple[int, int]:
    """Annotate all given files. Returns (success_count, total)."""
    success = 0
    file_list = sorted(files)
    if dry_run:
        print("[DRY RUN] Would annotate the following files:")
        for fp in file_list:
            print(f"  - {fp}")
        return 0, len(file_list)

    for file_path in file_list:
        if run_reuse_annotate(
            file_path,
            copyright_holder=copyright_holder,
            license_id=license_id,
            year=year,
        ):
            success += 1
    return success, len(file_list)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str]) -> CLIConfig:
    parser = argparse.ArgumentParser(
        description="Automatically annotate files missing REUSE SPDX headers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  reuse-fix.py --copyright "Jane Doe" --license MIT
  reuse-fix.py --copyright "Jane Doe" --license MIT --year 2025 /path/to/project
  reuse-fix.py --check-only --copyright "Jane" --license MIT
        """,
    )

    parser.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Project root directory (default: current working directory)",
    )

    parser.add_argument(
        "--copyright",
        required=True,
        help="Copyright holder (required)",
    )
    parser.add_argument(
        "--license",
        required=True,
        dest="license_id",
        help="SPDX License Identifier (required)",
    )
    parser.add_argument(
        "--year",
        help="Copyright year (optional; if omitted reuse will decide)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without modifying files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check compliance and list missing annotations (non-zero exit if non-compliant)",
    )

    args = parser.parse_args(argv)

    project_root = Path(args.project_path).resolve()
    return CLIConfig(
        project_root=project_root,
        copyright=args.copyright,
        license_id=args.license_id,
        year=args.year,
        dry_run=args.dry_run,
        check_only=args.check_only,
    )


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(config: CLIConfig) -> ExitCode:
    # Basic path validation
    if not config.project_root.exists():
        print(f"Error: Project path does not exist: {config.project_root}")
        return ExitCode.ERROR
    if not config.project_root.is_dir():
        print(f"Error: Project path is not a directory: {config.project_root}")
        return ExitCode.ERROR

    print(f"Analyzing project at: {config.project_root}")

    if not validate_project_setup(config.project_root):
        # Non-interactive: treat as error. User should add license files first.
        return ExitCode.ERROR

    initial_compliant = check_reuse_compliance(config.project_root)
    print(f"Initial REUSE compliance: {'✓ COMPLIANT' if initial_compliant else '✗ NON-COMPLIANT'}")

    if initial_compliant:
        return ExitCode.SUCCESS

    # Gather missing annotation files
    print("\nScanning for files missing annotations...")
    files_without_licenses, files_without_copyright = get_files_missing_annotations(
        config.project_root
    )
    to_fix = files_without_licenses | files_without_copyright

    if not to_fix:
        # Nothing to fix but earlier compliance check said non-compliant -> other issues
        print("No directly annotatable files found; run 'reuse lint' for details.")
        return ExitCode.NON_COMPLIANT

    print(f"\nFound {len(to_fix)} files needing annotation:")
    for fp in sorted(to_fix):
        rel = fp.relative_to(config.project_root)
        status_parts: list[str] = []
        if fp in files_without_licenses:
            status_parts.append("missing license")
        if fp in files_without_copyright:
            status_parts.append("missing copyright")
        print(f"  - {rel} ({', '.join(status_parts)})")

    if config.check_only:
        print("\nCheck complete (no modifications performed).")
        return ExitCode.NON_COMPLIANT

    # Perform annotation (non-interactive, immediate)
    print("\nAnnotating files...")
    success, total = annotate_files(
        to_fix,
        copyright_holder=config.copyright,
        license_id=config.license_id,
        year=config.year,
        dry_run=config.dry_run,
    )

    if config.dry_run:
        return ExitCode.NON_COMPLIANT  # Dry run cannot fix compliance

    print(f"\nAnnotation complete: {success}/{total} files processed successfully.")

    if success == 0:
        print("No files annotated; nothing changed.")
        return ExitCode.NON_COMPLIANT

    # Re-check compliance
    print("\nRechecking REUSE compliance...")
    if check_reuse_compliance(config.project_root):
        print("Final REUSE compliance: ✓ COMPLIANT")
        return ExitCode.SUCCESS

    print("Final REUSE compliance: ✗ NON-COMPLIANT")
    print("Remaining tasks may include:")
    print("  - Adding any missing license files in LICENSES/")
    print("  - Manually fixing complex or binary file annotations")
    print("  - Running 'reuse lint' for a detailed report")
    return ExitCode.NON_COMPLIANT


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = parse_args(argv or sys.argv[1:])
        return int(run(config))
    except KeyboardInterrupt:  # pragma: no cover - user interruption
        print("Aborted by user.")
        return int(ExitCode.ERROR)
    except Exception as e:  # pragma: no cover - defensive
        print(f"Unexpected error: {e}")
        return int(ExitCode.ERROR)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
