# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Check that committed .ipynb notebooks have no cell outputs."""

import json
import sys
from pathlib import Path

TUTORIALS_DIR = Path("examples/tutorials")
BENCHMARKS_DIRS = [Path("examples/benchmarks/custom"), Path("examples/benchmarks/liander2024")]


def main() -> None:
    """Validate that no .ipynb in tutorials or benchmarks has stored outputs."""
    failures: list[str] = []
    checked = 0

    for search_dir in [TUTORIALS_DIR, *BENCHMARKS_DIRS]:
        for nb_path in sorted(search_dir.glob("*.ipynb")):
            checked += 1
            nb = json.loads(nb_path.read_text(encoding="utf-8"))
            for i, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code" and cell.get("outputs"):
                    failures.append(f"  {nb_path.relative_to('.')}: cell {i} has outputs")
                    break

    if failures:
        print("ERROR: Notebooks with outputs found (run `poe notebooks-clear`):")
        print("\n".join(failures))
        sys.exit(1)

    print(f"OK: {checked} notebooks checked, no outputs found")


if __name__ == "__main__":
    main()
