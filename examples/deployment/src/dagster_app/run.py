# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0


# Dagster's materialize helper is not fully typed; silence that noise.

"""CLI entrypoints for the Dagster example.

Run with ``python -m dagster_app.run train`` or ``python -m dagster_app.run forecast``. Each
materializes the relevant assets once per target partition. The model is handed from training
to forecasting through the shared MLflow store, so run ``train`` before ``forecast``. For the
web UI (where you can materialize all partitions with one click), use ``dagster dev`` instead.
"""

from __future__ import annotations

import sys

import dagster as dg

from dagster_app import definitions as defs


def materialize(action: str) -> None:
    """Materialize the train or forecast assets for every target partition."""
    assets = [defs.input_data, defs.trained_model] if action == "train" else [defs.input_data, defs.forecast]
    for partition_key in defs.targets_by_key:
        dg.materialize(assets, partition_key=partition_key)


if __name__ == "__main__":
    materialize(sys.argv[1] if len(sys.argv) > 1 else "forecast")
