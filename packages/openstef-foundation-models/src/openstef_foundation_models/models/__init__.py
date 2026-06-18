# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Foundation-model artifacts: checkpoints and forecasters.

The published-checkpoint catalog is re-exported here so users select a checkpoint
with one short import — ``Chronos2.BASE.checkpoint(CheckpointVariant.STATIC)`` —
without reaching into a submodule.
"""

from openstef_foundation_models.models.catalog import CheckpointVariant, Chronos2

__all__ = [
    "CheckpointVariant",
    "Chronos2",
]
