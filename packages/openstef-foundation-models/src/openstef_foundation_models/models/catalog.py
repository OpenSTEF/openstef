# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Catalog of the foundation-model checkpoints OpenSTEF publishes.

OpenSTEF publishes each model size to its own HuggingFace repo,
``OpenSTEF/<slug>-onnx``, holding a few ONNX *variants* of the same weights. This
module mirrors that naming convention so a user selects a checkpoint by size and
variant instead of hand-typing repo ids and filenames. The strings here are the
wire contract with the publisher (``openstef-checkpoints``); keeping them in one
place is what lets the two repos stay in step.

Two variants matter when selecting:

* :attr:`CheckpointVariant.DYNAMIC` — symbolic shapes; runs on every execution
  provider and is the portable default.
* :attr:`CheckpointVariant.STATIC` — frozen shapes; on macOS this is what lets the
  CoreML provider engage in the default fallback chain, so prefer it there.

The module is import-light (pure pydantic config, no inference runtime), so a
checkpoint can be selected and a config built without ONNX Runtime installed.

Example::

    from openstef_foundation_models.models.catalog import Chronos2, CheckpointVariant

    checkpoint = Chronos2.BASE.checkpoint(CheckpointVariant.STATIC)
"""

from __future__ import annotations

import platform
from enum import StrEnum

from openstef_foundation_models.models.checkpoint import HubCheckpoint

#: HuggingFace namespace the checkpoints are published under. Shared by every
#: published size, so it has no single model to live on.
HF_NAMESPACE = "OpenSTEF"


class CheckpointVariant(StrEnum):
    """A published ONNX variant of a model's weights."""

    DYNAMIC = "dynamic"
    STATIC = "static"

    @property
    def filename_suffix(self) -> str:
        """The suffix this variant adds to the model slug in the weights filename.

        Returns:
            ``'_static'`` for the static-shape variant, ``''`` for the dynamic one.
        """
        return "_static" if self is CheckpointVariant.STATIC else ""

    @classmethod
    def recommended(cls) -> CheckpointVariant:
        """The variant to prefer on the host running this code.

        Returns :attr:`STATIC` on macOS, where frozen shapes let the CoreML
        provider engage in the default fallback chain, and :attr:`DYNAMIC`
        everywhere else. The choice is by platform only — it never imports the
        inference runtime — so static is recommended on macOS even when CoreML is
        absent, where it simply runs on CPU like the dynamic build.

        Returns:
            The recommended variant for this host.
        """
        return cls.STATIC if platform.system().lower() == "darwin" else cls.DYNAMIC


class Chronos2(StrEnum):
    """The published Chronos-2 model sizes, each selectable as a Hub checkpoint.

    The member value is the model slug, which is also the HuggingFace repo stem
    (``OpenSTEF/<slug>-onnx``) and the weights-filename stem.
    """

    BASE = "chronos-2"
    SMALL = "chronos-2-small"

    def checkpoint(self, variant: CheckpointVariant = CheckpointVariant.DYNAMIC) -> HubCheckpoint:
        """Build the Hub checkpoint reference for this size and *variant*.

        Args:
            variant: Which published ONNX variant to load. Defaults to the
                portable dynamic-shape build; pass :attr:`CheckpointVariant.STATIC`
                (or :meth:`CheckpointVariant.recommended`) on macOS for CoreML.

        Returns:
            A :class:`~openstef_foundation_models.models.checkpoint.HubCheckpoint`
            pointing at the published weights and their metadata.
        """
        return HubCheckpoint(
            repo_id=f"{HF_NAMESPACE}/{self.value}-onnx",
            filename=f"{self.value}{variant.filename_suffix}.onnx",
        )


__all__ = [
    "CheckpointVariant",
    "Chronos2",
]
