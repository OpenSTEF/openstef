# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""The :class:`InferenceBackend` contract shared by all execution backends."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np

from openstef_foundation_models.models.checkpoint import CheckpointMetadata


@runtime_checkable
class InferenceBackend(Protocol):
    """A model-agnostic execution backend.

    A backend takes a mapping of named input tensors to a mapping of named
    output tensors. It owns whatever runtime resources are needed (e.g. an ONNX
    Runtime session) and is loaded once, then reused across an entire backtest.
    Model-family specifics live in :attr:`metadata`, not in the backend itself.
    """

    @property
    def metadata(self) -> CheckpointMetadata:
        """Metadata describing the checkpoint this backend executes."""
        ...

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Execute the model on a batch of named input tensors.

        Args:
            inputs: Named input tensors. Keys must match ``metadata.input_names``.

        Returns:
            Named output tensors, including ``metadata.output_name``.
        """
        ...

    def close(self) -> None:
        """Release any runtime resources held by the backend."""
        ...
