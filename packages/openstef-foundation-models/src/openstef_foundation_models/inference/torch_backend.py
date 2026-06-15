# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Torch execution backend.

Importing this module requires PyTorch (the ``[torch]`` extra). The import fails
early and loudly with :class:`MissingExtraError` if it is not installed.

This backend is **model-independent**: it wraps any ``torch.nn.Module`` that
follows the named-tensor contract and makes no assumptions about which model it
runs. Loading a *specific* model (e.g. Chronos-2) into a module is the caller's
responsibility — see ``scripts/load_chronos2_torch.py`` for a throwaway helper
used by the ONNX-vs-Torch benchmark comparison.
"""

import logging
from collections.abc import Mapping

import numpy as np

from openstef_core.exceptions import MissingExtraError
from openstef_foundation_models.models.checkpoint import CheckpointMetadata

try:
    import torch  # ty: ignore[unresolved-import]
except ImportError as e:
    raise MissingExtraError("torch", "openstef-foundation-models") from e

logger = logging.getLogger(__name__)


class TorchBackend:
    """An :class:`~openstef_foundation_models.inference.backend.InferenceBackend` backed by a Torch module.

    The module is held for the lifetime of the backend and reused for every
    :meth:`run` call. Inference runs under ``torch.inference_mode`` on the
    configured device. The backend is model-agnostic: it only requires that the
    module accept the named input tensors and return a single output tensor.
    """

    def __init__(
        self,
        metadata: CheckpointMetadata,
        module: torch.nn.Module,
        device: str = "cpu",
    ) -> None:
        """Wrap a pre-built Torch module.

        Args:
            metadata: Metadata describing the model the module executes.
            module: The Torch module to run inference with.
            device: The Torch device to run on (e.g. ``"cpu"``, ``"cuda"``).
        """
        self._metadata = metadata
        self._device = device
        self._module: torch.nn.Module | None = module.to(device).eval()

    @property
    def metadata(self) -> CheckpointMetadata:
        """Metadata describing the checkpoint this backend executes."""
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Execute the Torch module on a batch of named input tensors.

        Args:
            inputs: Named input tensors. Keys must match ``metadata.input_names``.

        Returns:
            A single named output tensor keyed by ``metadata.output_name``.

        Raises:
            RuntimeError: If the backend has been closed.
        """
        if self._module is None:
            msg = "TorchBackend has been closed."
            raise RuntimeError(msg)
        torch_inputs = {
            name: torch.from_numpy(np.asarray(value)).to(self._device) for name, value in inputs.items()
        }
        with torch.inference_mode():
            output = self._module(**torch_inputs)
        array = output.detach().cpu().numpy() if isinstance(output, torch.Tensor) else np.asarray(output)
        return {self._metadata.output_name: array}

    def close(self) -> None:
        """Release the underlying Torch module."""
        self._module = None
