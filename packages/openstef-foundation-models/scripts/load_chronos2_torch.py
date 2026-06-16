# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""THROWAWAY: load Chronos-2 as a Torch :class:`TorchBackend`.

This helper exists solely to obtain an apples-to-apples Torch reference for the
ONNX-vs-Torch benchmark comparison while the ONNX export is being validated. It
intentionally lives outside the package's import graph (a script, not a module)
because it is model-specific glue and will be **removed** once the comparison is
done — the :class:`TorchBackend` itself is model-independent and must stay that way.

REMOVE WHEN THE TORCH BACKEND IS DROPPED: delete this script together with
``inference/torch_backend.py`` and ``tests/integration/test_onnx_torch_parity.py``.

Requires the optional ``[torch]`` extra plus ``chronos-forecasting``::

    uv run --extra torch python packages/openstef-foundation-models/scripts/load_chronos2_torch.py
"""
# ruff: noqa: INP001  # throwaway script, intentionally not part of a package

from openstef_foundation_models.inference.torch_backend import TorchBackend
from openstef_foundation_models.models.checkpoint import CheckpointMetadata


def load_chronos2_torch_backend(
    metadata: CheckpointMetadata,
    model_id: str = "amazon/chronos-2",
    device: str = "cpu",
) -> TorchBackend:
    """Load the Chronos-2 Torch module and wrap it in a :class:`TorchBackend`.

    Args:
        metadata: Metadata describing the Chronos-2 checkpoint.
        model_id: HuggingFace model id to load the Torch weights from.
        device: Torch device to run on.

    Returns:
        A :class:`TorchBackend` wrapping the loaded Chronos-2 module.

    Raises:
        ImportError: When ``chronos-forecasting`` is not installed.
    """
    try:
        from chronos import BaseChronosPipeline  # noqa: PLC0415
    except ImportError as e:
        msg = "load_chronos2_torch_backend requires chronos-forecasting: pip install chronos-forecasting"
        raise ImportError(msg) from e

    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map=device)
    return TorchBackend(metadata=metadata, module=pipeline.model, device=device)
