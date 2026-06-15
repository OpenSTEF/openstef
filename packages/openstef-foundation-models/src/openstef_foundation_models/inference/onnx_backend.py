# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""ONNX Runtime execution backend.

Importing this module requires ONNX Runtime (the ``[cpu]`` or ``[gpu]`` extra).
The import fails early and loudly with :class:`MissingExtraError` if it is not
installed, rather than deferring the failure to inference time.
"""

import logging
from collections.abc import Mapping, Sequence

import numpy as np

from openstef_core.exceptions import MissingExtraError
from openstef_foundation_models.inference.providers import (
    CpuProvider,
    ExecutionProvider,
    SessionOptionsConfig,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata, ResolvedCheckpoint

try:
    import onnxruntime as ort
except ImportError as e:
    raise MissingExtraError("onnxruntime", "openstef-foundation-models") from e

logger = logging.getLogger(__name__)


class OnnxBackend:
    """An :class:`~openstef_foundation_models.inference.backend.InferenceBackend` backed by ONNX Runtime.

    The session is built once on construction and reused for every
    :meth:`run` call, so a single backend instance can be shared across an
    entire backtest. Users may either let the backend build a session from a
    resolved checkpoint and provider chain, or pass a pre-built session they own.
    """

    def __init__(
        self,
        metadata: CheckpointMetadata,
        session: ort.InferenceSession,
    ) -> None:
        """Wrap a pre-built ONNX Runtime session.

        Prefer :meth:`from_checkpoint` unless you need to own the session
        lifecycle yourself.

        Args:
            metadata: Metadata describing the checkpoint the session executes.
            session: A pre-built ONNX Runtime inference session.
        """
        self._metadata = metadata
        self._session: ort.InferenceSession | None = session

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: ResolvedCheckpoint,
        providers: Sequence[ExecutionProvider] | None = None,
        session_options: SessionOptionsConfig | None = None,
        *,
        strict_providers: bool = False,
    ) -> "OnnxBackend":
        """Build a backend by loading a checkpoint into a new ONNX Runtime session.

        Args:
            checkpoint: The resolved checkpoint (weights + metadata) to load.
            providers: Ordered execution providers to try. Defaults to CPU only.
            session_options: Optional ONNX Runtime session options.
            strict_providers: When ``True``, raise if the realized provider chain
                falls back to CPU despite an accelerator being requested. When
                ``False`` (default), only warn.

        Returns:
            A backend wrapping the newly built session.
        """
        provider_configs = list(providers) if providers else [CpuProvider()]
        ort_providers = [config.to_ort() for config in provider_configs]
        so = _build_session_options(session_options) if session_options else None

        session = ort.InferenceSession(
            str(checkpoint.weights_path),
            sess_options=so,
            providers=ort_providers,
        )
        _check_provider_fallback(
            requested=provider_configs,
            realized=session.get_providers(),
            strict=strict_providers,
        )
        return cls(metadata=checkpoint.metadata, session=session)

    @property
    def metadata(self) -> CheckpointMetadata:
        """Metadata describing the checkpoint this backend executes."""
        return self._metadata

    def run(self, inputs: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        """Execute the ONNX model on a batch of named input tensors.

        Args:
            inputs: Named input tensors. Keys must match ``metadata.input_names``.

        Returns:
            Named output tensors keyed by the model's output names.

        Raises:
            RuntimeError: If the backend has been closed.
        """
        if self._session is None:
            msg = "OnnxBackend has been closed."
            raise RuntimeError(msg)
        output_names = [out.name for out in self._session.get_outputs()]
        results = self._session.run(output_names, dict(inputs))
        return {name: np.asarray(result) for name, result in zip(output_names, results, strict=True)}

    def close(self) -> None:
        """Release the underlying ONNX Runtime session.

        ONNX Runtime frees native resources on garbage collection, so dropping
        the reference is the supported way to release them.
        """
        self._session = None


def _build_session_options(config: SessionOptionsConfig) -> ort.SessionOptions:
    """Translate a :class:`SessionOptionsConfig` into ONNX Runtime session options.

    Args:
        config: The typed session-options configuration.

    Returns:
        The corresponding ONNX Runtime ``SessionOptions``.
    """
    so = ort.SessionOptions()
    so.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel,
        f"ORT_{config.graph_optimization_level}",
    )
    if config.intra_op_num_threads is not None:
        so.intra_op_num_threads = config.intra_op_num_threads
    if config.inter_op_num_threads is not None:
        so.inter_op_num_threads = config.inter_op_num_threads
    return so


def _check_provider_fallback(
    requested: Sequence[ExecutionProvider],
    realized: Sequence[str],
    *,
    strict: bool,
) -> None:
    """Detect and report a silent fallback to the CPU execution provider.

    ONNX Runtime silently drops accelerators it cannot initialize (missing
    libraries, unsupported ops) and falls back to CPU. This compares the
    requested chain against what was actually realized and warns (or raises).

    Args:
        requested: The execution providers that were requested.
        realized: The provider names ONNX Runtime actually loaded.
        strict: When ``True``, raise instead of warning.

    Raises:
        RuntimeError: If ``strict`` is set and an accelerator fell back to CPU.
    """
    requested_names = {config.to_ort()[0] for config in requested}
    accelerators = requested_names - {"CPUExecutionProvider"}
    if not accelerators:
        return
    realized_set = set(realized)
    missing = accelerators - realized_set
    if not missing:
        return
    msg = (
        f"Requested execution provider(s) {sorted(missing)} were not realized; "
        f"ONNX Runtime fell back to {realized}. Inference will run on CPU."
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)
