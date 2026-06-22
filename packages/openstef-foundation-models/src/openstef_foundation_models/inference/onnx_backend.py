# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""ONNX Runtime execution backend.

Importing this module requires ONNX Runtime (the ``[cpu]`` or ``[gpu]`` extra)
and raises :class:`MissingExtraError` if it is missing.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Self

import numpy as np

from openstef_core.exceptions import MissingExtraError
from openstef_foundation_models.inference.provider_selection import (
    DefaultProviderPolicy,
    HostCapabilities,
    ProviderPolicy,
)
from openstef_foundation_models.inference.providers import (
    ExecutionProvider,
    SessionOptionsConfig,
)
from openstef_foundation_models.models.checkpoint import CheckpointMetadata, ResolvedCheckpoint

try:
    import onnxruntime as ort
except ImportError as e:
    raise MissingExtraError("onnxruntime", "openstef-foundation-models", install_extra="cpu") from e

# onnxruntime-gpu ships the CUDA execution-provider plugin but loads the CUDA/cuDNN
# runtime (the nvidia-*-cu12 wheels the [gpu] extra pulls) lazily at session creation.
# preload_dlls() loads them from the nvidia site-packages so the CUDA provider can be
# realized without a system CUDA install or LD_LIBRARY_PATH. It is a no-op on the CPU
# runtime; the guard covers onnxruntime < 1.21, which predates the API.
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()

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
        policy: ProviderPolicy | None = None,
    ) -> Self:
        """Build a backend by loading a checkpoint into a new ONNX Runtime session.

        With ``providers=None`` the *policy* selects a chain from the checkpoint
        and host; an explicit ``providers`` list is used as given and *policy* is
        ignored. See :class:`~openstef_foundation_models.inference.provider_selection.ProviderPolicy`
        for how a chain is chosen and how strictly its realization is enforced.

        Args:
            checkpoint: The resolved checkpoint (weights + metadata) to load.
            providers: Ordered execution providers to try. ``None`` lets the policy
                pick a host-appropriate chain from the checkpoint metadata.
            session_options: Optional ONNX Runtime session options.
            policy: Selection policy used when ``providers is None``. Defaults to
                :class:`DefaultProviderPolicy`.

        Returns:
            A backend wrapping the newly built session.
        """
        metadata = checkpoint.metadata
        if providers is not None:
            provider_configs = list(providers)
            explicit = True
            logger.info(
                "Using explicit execution-provider chain %s for checkpoint '%s'.",
                [config.to_ort()[0] for config in provider_configs],
                metadata.model_family,
            )
        else:
            selector = policy or DefaultProviderPolicy()
            host = HostCapabilities.detect()
            provider_configs = selector.select(metadata, host)
            explicit = False
            logger.debug(
                "Detected host: platform=%s, available_providers=%s",
                host.platform,
                sorted(host.available_providers),
            )
            logger.info(
                "%s selected execution-provider chain %s for checkpoint '%s' (precision=%s, static_shapes=%s) on %s.",
                type(selector).__name__,
                [config.to_ort()[0] for config in provider_configs],
                metadata.model_family,
                metadata.precision,
                metadata.static_shapes,
                host.platform,
            )
        ort_providers = [config.to_ort() for config in provider_configs]
        so = _build_session_options(session_options) if session_options else None

        session = ort.InferenceSession(
            str(checkpoint.weights_path),
            sess_options=so,
            providers=ort_providers,
        )
        logger.info("ONNX Runtime session built; realized providers: %s.", session.get_providers())
        _check_provider_fallback(
            requested=provider_configs,
            realized=session.get_providers(),
            strict=explicit,
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

    Compares the requested chain against what ONNX Runtime actually realized. See
    :class:`~openstef_foundation_models.inference.provider_selection.ProviderPolicy`
    for the strict-vs-graceful contract this enforces.

    Args:
        requested: The execution providers that were requested.
        realized: The provider names ONNX Runtime actually loaded.
        strict: When ``True``, raise on any missing accelerator; otherwise warn
            only on a full fallback to CPU.

    Raises:
        RuntimeError: If ``strict`` is set and a requested accelerator is missing.
    """
    requested_names = {config.to_ort()[0] for config in requested}
    accelerators = requested_names - {"CPUExecutionProvider"}
    if not accelerators:
        return
    realized_set = set(realized)
    missing = accelerators - realized_set
    if strict:
        if missing:
            msg = (
                f"Requested execution provider(s) {sorted(missing)} were not realized; "
                f"ONNX Runtime fell back to {realized}."
            )
            raise RuntimeError(msg)
        return
    if accelerators & realized_set:
        return
    logger.warning(
        "No requested accelerator (%s) was realized; ONNX Runtime fell back to %s. Inference will run on CPU.",
        sorted(accelerators),
        realized,
    )
