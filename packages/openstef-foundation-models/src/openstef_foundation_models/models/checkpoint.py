# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Checkpoint resolution for foundation-model forecasters.

A *checkpoint* bundles an ONNX weights file with a :class:`CheckpointMetadata`
JSON document, written next to the weights, describing the model-family specifics
(IO tensor names, native quantile grid, context/horizon sizing, resolution).
Keeping these specifics in data rather than code keeps the inference backends
and forecasters generic.

Checkpoints can be resolved from a local path (:class:`LocalCheckpoint`) or
downloaded from the HuggingFace Hub (:class:`HubCheckpoint`). Both expose a
common :meth:`resolve` returning a :class:`ResolvedCheckpoint`.
"""

from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from openstef_core.base_model import BaseConfig
from openstef_core.types import Quantile

#: Current schema version for :class:`CheckpointMetadata`. Bumped when the
#: metadata layout changes in a backwards-incompatible way so externally
#: published checkpoints can be validated against the reader.
METADATA_SCHEMA_VERSION = 2


class CheckpointMetadata(BaseConfig):
    """Metadata describing a foundation-model checkpoint.

    This document travels next to the weights file and drives the generic
    inference path, so that no model-family specifics are hardcoded in the
    backends or forecasters.
    """

    schema_version: int = Field(
        default=METADATA_SCHEMA_VERSION,
        description="Version of the metadata layout, for forward compatibility.",
    )
    model_family: str = Field(
        description="Model family identifier, e.g. 'chronos2'.",
    )
    input_names: list[str] = Field(
        min_length=1,
        description="Ordered names of the model's input tensors.",
    )
    output_name: str = Field(
        description="Name of the model's quantile-prediction output tensor.",
    )
    native_quantiles: list[Quantile] = Field(
        min_length=1,
        description="Quantile levels the model natively emits, in ascending order.",
    )
    context_length: int = Field(
        gt=0,
        description="Number of historical timesteps the model consumes as context.",
    )
    output_patch_size: int = Field(
        gt=0,
        description="Number of timesteps produced per output patch.",
    )
    horizon_patches: int = Field(
        gt=0,
        description="Number of output patches the model emits (frozen horizon).",
    )
    resolution_minutes: int = Field(
        gt=0,
        description="Sampling interval of the series the model expects, in minutes.",
    )
    precision: Literal["fp32", "fp16", "int8"] = Field(
        default="fp32",
        description="Numeric precision of the weights. Drives provider selection: int8 (QDQ) is fast on CPU "
        "but cannot be accelerated by CoreML, fp16/fp32 follow the static-shape CoreML path on macOS.",
    )
    static_shapes: bool = Field(
        default=False,
        description="Whether ALL graph axes are frozen (no symbolic dims), making the checkpoint eligible for "
        "shape-strict runtimes such as CoreML. The frozen sizes are given by context_length and horizon_length; "
        "when False those are maxima on symbolic axes.",
    )
    max_covariates: int | None = Field(
        default=None,
        gt=0,
        description="Frozen number of covariate series the graph accepts, or None if that axis is dynamic. "
        "Independent of static_shapes (the covariate axis is not the series/batch axis): a graph may freeze "
        "the covariate count while leaving the batch dynamic. Consumed only by the forecaster, which "
        "pads/validates a series' covariate columns to it.",
    )

    @property
    def horizon_length(self) -> int:
        """Total number of forecast timesteps the model emits.

        Returns:
            ``output_patch_size * horizon_patches``.
        """
        return self.output_patch_size * self.horizon_patches


class ResolvedCheckpoint(BaseConfig):
    """A checkpoint whose weights and metadata have been materialized locally."""

    model_config = BaseConfig.model_config | {"arbitrary_types_allowed": True}

    weights_path: Path = Field(description="Local path to the resolved weights file.")
    metadata: CheckpointMetadata = Field(description="Parsed model metadata.")


def _default_metadata_path(weights_path: Path) -> Path:
    """Derive the default metadata path for a weights file.

    Args:
        weights_path: Path to the weights file (e.g. ``chronos-2.onnx``).

    Returns:
        The weights path with its final suffix replaced by ``.metadata.json``
        (e.g. ``chronos-2.metadata.json``).
    """
    return weights_path.with_suffix(".metadata.json")


class LocalCheckpoint(BaseConfig):
    """A checkpoint resolved from a local filesystem path."""

    kind: Literal["local"] = Field(
        default="local",
        description="Discriminator tag for checkpoint reference type.",
    )
    path: Path = Field(description="Path to the local weights file.")
    metadata_path: Path | None = Field(
        default=None,
        description="Path to the metadata JSON. Defaults to the weights path with a '.metadata.json' suffix.",
    )

    def resolve(self) -> ResolvedCheckpoint:
        """Resolve the local weights and metadata.

        Returns:
            A :class:`ResolvedCheckpoint` pointing at the local weights file.

        Raises:
            FileNotFoundError: If the weights or metadata file does not exist.
        """
        if not self.path.is_file():
            msg = f"Checkpoint weights not found: {self.path}"
            raise FileNotFoundError(msg)
        metadata_path = self.metadata_path or _default_metadata_path(self.path)
        if not metadata_path.is_file():
            msg = f"Checkpoint metadata not found: {metadata_path}"
            raise FileNotFoundError(msg)
        metadata = CheckpointMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))
        return ResolvedCheckpoint(weights_path=self.path, metadata=metadata)


class HubCheckpoint(BaseConfig):
    """A checkpoint resolved from a HuggingFace Hub repository."""

    kind: Literal["hub"] = Field(
        default="hub",
        description="Discriminator tag for checkpoint reference type.",
    )
    repo_id: str = Field(description="HuggingFace repository ID, e.g. 'openstef/chronos-2-onnx'.")
    filename: str = Field(description="Weights filename within the repository, e.g. 'chronos-2.onnx'.")
    metadata_filename: str | None = Field(
        default=None,
        description="Metadata filename within the repository. Defaults to the weights filename with a "
        "'.metadata.json' suffix.",
    )
    revision: str | None = Field(
        default=None,
        description="Git revision (branch, tag, or commit) to download. Defaults to the latest revision.",
    )
    repo_type: str = Field(default="model", description="HuggingFace repository type.")
    local_dir: Path | None = Field(
        default=None,
        description="Directory to download files into. Defaults to the shared HuggingFace Hub cache.",
    )

    def resolve(self) -> ResolvedCheckpoint:
        """Download the weights and metadata from the Hub and resolve them.

        Returns:
            A :class:`ResolvedCheckpoint` pointing at the downloaded weights file.
        """
        # Imported lazily so building a checkpoint config stays cheap; huggingface-hub
        # is a base dependency, so the import itself cannot fail.
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        metadata_filename = self.metadata_filename or str(_default_metadata_path(Path(self.filename)))
        weights_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type=self.repo_type,
            revision=self.revision,
            local_dir=self.local_dir,
        )
        metadata_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=metadata_filename,
            repo_type=self.repo_type,
            revision=self.revision,
            local_dir=self.local_dir,
        )
        metadata = CheckpointMetadata.model_validate_json(Path(metadata_path).read_text(encoding="utf-8"))
        return ResolvedCheckpoint(weights_path=Path(weights_path), metadata=metadata)


#: A checkpoint reference, discriminated by its ``kind`` tag.
CheckpointRef = Annotated[LocalCheckpoint | HubCheckpoint, Field(discriminator="kind")]
