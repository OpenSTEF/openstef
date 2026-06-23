<!--
SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# openstef-foundation-models

Foundation model support for OpenSTEF — bringing pre-trained, ONNX-based forecasting models to the OpenSTEF ecosystem.

## Installation

Pick exactly one ONNX runtime — `[cpu]` and `[gpu]` are mutually exclusive.

CPU (default — the meta-package `openstef` installs this for you):

```bash
pip install "openstef-foundation-models[cpu]"
```

GPU (CUDA):

```bash
pip install "openstef-foundation-models[gpu]"
```

> **Note:** Do **not** install both `[cpu]` and `[gpu]` in the same environment —
> `onnxruntime` and `onnxruntime-gpu` collide. They're declared as conflicting
> extras so uv enforces the choice; `pip` does not, so pick one yourself.

## Selecting a checkpoint

OpenSTEF publishes its checkpoints to the HuggingFace Hub. Pick a model size and
variant from the catalog instead of hand-writing repo ids and filenames:

```python
from openstef_foundation_models.models import Chronos2, CheckpointVariant
from openstef_foundation_models.presets import ForecastingWorkflowConfig

# Default: the base Chronos-2, dynamic shapes — runs on any provider.
config = ForecastingWorkflowConfig()

# The smaller model, static shapes — enables the CoreML provider on macOS.
config = ForecastingWorkflowConfig(
    checkpoint=Chronos2.SMALL.checkpoint(CheckpointVariant.STATIC),
)

# Let the host decide: static on macOS, dynamic elsewhere.
config = ForecastingWorkflowConfig(
    checkpoint=Chronos2.BASE.checkpoint(CheckpointVariant.recommended()),
)
```

Available sizes are `Chronos2.BASE` (`chronos-2`) and `Chronos2.SMALL`
(`chronos-2-small`); variants are `DYNAMIC` (portable) and `STATIC` (frozen shapes,
the macOS/CoreML path in the default provider fallback chain).
