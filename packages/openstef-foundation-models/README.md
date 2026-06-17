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
