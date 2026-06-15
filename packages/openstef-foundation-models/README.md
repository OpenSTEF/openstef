<!--
SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# openstef-foundation-models

Foundation model support for OpenSTEF — bringing pre-trained, ONNX-based forecasting models to the OpenSTEF ecosystem.

## Installation

CPU (default):

```bash
pip install openstef-foundation-models
```

GPU (CUDA — replaces the CPU runtime):

```bash
pip install "openstef-foundation-models[gpu]"
```

> **Note:** Do **not** install both `onnxruntime` and `onnxruntime-gpu` in the same environment.
