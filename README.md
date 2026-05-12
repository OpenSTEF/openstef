<!--
SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

<p align="center">
  <img src="docs/logos/openstef-horizontal-color.svg" alt="OpenSTEF Logo" width="400">
</p>

# OpenSTEF

[![Downloads](https://static.pepy.tech/badge/openstef)](https://pepy.tech/project/openstef)
[![Downloads](https://static.pepy.tech/badge/openstef/month)](https://pepy.tech/project/openstef)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5585/badge)](https://bestpractices.coreinfrastructure.org/projects/5585)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v4.0.0-blue.svg)](https://github.com/paula-passet/openstef_Sia/releases/tag/release/v4.0.0)

## What is OpenSTEF

**OpenSTEF** (Open Short-Term Energy Forecasting) is an open-source, model-agnostic Python framework for creating short-term forecasts in the energy sector. It provides complete machine learning pipelines for data preprocessing, feature engineering, model training, probabilistic forecasting, evaluation, and post-processing. OpenSTEF generates forecasts with uncertainty bandwidths and includes built-in domain knowledge specific to energy forecasting.

For more information, visit the [OpenSTEF project page on LF Energy](https://www.lfenergy.org/projects/openstef/).

## Monorepo Overview

OpenSTEF 4.0 is organized as a monorepo with specialized packages under the `packages/` directory:

| Package | Purpose |
|---------|---------|
| **openstef** | Meta-package that installs all core components |
| **openstef-core** | Core utilities, dataset types, shared types and base models |
| **openstef-models** | ML models, feature engineering, data processing |
| **openstef-beam** | Backtesting, Evaluation, Analysis, and Metrics |
| **openstef-meta** | Meta models for OpenSTEF |

## How to Install

**Requirements:** Python ≥3.12, <4.0

```bash
# Install the complete framework
pip install openstef

# Or install individual packages
pip install openstef-models
pip install openstef-beam
pip install openstef-core

# Using uv (recommended for development)
uv add openstef
```

For optional features:

```bash
pip install "openstef-models[lgbm]"       # LightGBM support
pip install "openstef-models[xgb-cpu]"    # XGBoost CPU support
pip install "openstef-models[tuning]"     # Optuna hyperparameter tuning
```

See the [Complete Installation Guide](https://openstef.github.io/openstef/v4/user_guide/installation.html) for detailed instructions including troubleshooting.

## Examples

Explore the [`examples/`](examples/) folder for runnable demonstrations of OpenSTEF's capabilities. The folder contains its own `README.md` with an overview of available examples.

For step-by-step tutorials, see the [documentation tutorials](https://openstef.github.io/openstef/v4/user_guide/tutorials.html).

## License

**Mozilla Public License Version 2.0** - see [LICENSE.md](LICENSE.md) for details.

This project includes third-party libraries licensed under their respective Open-Source licenses. SPDX-License-Identifier headers show applicable licenses. License files are in the [LICENSES/](LICENSES/) directory.
## Contributing

We welcome contributions to OpenSTEF 4.0! 

**[Read our Contributing Guide](https://openstef.github.io/openstef/v4/contribute/)** - documentation for contributors including:

- How to report bugs and suggest features
- Documentation improvements and examples
- Code contributions and development setup
- Sharing datasets and real-world use cases

### Quick Development Setup

```bash
# Clone and set up for development
git clone https://github.com/OpenSTEF/openstef.git
cd openstef
uv sync --dev

# Run tests and quality checks
uv run poe all
```

**Code of Conduct**: We follow the [Contributor Code of Conduct](https://openstef.github.io/openstef/v4/contribute/code_of_conduct.html) to ensure a welcoming environment for all contributors.
## Citations

If you use OpenSTEF in your research, please cite the project. Refer to the [OpenSTEF project page](https://www.lfenergy.org/projects/openstef/) for the latest citation guidance and any associated publications.

## Contact

- **Slack:** [LF Energy Slack](https://slack.lfenergy.org/)
- **Email:** openstef@lfenergy.org
- **Community meeting:** [OpenSTEF four-weekly community meeting](https://lf-energy.atlassian.net/wiki/spaces/OS/pages/32278358/OpenSTEF+four-weekly+community+meeting)
- **Issue Tracker:** [GitHub Issues](https://github.com/OpenSTEF/openstef/issues)