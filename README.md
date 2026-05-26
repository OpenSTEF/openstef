<!--
SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

<p align="center">
  <a href="https://www.lfenergy.org/projects/openstef/">
    <img src="https://raw.githubusercontent.com/OpenSTEF/openstef/main/docs/source/_static/logo.png" alt="OpenSTEF Logo" width="400">
  </a>
</p>

# OpenSTEF

[![Downloads](https://static.pepy.tech/badge/openstef)](https://pepy.tech/project/openstef)
[![Downloads](https://static.pepy.tech/badge/openstef/month)](https://pepy.tech/project/openstef)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5585/badge)](https://bestpractices.coreinfrastructure.org/projects/5585)
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE.md)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Release](https://img.shields.io/badge/release-v4.0.0-blue.svg)](https://github.com/paula-passet/openstef_Sia/releases/tag/release/v4.0.0)

## What is OpenSTEF

**OpenSTEF** (Open Short-Term Energy Forecasting) is an open-source, model-agnostic Python framework for creating short-term forecasts in the energy sector. It provides complete machine learning pipelines for data preprocessing, feature engineering, model training, probabilistic forecasting, and evaluation. Version 4.0.0 introduces a complete architectural refactor with enhanced modularity, full type safety, and modern Python development practices.

For more information, visit the [OpenSTEF project page at LF Energy](https://www.lfenergy.org/projects/openstef/).

## Monorepo Overview

OpenSTEF 4.0.0 is organized as a monorepo with specialized packages under the `packages/` directory:

| Package | Purpose |
|---------|---------|
| **openstef** | Meta-package installing all core components |
| **openstef-core** | Core utilities, dataset types, shared types and base models |
| **openstef-models** | ML models, feature engineering, data processing |
| **openstef-beam** | Backtesting, Evaluation, Analysis, and Metrics |
| **openstef-meta** | Meta models for OpenSTEF |

## How to Install

**Requirements:** Python ≥3.12, 64-bit OS (Windows, macOS, Linux)

```bash
# Install the complete framework
pip install openstef

# Or install individual packages
pip install openstef-models
pip install openstef-beam
pip install openstef-core

# With optional features
pip install "openstef-models[lgbm]"
pip install "openstef-models[xgb-cpu]"
pip install "openstef-beam[all]"
```

Using [uv](https://docs.astral.sh/uv/) (recommended for development):

```bash
uv add openstef
```

For the complete installation guide including troubleshooting, see the [Installation Documentation](https://openstef.github.io/openstef/user_guide/installation.html).

## Examples

Tutorials and usage examples are available in the [`examples/`](examples/) folder. See the examples folder's own `README.md` for an overview of available examples including:

- Forecasting quickstart
- Feature engineering

Additional tutorials are available in the [online documentation](https://openstef.github.io/openstef/user_guide/tutorials.html).

## License

**Mozilla Public License Version 2.0** - see [LICENSE.md](LICENSE.md) for details.

This project includes third-party libraries licensed under their respective Open-Source licenses. SPDX-License-Identifier headers show applicable licenses. License files are in the [LICENSES/](LICENSES/) directory.
## Contributing

We welcome contributions to OpenSTEF 4.0! 

**[Read our Contributing Guide](https://openstef.github.io/openstef/contribute/)** - documentation for contributors including:

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

**Code of Conduct**: We follow the [Contributor Code of Conduct](https://openstef.github.io/openstef/contribute/code_of_conduct.html) to ensure a welcoming environment for all contributors.
## Citations

If you use OpenSTEF in your research or publications, please cite the project. Refer to the [CITATION.cff](CITATION.cff) file in this repository for the preferred citation format, or use:

> Contributors to the OpenSTEF project. *OpenSTEF — Open Short-Term Energy Forecasting*. LF Energy, 2017–2025. Available at: https://github.com/OpenSTEF/openstef

## Contact

- **Slack:** [LF Energy Slack](https://slack.lfenergy.org/)
- **Email:** openstef@lfenergy.org
- **Community meeting:** [OpenSTEF four-weekly community meeting](https://lf-energy.atlassian.net/wiki/spaces/OS/pages/32278358/OpenSTEF+four-weekly+community+meeting)
- **Issues:** [GitHub Issue Tracker](https://github.com/OpenSTEF/openstef/issues)
- **Support Guide:** [How to get help](https://openstef.github.io/openstef/project/support.html)