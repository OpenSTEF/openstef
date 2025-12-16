<!--
SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>

SPDX-License-Identifier: MPL-2.0
-->

# OpenSTEF

<!-- Badges -->

[![Downloads](https://static.pepy.tech/badge/openstef)](https://pepy.tech/project/openstef)
[![Downloads](https://static.pepy.tech/badge/openstef/month)](https://pepy.tech/project/openstef)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5585/badge)](https://bestpractices.coreinfrastructure.org/projects/5585)

**OpenSTEF** is a modular library for creating short-term forecasts in the energy sector. Version 4.0 introduces a complete architectural refactor with enhanced modularity, type safety, and modern Python development practices.

## What's New in 4.0

- **Modular Architecture**: Install only the components you need
- **Modern Tooling**: Built with uv, ruff, pyright, and poe for optimal developer experience
- **Full Type Safety**: Comprehensive type hints throughout the codebase
- **Monorepo Structure**: Unified development with specialized packages
- **Enhanced Workflows**: Streamlined development and contribution processes

## Table of Contents

- [OpenSTEF](#openstef)
  - [What's New in 4.0](#whats-new-in-40)
  - [Table of Contents](#table-of-contents)
  - [Quick Start](#quick-start)
  - [Package Architecture](#package-architecture)
  - [Installation](#installation)
    - [Requirements](#requirements)
    - [Basic Installation](#basic-installation)
    - [Modern Package Managers](#modern-package-managers)
  - [Documentation](#documentation)
  - [Contributing](#contributing)
    - [Quick Development Setup](#quick-development-setup)
  - [License](#license)
  - [Contact \& Support](#contact--support)

## Quick Start

```bash
# Install OpenSTEF
pip install openstef

# Start forecasting
python -c "import openstef_models; print('OpenSTEF 4.0 ready!')"
```

**[Get started with our Quick Start Guide](https://openstef.github.io/openstef/v4/user_guide/quick_start.html)** - step-by-step tutorial with real examples.

## Package Architecture

OpenSTEF 4.0 uses a modular design with specialized packages:

| Package | Purpose | Install Command |
|---------|---------|-----------------|
| **openstef** | Meta-package with core components | `pip install openstef` |
| **openstef-models** | ML models, feature engineering, data processing | `pip install openstef-models` |
| **openstef-beam** | Backtesting, Evaluation, Analysis, Metrics | `pip install openstef-beam` |
| **openstef-core** | Core utilities, dataset types, shared types and base models used by other packages | `pip install openstef-core` |
| **openstef-compatibility** | OpenSTEF 3.x compatibility layer | Coming soon |
| **openstef-foundational-models** | Deep learning and foundational models | Coming soon |

**[Learn more about the architecture](https://openstef.github.io/openstef/v4/user_guide/installation.html#package-architecture)** in our documentation.

## Installation

### Requirements
- **Python 3.12+** (Python 3.13 supported)
- **64-bit operating system** (Windows, macOS, Linux)

### Basic Installation

```bash
# For most users
pip install openstef

# Core forecasting only
pip install openstef-models

# With all optional tools
pip install "openstef[all]"
```

### Modern Package Managers

```bash
# Using uv (recommended for development)
uv add openstef

# Using conda
conda install -c conda-forge openstef
```

**[Complete Installation Guide](https://openstef.github.io/openstef/v4/user_guide/installation.html)** - detailed instructions including troubleshooting for Apple Silicon, GPU support, and development setup.

## Documentation

- **[Main Documentation](https://openstef.github.io/openstef/v4/)** - guides and API reference
- **[Quick Start Guide](https://openstef.github.io/openstef/v4/user_guide/quick_start.html)** - get up and running fast
- **[Tutorials](https://openstef.github.io/openstef/v4/user_guide/tutorials.html)** - step-by-step examples
- **[API Reference](https://openstef.github.io/openstef/v4/api/)** - detailed function documentation
- **[Contributing Guide](https://openstef.github.io/openstef/v4/contribute/)** - how to contribute to OpenSTEF

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

## License

**Mozilla Public License Version 2.0** - see [LICENSE.md](LICENSE.md) for details.

This project includes third-party libraries licensed under their respective Open-Source licenses. SPDX-License-Identifier headers show applicable licenses. License files are in the [LICENSES/](LICENSES/) directory.

## Contact & Support

- **[Support Guide](https://openstef.github.io/openstef/v4/project/support.html)** - how to get help
- **[GitHub Discussions](https://github.com/OpenSTEF/openstef/discussions)** - community Q&A and discussions
- **[Issue Tracker](https://github.com/OpenSTEF/openstef/issues)** - bug reports and feature requests
- **[LF Energy OpenSTEF](https://www.lfenergy.org/projects/openstef/)** - project homepage
