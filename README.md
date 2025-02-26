<!--
SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# OpenSTEF

<!-- Badges -->

[![Downloads](https://static.pepy.tech/badge/openstef)](https://pepy.tech/project/openstef)
[![Downloads](https://static.pepy.tech/badge/openstef/month)](https://pepy.tech/project/openstef)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5585/badge)](https://bestpractices.coreinfrastructure.org/projects/5585)

<!-- SonarCloud badges -->

[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=bugs)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=code_smells)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=coverage)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=duplicated_lines_density)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=security_rating)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=sqale_index)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=OpenSTEF_openstef&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=OpenSTEF_openstef)

OpenSTEF is a Python package designed for generating short-term forecasts in the energy sector. The repository includes all the essential components required for machine learning pipelines that facilitate the forecasting process. To utilize the package, users are required to furnish their own data storage and retrieval interface.

# Table of contents

- [OpenSTEF](#openstef)
- [Table of contents](#table-of-contents)
- [External information sources](#external-information-sources)
- [Installation](#installation)
- [Usage](#usage)
  - [Example notebooks](#example-notebooks)
  - [Reference Implementation](#reference-implementation)
  - [Database connector for OpenSTEF](#database-connector-for-openstef)
- [License](license)
- [Contributing](#contributing)
- [Contact](#contact)

# External information sources

- [Documentation website](https://openstef.github.io/openstef/index.html);
- [Python package](https://pypi.org/project/openstef/);
- [Linux Foundation project page](https://www.lfenergy.org/projects/openstef/);
- [Documentation on dashboard](https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html);
- [Video about OpenSTEF](https://www.lfenergy.org/forecasting-to-create-a-more-resilient-optimized-grid/);

# Installation

## Install the openstef package

```shell
pip install openstef
```

### Remark regarding installation within a **conda environment on Windows**

A version of the pywin32 package will be installed as a secondary dependency along with the installation of the openstef package. Since conda relies on an old version of pywin32, the new installation can break conda's functionality. The following command can solve this issue:

```shell
pip install pywin32==300
```

For more information on this issue see the [readme of pywin32](https://github.com/mhammond/pywin32#installing-via-pip) or [this Github issue](https://github.com/mhammond/pywin32/issues/1865#issue-1212752696).

## Remark regarding installation on Apple Silicon

If you want to install the `openstef` package on Apple Silicon (Mac with M1-chip or newer), you can encounter issues with the dependencies, such as `xgboost`. Solution:

1. Run `brew install libomp` (if you haven’t installed Homebrew: [follow instructions here](https://brew.sh/))
2. If your interpreter can not find the `libomp` installation in `/usr/local/bin`, it is probably in `/opt/brew/Cellar`. Run:

```sh
mkdir -p /usr/local/opt/libomp/
ln -s /opt/brew/Cellar/libomp/{your_version}/lib /usr/local/opt/libomp/lib
```

3. Uninstall `xgboost` with `pip` (`pip uninstall xgboost`) and install with `conda-forge` (`conda install -c conda-forge xgboost`)
4. If you encounter similar issues with `lightgbm`: uninstall `lightgbm` with `pip` (`pip uninstall lightgbm`) and install later version with `conda-forge` (`conda install -c conda-forge 'lightgbm>=4.2.0'`)

### Remark regarding installation with minimal XGBoost dependency

It is possible to install openSTEF with a minimal XGBoost (CPU-only) package. This only works on x86_64 (amd64) Linux and Windows platforms. Advantage is that significantly smaller dependencies are installed. In that case run:

```shell
pip install openstef[cpu]
```

# Usage

## Example notebooks

To help you get started, a set of fundamental example notebooks has been created. You can access these offline examples [here](https://github.com/OpenSTEF/openstef-offline-example).

## Reference Implementation

A complete implementation including databases, user interface, example data, etc. is available at: https://github.com/OpenSTEF/openstef-reference

![screenshot](https://user-images.githubusercontent.com/60883372/146760483-29af3ac7-62af-4f13-98c7-982a79c517d1.jpg)
Screenshot of the operational dashboard showing the key functionality of OpenSTEF.
Dashboard documentation can be found [here](https://raw.githack.com/OpenSTEF/.github/main/profile/html/openstef_dashboard_doc.html).

To run a task use:

```shell
python -m openstef task <task_name>
```

## Database connector for openstef

This repository provides an interface to OpenSTEF (reference) databases. The repository can be found [here](https://github.com/OpenSTEF/openstef-dbc).

# License

This project is licensed under the Mozilla Public License, version 2.0 - see LICENSE for details.

## Licenses third-party libraries

This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable. The concerning license files can be found in the LICENSES directory.

# Contributing

Please read [CODE_OF_CONDUCT.md](https://github.com/OpenSTEF/.github/blob/main/CODE_OF_CONDUCT.md), [CONTRIBUTING.md](https://github.com/OpenSTEF/.github/blob/main/CONTRIBUTING.md) and [PROJECT_GOVERNANCE.md](https://github.com/OpenSTEF/.github/blob/main/PROJECT_GOVERNANCE.md) for details on the process for submitting pull requests to us.

# Contact

Please read [SUPPORT.md](https://github.com/OpenSTEF/.github/blob/main/SUPPORT.md) for how to connect and get into contact with the OpenSTEF project
