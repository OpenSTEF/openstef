<!--
SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

<!-- Github Actions badges -->
[![Python Build](https://github.com/alliander-opensource/short-term-forecasting/actions/workflows/python-build.yaml/badge.svg)](https://github.com/alliander-opensource/short-term-forecasting/actions/workflows/python-build.yaml)
[![REUSE Compliance Check](https://github.com/alliander-opensource/short-term-forecasting/actions/workflows/reuse-compliance.yaml/badge.svg)](https://github.com/alliander-opensource/short-term-forecasting/actions/workflows/reuse-compliance.yaml)
<!-- SonarCloud badges -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=alert_status)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=bugs)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=code_smells)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=coverage)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=duplicated_lines_density)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=security_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=sqale_index)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_short-term-forecasting&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=alliander-opensource_short-term-forecasting)

# Short Term Forcasting

Short term forcasting builds the `openstf` Python package which is used to make short term forecasts for the energy sector. This repository contains all components for the machine learning pipeline required to make a forecast. In order to use the package you need to provide your own data storage and retrieval interface. `openstf` is available at: https://pypi.org/project/openstf/

# Installation

## Install the openstf package

```shell
pip install openstf
```

## Install own implementation of ktpbase

The code in this repository expects a data storage and retrieval interface to be available through the `ktpbase` package. For more information about the implemented interface see TODO.

# Usage

To run a task use:

```shell
python -m openstf task <task_name>
```

## License
This project is licensed under the Mozilla Public License, version 2.0 - see LICENSE for details.

## Licenses third-party libraries
This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable. The concerning license files can be found in the LICENSES directory.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Contact

N.V. Alliander - Team Korte Termijn Prognoses <korte.termijn.prognoses@alliander.com>
