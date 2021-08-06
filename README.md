<!--
SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

<!-- Github Actions badges -->
[![Python Build](https://github.com/alliander-opensource/openstf/actions/workflows/python-build.yaml/badge.svg)](https://github.com/alliander-opensource/openstf/actions/workflows/python-build.yaml)
[![REUSE Compliance Check](https://github.com/alliander-opensource/openstf/actions/workflows/reuse-compliance.yaml/badge.svg)](https://github.com/alliander-opensource/openstf/actions/workflows/reuse-compliance.yaml)
<!-- SonarCloud badges -->
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=alert_status)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=bugs)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=code_smells)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=coverage)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Duplicated Lines (%)](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=duplicated_lines_density)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Reliability Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=reliability_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Security Rating](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=security_rating)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Technical Debt](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=sqale_index)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)
[![Vulnerabilities](https://sonarcloud.io/api/project_badges/measure?project=alliander-opensource_openstf&metric=vulnerabilities)](https://sonarcloud.io/dashboard?id=alliander-opensource_openstf)

# Openstf

Openstf is a Python package which is used to make short term forecasts for the energy sector. This repository contains all components for the machine learning pipeline required to make a forecast. In order to use the package you need to provide your own data storage and retrieval interface. `openstf` is available at: https://pypi.org/project/openstf/

# Installation

## Install the openstf package

```shell
pip install openstf
```

# Usage

To run a task use:

```shell
python -m openstf task <task_name>
```

## Reference Implementation
A complete implementation including databases, user interface, example data, etc. is available at: https://github.com/alliander-opensource/openstf-reference
![image](https://user-images.githubusercontent.com/18208480/127109029-77e09c97-8d06-4158-8789-4c1d5ecede61.png)

## License
This project is licensed under the Mozilla Public License, version 2.0 - see LICENSE for details.

## Licenses third-party libraries
This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable. The concerning license files can be found in the LICENSES directory.

## Contributing

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) and [CONTRIBUTING.md](CONTRIBUTING.md) for details on the process for submitting pull requests to us. 

## Contact

N.V. Alliander - Team Korte Termijn Prognoses (Short Term Forecasting) <korte.termijn.prognoses@alliander.com>
