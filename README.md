<!--
SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

<!-- Github Actions badges -->
[![Python Build](https://github.com/openstef/openstef/actions/workflows/python-build.yaml/badge.svg)](https://github.com/openstef/openstef/actions/workflows/python-build.yaml)
[![REUSE Compliance Check](https://github.com/openstef/openstef/actions/workflows/reuse-compliance.yaml/badge.svg)](https://github.com/openstef/openstef/actions/workflows/reuse-compliance.yaml)
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
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5585/badge)](https://bestpractices.coreinfrastructure.org/projects/5585)

# OpenSTEF

OpenSTEF is a Python package which is used to make short term forecasts for the energy sector. This repository contains all components for the machine learning pipeline required to make a forecast. In order to use the package you need to provide your own data storage and retrieval interface. `openstef` is available at: https://pypi.org/project/openstef/

# Installation

## Install the openstef package

```shell
pip install openstef
```

_**Optional**_: if you would like to use the proloaf model with OpenSTEF install the proloaf dependencies by running:
```shell
pip install openstef[proloaf]
```

# Usage

To run a task use:

```shell
python -m openstef task <task_name>
```

## Reference Implementation
A complete implementation including databases, user interface, example data, etc. is available at: https://github.com/OpenSTEF/openstef-reference

![screenshot](https://user-images.githubusercontent.com/60883372/146760483-29af3ac7-62af-4f13-98c7-982a79c517d1.jpg)
Screenshot of the operational dashboard showing the key functionality of OpenSTEF. 
Dashboard documentation can be found [here](https://github.com/OpenSTEF/.github/blob/main/profile/README.md).

## License
This project is licensed under the Mozilla Public License, version 2.0 - see LICENSE for details.

## Licenses third-party libraries
This project includes third-party libraries, which are licensed under their own respective Open-Source licenses. SPDX-License-Identifier headers are used to show which license is applicable. The concerning license files can be found in the LICENSES directory.

## Contributing
Please read [CODE_OF_CONDUCT.md](https://github.com/OpenSTEF/.github/blob/main/CODE_OF_CONDUCT.md), [CONTRIBUTING.md](https://github.com/OpenSTEF/.github/blob/main/CONTRIBUTING.md) and [PROJECT_GOVERNANACE.md](https://github.com/OpenSTEF/.github/blob/main/PROJECT_GOVERNANCE.md) for details on the process for submitting pull requests to us.

## Contact
Please read [SUPPORT.md](https://github.com/OpenSTEF/.github/blob/main/SUPPORT.md) for how to connect and get into contact with the OpenSTEF project
