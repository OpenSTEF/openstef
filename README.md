<!--
SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Short Term Forcasting

Short term forcasting is a Python package which is used to make short term forecasts for the energy sector. This repository contains all components for the machine learning pipeline required to make a forecast. In order to use the package you need to provide your own data storage and retrieval interface.

# Installation

## Clone this repository:

```shell
git clone git@github.com:alliander-opensource/short-term-forecasting.git
```

## Install dependencies

```shell
pip install -r requirements.txt
```

## Install the stf package

```shell
pip install -e stf
```

## Install own implementation of ktpbase

The code in this repository expects a data storage and retrieval interface to be available through the `ktpbase` package. For more information about the implemented interface see TODO.

# Usage

To run a task use:

```shell
python -m stf task <task_name>
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Contact

N.V. Alliander - Team Korte Termijn Prognoses <korte.termijn.prognoses@alliander.com>
