<!--
SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->
# Release strategy
Releases are made very frequently, at least every two weeks.
Every new Pull Request merged to main triggers a new automatic github release with bumped patch version (e.g. 0.0.1a -> 0.0.2), a new pypi release and new published documentation. 

If needed, a manual release can be done:
- Major (1.0.0) or minor (0.1.0) version need to be bumped besides patch version: do this yourself in the feature branch in the setup.py.
- Pre-release needs to be made: do this yourself in the feature branch with new pre-release version in setup.py and manual pre-release in github GUI.