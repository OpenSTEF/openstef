<!--
SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>

SPDX-License-Identifier: MPL-2.0
-->

# Documentation instructions

Documentation is generated using Sphinx:
https://www.sphinx-doc.org/en/master/index.html

For every *pull_request*:
- check if documentation can be generated

For every *release*:
1. A github action triggers the building of documentation
    1. sphinx-apidoc generates automatically .rst files based on source code
    2. sphinx-build generates automatically .html files from the rst files
2. The github action posts the results to `gh_pages` branch
3. `gh_pages` is automatically turned into a website by github

Important files:
- `conf.py`: defining settings
- `Makefile` and `make.bat`: not exactly sure, we use defaults
- `index.rst`: Define the index of the final documentation

Running locally, documentation html files are generated but not added to gh-pages:
```
pip install -r requirements.txt
pip install -r docs/doc-requirements.txt
sphinx-apidoc -o docs openstef
sphinx-build docs output
```

Run docstring formatting and checks locally:
```
pip install -r requirements.txt
pip install -r docs/doc-requirements.txt
pydocstyle .
docformatter openstef --recursive --wrap-summaries 120 --in-place
```
