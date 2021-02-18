# Documentation instructions
Documentation is generated using Sphinx:
https://www.sphinx-doc.org/en/master/index.html

Automation works in these steps:
1. a github action triggers the building of documentation
    1. sphinx-apidoc generates automatically .rst files based on source code
    2. sphinx-build generates automatically .html files from the rst files 
2. the github action posts the results to `gh_pages` branch 
3. `gh_pages` is automatically turned into a website by github

Important files:
- `conf.py`: defining settings
- `Makefile` and `make.bat`: not exactly sure, we use defaults
- `index.rst`: Define the index of the final documentation
- `requirements.txt`: requirements needed for building documentation. This is more than the package itself  
- `ktpbase`: empty ktpbase implementation so other python modules do not give an exception when imported

Running locally:
`sphinx-apidoc -o docs stf`
`sphinx-build docs output`