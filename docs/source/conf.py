# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from typing import Any
from sphinx_pyproject import SphinxConfig

project = 'OpenSTEF'
copyright = '2017-2025 Contributors to the OpenSTEF project'
author = 'Contributors to the OpenSTEF project'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
    'sphinx_copybutton',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_exclude = "style"


# Autosummary settings (SciPy style)
autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Sphinx version switcher
config = SphinxConfig("../../pyproject.toml", globalns=globals())
version = config.version
release = config.version

if "dev" in version or "+" in version or version == "0.0.0":
    version_match = "dev"
    json_url = "_static/versions.json"
else:
    version_match = ".".join(version.split(".")[:2])
    json_url = "https://openstef.github.io/docs/_static/versions.json"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_logo = "logos/logo_openstef_small.png"
html_favicon = "logos/favicon.ico"
html_static_path = ['_static']
html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    "use_edit_page_button": True,
    "external_links": [],
    "icon_links_label": "Icon Links",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/OpenSTEF/openstef",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "LFE Homepage",
            "url": "https://www.lfenergy.org/projects/openstef/",
            "icon": "fa-solid fa-bolt",
            "type": "fontawesome",
        },
    ],
    "show_prev_next": False,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 2,
    "show_nav_level": 1,
    "show_toc_level": 1,
    "navbar_align": "left",
    "header_links_before_dropdown": 5,
    "header_dropdown_text": "More",
    "switcher": {
        "json_url": json_url,
        "version_match": version_match,
    },
    "check_switcher": True,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
    "logo": {
        "alt_text": "OpenSTEF homepage",
        "image_light": "logo_openstef_small.png",
        "image_dark": "logo_openstef_small.png",
    },
    "surface_warnings": True,
}

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "openstef"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images", "css", "js"]

# Custom sidebar templates, maps document names to template names.
# Workaround for removing the left sidebar on pages without TOC
# A better solution would be to follow the merge of:
# https://github.com/pydata/pydata-sphinx-theme/pull/1682
html_sidebars: Any = {
    "install": [],
    "getting_started": [],
    "glossary": [],
    "faq": [],
    "support": [],
    "related_projects": [],
    "roadmap": [],
    "governance": [],
    "about": [],
}

# Repository configuration for edit buttons
html_context = {
    "github_user": "OpenSTEF",
    "github_repo": "openstef",
    "github_version": "main",
    "doc_path": "docs/source",
}