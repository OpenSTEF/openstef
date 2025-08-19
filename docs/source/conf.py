# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import re
from typing import Any
from sphinx_pyproject import SphinxConfig
from sphinx.application import Sphinx
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

project = "OpenSTEF"
copyright = "2017-2025 Contributors to the OpenSTEF project"
author = "Contributors to the OpenSTEF project"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

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
autosummary_generate_overwrite = True
autosummary_imported_members = False

# Autodoc settings for better docstring rendering
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": False,  # Only document explicitly defined members
    "exclude-members": "__weakref__",
    "imported-members": False,
}

# Better type hints handling
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"
autodoc_preserve_defaults = True
autodoc_typehints_format = "short"
autodoc_typehints_signature = True

autodoc_inherit_docstrings = True
autodoc_member_order = "bysource"

# Respect __all__ definitions
automodule_skip_lines = 0

# Configure the typehints extension
typehints_fully_qualified = False  # Use short names like 'str' instead of 'builtins.str'
typehints_document_rtype = True  # Document return types
typehints_use_signature = True  # Show in signatures
typehints_use_signature_return = True  # Include return type in signature
always_document_param_types = True  # Always show param types even without docstring

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Disable NumPy style
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configure MyST for docstrings
myst_enable_extensions = [
    "deflist",
    "tasklist",
    "colon_fence",
]

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


# Citations


def cff_to_bibtex(cff_data: dict[str, Any]) -> str:
    """Convert CFF data to BibTeX format"""
    title: str = cff_data.get("title", "No Title")
    doi: str = cff_data.get("doi", "")
    url: str = cff_data.get("url", "")
    version: str = cff_data.get("version", "")

    year = ""
    if "date-released" in cff_data:
        year = str(cff_data["date-released"])[:4]

    authors = cff_data.get("authors", [])
    bib_authors: list[str] = []

    for author in authors:
        if "name" in author and "Contributors" in author["name"]:
            continue

        parts: list[str] = []
        if "family-names" in author:
            family = author["family-names"]
            if "name-particle" in author:
                family = f"{author['name-particle']} {family}"
            parts.append(family)

        if "given-names" in author:
            parts.append(author["given-names"])

        if parts:
            full_name = f"{parts[0]}, {parts[1]}" if len(parts) == 2 else " ".join(parts)
            bib_authors.append(full_name)

    authors_str = " and ".join(bib_authors) if bib_authors else "Unknown"
    bib_key: str = re.sub(r"[^a-zA-Z0-9]", "", title.lower().replace("/", ""))

    bibtex_lines = [f"@software{{{bib_key},", f"  title = {{{title}}},", f"  author = {{{authors_str}}},"]

    if year:
        bibtex_lines.append(f"  year = {{{year}}},")
    if version:
        bibtex_lines.append(f"  version = {{{version}}},")
    if doi:
        bibtex_lines.append(f"  doi = {{{doi}}},")
    if url:
        bibtex_lines.append(f"  url = {{{url}}},")

    bibtex_lines.append("}")
    return "\n".join(bibtex_lines)


# Load citation data
citation_cff = None
citation_bibtex = None
# Try to load CITATION.cff from project root
cff_path = ROOT_DIR / "CITATION.cff"
if cff_path.exists():
    try:
        with cff_path.open(mode="r", encoding="utf-8") as f:
            citation_cff = yaml.safe_load(f)
        citation_bibtex = cff_to_bibtex(citation_cff)
        print(f"Loaded citation data from {cff_path}")
    except Exception as e:
        print(f"Warning: Could not load CITATION.cff: {e}")


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "logos/logo_openstef_small.png"
html_favicon = "logos/favicon.ico"
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
    "show_prev_next": True,
    "search_bar_text": "Search the docs ...",
    "navigation_with_keys": False,
    "collapse_navigation": False,
    "navigation_depth": 1,  # Only show first level in navbar
    "show_nav_level": 1,  # Limit navbar to level 1 only
    "navbar_center": ["navbar-nav"],  # Use only primary navigation
    "show_toc_level": 2,  # Keep detailed TOC in sidebar
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
        "image_light": "logos/openstef-horizontal-color.svg",
        "image_dark": "logos/openstef-horizontal-white.svg",
    },
    "surface_warnings": True,
}

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "openstef"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images", "js", "_static"]

html_css_files = [
    "css/custom.css",  # Custom CSS for styling
]

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

# -- Sphinx setup ------------------------------------------------------------


def rstjinja(app: Sphinx, _docname: str, source: list[str]) -> None:
    """Render RST files as Jinja templates for variable substitution."""
    # Only process HTML builds
    if app.builder.format != "html":  # type: ignore[attr-defined]
        return

    src: str = source[0]
    rendered: str = app.builder.templates.render_string(  # type: ignore[attr-defined]
        src,
        app.config.html_context,  # type: ignore[attr-defined]
    )
    source[0] = rendered


def setup(app: Sphinx) -> None:
    """Sphinx setup function to make citation data available in templates."""
    if citation_cff:
        context_update: dict[str, object] = {
            "citation_cff": citation_cff,
            "citation_bibtex": citation_bibtex,
        }
        app.config.html_context.update(context_update)  # type: ignore[attr-defined]
        app.connect("source-read", rstjinja)
