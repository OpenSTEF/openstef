# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib
import pkgutil
import sys
from pathlib import Path
import re
from typing import Any
import warnings
from docutils import nodes as docutils_nodes
from sphinx_pyproject import SphinxConfig
from sphinx.application import Sphinx
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Make docs/ importable so we can use sync_sources
sys.path.insert(0, str(ROOT_DIR / "docs"))
from sync_sources import sync as _sync_sources  # noqa: E402

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
    "myst_nb",
    "sphinx_design",
    "sphinx_copybutton",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.mermaid",
]

# Mermaid configuration
mermaid_version = "10.6.1"
mermaid_init_js = "mermaid.initialize({startOnLoad:true});"
mermaid_d3_zoom = False  # Disable zoom feature that adds extra wrapper

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["conf.py", "**/*.ipynb"]

# Specify how to identify the prompt when copying code snippets
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True
copybutton_exclude = "style"


# Autosummary settings (SciPy style)
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = False

# Suppress benign import_cycle warnings from recursive autosummary (modules are
# listed in both the parent's :recursive: directive and the template's toctree)
suppress_warnings = ["autosummary.import_cycle"]


def _discover_submodules(fullname: str) -> list[str]:
    """Discover child modules/packages of *fullname* via pkgutil.

    Returns an empty list when *fullname* is not a package (has no ``__path__``)
    or when the import fails (e.g. missing optional dependency).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod = importlib.import_module(fullname)
        except Exception:  # noqa: BLE001
            return []
    pkg_path = getattr(mod, "__path__", None)
    if pkg_path is None:
        return []
    return sorted(name for _, name, _ispkg in pkgutil.iter_modules(pkg_path))


autosummary_context = {
    "discover_submodules": _discover_submodules,
}

# Don't generate separate pages for class members
autosummary_mock_imports = []

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
autodoc_typehints_signature = False  # Don't show types in signatures to avoid duplication

autodoc_inherit_docstrings = True
autodoc_member_order = "groupwise"

# Respect __all__ definitions
automodule_skip_lines = 0

# Configure the typehints extension
typehints_fully_qualified = False  # Use short names like 'str' instead of 'builtins.str'
typehints_document_rtype = True  # Document return types
typehints_use_signature = True  # Show in signatures
typehints_use_signature_return = True  # Include return type in signature
always_document_param_types = True  # Always show param types even without docstring

# Autosummary configuration for better API reference (scikit-learn style)
autosummary_filename_map = {}
autosummary_ignore_module_all = False

# Napoleon settings for Google-style docstrings (similar to scikit-learn)
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Disable NumPy style
napoleon_include_init_with_doc = False  # Don't include __init__ params in class docstring to avoid duplication
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True  # Enable instance variable documentation
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True
napoleon_custom_sections = ["Invariants"]

# Configure MyST for docstrings
myst_enable_extensions = [
    "deflist",
    "dollarmath",
    "tasklist",
    "colon_fence",
]

# -- Notebook execution (myst-nb) -------------------------------------------
nb_custom_formats = {".py": ["jupytext.reads", {"fmt": "py:percent"}]}
nb_execution_mode = "cache"
nb_execution_cache_path = str(ROOT_DIR / "docs" / "build" / ".jupyter_cache")
nb_execution_timeout = 120
nb_execution_raise_on_error = True
nb_execution_show_tb = True
nb_execution_excludepatterns = [
    "benchmarks/*",  # Benchmarks are too expensive to execute during docs build
    "benchmarks/*/*",
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
    "navigation_depth": 4,
    "show_nav_level": 2,
    "navbar_center": ["navbar-nav"],  # Use only primary navigation
    "show_toc_level": 2,
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

# Disable default "View page source" link (in favor of PyData theme's "Edit this page" link)
html_show_sourcelink = False

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
    # Edit page source
    "github_user": "OpenSTEF",
    "github_repo": "openstef",
    "github_version": "release/v4.0.0",
    "doc_path": "docs/source",
}

# -- Sphinx setup ------------------------------------------------------------


def rstjinja(app: Sphinx, docname: str, source: list[str]) -> None:
    """Render RST files as Jinja templates for variable substitution."""
    # Only process HTML builds
    if app.builder.format != "html":  # type: ignore[attr-defined]
        return

    # Only process .rst sources — skip notebooks/MyST which contain {} literals
    rst_path = Path(app.srcdir) / f"{docname}.rst"
    if not rst_path.is_file() or not source[0].strip():
        return

    src: str = source[0]
    rendered: str = app.builder.templates.render_string(  # type: ignore[attr-defined]
        src,
        app.config.html_context,  # type: ignore[attr-defined]
    )
    source[0] = rendered


def _inject_pydantic_field_descriptions(
    app: Sphinx,  # noqa: ARG001
    domain: str,
    objtype: str,
    contentnode: docutils_nodes.Element,
) -> None:
    """Inject Pydantic ``Field(description=...)`` text into attribute nodes.

    This runs via ``object-description-transform`` *after* autodoc has rendered
    each attribute directive.  For Pydantic model fields whose content area is
    empty, we look up the ``description`` from ``model_fields`` and insert a
    paragraph node so the description appears next to the type annotation.
    """
    if domain != "py" or objtype != "attribute":
        return
    # Only inject into empty content areas
    if contentnode.children:
        return
    # The signature node is the first child of the parent desc node
    sig = contentnode.parent[0] if contentnode.parent else None
    if sig is None:
        return
    ids = sig.get("ids", [])
    if not ids:
        return
    # ID looks like "pkg.mod.ClassName.field_name"
    full_id = ids[0]
    parts = full_id.rsplit(".", 2)
    if len(parts) < 3:
        return
    field_name = parts[-1]
    parent_qualname = ".".join(parts[:-1])
    parent_module, _, parent_attr = parent_qualname.rpartition(".")
    if not parent_module:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mod = importlib.import_module(parent_module)
        except Exception:  # noqa: BLE001
            return
    parent_cls = getattr(mod, parent_attr, None)
    model_fields = getattr(parent_cls, "model_fields", None)
    if model_fields is None or field_name not in model_fields:
        return
    desc = model_fields[field_name].description
    if not desc:
        return
    para = docutils_nodes.paragraph("", desc)
    contentnode.append(para)


def setup(app: Sphinx) -> None:
    """Sphinx setup function to make citation data available in templates."""
    # Sync tutorial/benchmark sources into docs/source on every build
    _sync_sources()

    if citation_cff:
        context_update: dict[str, object] = {
            "citation_cff": citation_cff,
            "citation_bibtex": citation_bibtex,
        }
        app.config.html_context.update(context_update)  # type: ignore[attr-defined]
        app.connect("source-read", rstjinja)
    app.connect("object-description-transform", _inject_pydantic_field_descriptions)
