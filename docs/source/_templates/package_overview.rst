.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

{% set parts = fullname.split('.') %}
{% set short_name = parts[-1] if parts|length >= 3 else (parts[1:] | join('.') if parts|length > 1 else fullname) %}
{{ short_name }}
{{ "=" * short_name|length }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block package_overview %}
{# Dynamically discover child modules/packages via pkgutil (no hardcoding needed) #}
{% set discovered = discover_submodules(fullname) %}
{# Remove any modules already provided by autosummary to avoid duplicates #}
{% set known = modules | map('replace', fullname ~ '.', '') | list %}
{% set extra_submodules = discovered | reject('in', known) | list %}
{% if modules or extra_submodules or functions or classes or members %}

{% if modules or extra_submodules %}
Submodules
----------

{% if extra_submodules %}
.. autosummary::
   :toctree: .
   :template: module_overview.rst
{% for item in extra_submodules %}
   {{ fullname }}.{{ item }}
{%- endfor %}

{% endif %}
{% if modules %}
.. autosummary::
   :toctree: .
   :template: module_overview.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}

{% endif %}
{% endif %}
{% if functions %}
Functions
---------

.. autosummary::
   :toctree: .
   :template: custom_function.rst
{% for item in functions %}
   {{ item }}
{%- endfor %}

{% endif %}
{% if classes %}
Classes
-------

.. autosummary::
   :toctree: .
   :template: custom_class.rst
{% for item in classes %}
   {{ item }}
{%- endfor %}

{% endif %}

{# Handle the case where classes are in members but not in classes variable #}
{% if members and not classes and not functions %}
{# Check if members are likely to be classes by checking if it's an exceptions module #}
{% if 'exceptions' in fullname %}
Classes
-------

.. autosummary::
   :toctree: .
   :template: custom_class.rst
{% for item in members %}
   {{ item }}
{%- endfor %}

{% endif %}
{% endif %}

{% if attributes %}
Module Attributes
-----------------

.. autosummary::
{% for item in attributes %}
   {{ item }}
{%- endfor %}

{% endif %}
{% endif %}
{% endblock %}
