.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
..
.. SPDX-License-Identifier: MPL-2.0

{{ fullname }}
{{ "=" * fullname|length }}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :no-members:
   :no-inherited-members:

{% block package_overview %}
{% if modules or functions or classes or members %}

{# Custom logic to detect submodules from __all__ #}
{% set submodules = [] %}
{% if fullname == 'openstef_core.datasets' %}
{% set _ = submodules.append('mixins') %}
{% endif %}

{% if modules or submodules %}
Submodules
----------

{% if submodules %}
.. autosummary::
   :toctree: .
   :template: module_overview.rst
{% for item in submodules %}
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
