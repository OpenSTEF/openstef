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

{% block overview_table %}
{% if modules or functions or classes %}

{% if modules %}
Submodules
----------

.. autosummary::
   :toctree: .
   :template: module_overview.rst
{% for item in modules %}
   {{ item }}
{%- endfor %}

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
{% endif %}
{% endblock %}
