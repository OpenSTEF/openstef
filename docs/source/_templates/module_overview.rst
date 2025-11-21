.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

{{ fullname }}
{{ "=" * fullname|length }}

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
