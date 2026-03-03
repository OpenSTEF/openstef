.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _citing_openstef:

Citing OpenSTEF
===============

{{ citation_cff.message }}
You can cite the project using the DOI `{{ citation_cff.doi }} <https://doi.org/{{ citation_cff.doi }}>`_
or reference the GitHub repository at {{ citation_cff.url }}.

BibTeX Format
-------------

.. code-block:: bibtex

{{ citation_bibtex | indent(3, true) }}


DOI
---

The following DOI represents the OpenSTEF project:

.. image:: https://zenodo.org/badge/DOI/{{ citation_cff.doi }}.svg
   :target: https://doi.org/{{ citation_cff.doi }}


Citation File Format (CFF)
---------------------------

.. literalinclude:: ../../../CITATION.cff
   :language: yaml


.. container:: sphx-glr-download

   :download:`Download CFF citation file: CITATION.cff <../../../CITATION.cff>`
