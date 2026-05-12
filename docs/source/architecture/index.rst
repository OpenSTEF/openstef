Architecture
============

OpenSTEF is organized into four packages with clear responsibilities: ``openstef_core`` provides foundational data structures and utilities, ``openstef_models`` implements forecasting models and transforms, ``openstef_beam`` handles backtesting, evaluation, and metrics pipelines, and ``openstef_meta`` combines models into ensemble forecasts.

.. mermaid:: /diagrams/architecture/index_diagram_1.mmd

**Core Package** (:doc:`core`)
   Deep dive into ``openstef_core``: the validated dataset hierarchy, shared utilities, and base abstractions that all other packages build upon.

**Models Package** (:doc:`models`)
   Deep dive into ``openstef_models``: domain-organized transforms, forecasting model implementations, and the model training lifecycle.

**BEAM Package** (:doc:`beam`)
   Deep dive into ``openstef_beam``: backtesting pipelines, evaluation metrics, analysis tools, and reporting capabilities.

**Meta Package** (:doc:`meta`)
   Deep dive into ``openstef_meta``: ensemble forecasting, forecast combination strategies, and multi-model orchestration.

.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   core
   models
   beam
   meta
