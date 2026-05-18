.. comment:
    SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
    SPDX-License-Identifier: MPL-2.0

.. _examples:

Examples
========

End-to-end tutorials demonstrating OpenSTEF workflows. Each tutorial is a runnable
Jupyter notebook rendered with executed outputs.

Tutorials
---------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Forecasting Quickstart <tutorials/forecasting_quickstart>
   Backtesting Quickstart <tutorials/backtesting_quickstart>

.. toctree::
   :maxdepth: 1
   :caption: Model Training

   Building a Custom Pipeline <tutorials/custom_pipeline>
   Ensemble Forecasting <tutorials/ensemble_forecasting>
   Hyperparameter Tuning <tutorials/hyperparameter_tuning_with_optuna>
   Forecasting with Presets <tutorials/forecasting_with_workflow_presets>

.. toctree::
   :maxdepth: 1
   :caption: Evaluation & Analysis

   Model Explainability <tutorials/model_explainability>
   Quantile Calibration <tutorials/quantile_calibration>


Benchmarks
----------

Compare models on real energy data. These notebooks are **not executed** during
docs build — run them locally.

.. toctree::
   :maxdepth: 2
   :caption: Benchmarking

   Benchmarking Guide <benchmarks/README>
   Liander 2024 <benchmarks/liander2024/README>
   Build Your Own <benchmarks/custom/README>
