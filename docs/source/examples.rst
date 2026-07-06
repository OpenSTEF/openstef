.. comment:
    SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
    SPDX-License-Identifier: MPL-2.0

.. _examples:

========
Examples
========

End-to-end tutorials and benchmarks demonstrating OpenSTEF workflows. Each
tutorial is a runnable Jupyter notebook rendered with executed outputs.

Tutorials
---------

Getting Started
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - Description
   * - :doc:`Forecasting Quickstart <tutorials/forecasting_quickstart>`
     - Train your first XGBoost model and produce a 47-hour ahead forecast using the Liander dataset.
   * - :doc:`Backtesting Quickstart <tutorials/backtesting_quickstart>`
     - Evaluate forecast accuracy on historical data using BEAM's rolling-window backtesting pipeline.
   * - :doc:`Understanding Datasets <tutorials/datasets>`
     - Learn how versioned time series data works and why it matters for honest forecasting.

Training & Forecasting
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - Description
   * - :doc:`Custom Pipeline <tutorials/custom_pipeline>`
     - Build a forecasting workflow from scratch with custom transforms, feature selection, and callbacks.
   * - :doc:`Feature Engineering <tutorials/feature_engineering>`
     - Explore transforms by domain, apply them individually, and compose preprocessing pipelines.
   * - :doc:`Ensemble Forecasting <tutorials/ensemble_forecasting>`
     - Combine multiple models into an ensemble for improved accuracy and robustness.
   * - :doc:`Hyperparameter Tuning <tutorials/hyperparameter_tuning_with_optuna>`
     - Optimize model hyperparameters using Optuna integration with cross-validated backtesting.
   * - :doc:`Foundation Model Forecasting Quickstart <tutorials/foundation_model_forecasting_quickstart>`
     - Produce a zero-shot probabilistic forecast with the pretrained Chronos-2 model via OpenSTEF's ONNX backend, including batched multi-origin inference.
   * - :doc:`Foundation Model Forecasting <tutorials/foundation_model_forecasting>`
     - Choose a checkpoint, run one from disk, pick the execution provider for your hardware, and backtest with openstef-beam.

Evaluation & Analysis
^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Tutorial
     - Description
   * - :doc:`Model Explainability <tutorials/model_explainability>`
     - Inspect feature importances, SHAP values, and contribution plots for trained models.
   * - :doc:`Quantile Calibration <tutorials/quantile_calibration>`
     - Calibrate prediction intervals using isotonic regression for reliable uncertainty estimates.


Benchmarks
----------

Compare models on real energy data. These notebooks are **not executed** during
the docs build — run them locally to reproduce results.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Benchmark
     - Description
   * - :doc:`Benchmarking Guide <benchmarks/README>`
     - Overview of the benchmarking framework and how to interpret results.
   * - :doc:`Liander 2024 <benchmarks/liander2024/README>`
     - Full benchmark on Liander's MV feeder dataset comparing XGBoost, LightGBM, and linear baselines.
   * - :doc:`Build Your Own <benchmarks/custom/README>`
     - Template for creating custom benchmarks on your own data.


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   Forecasting Quickstart <tutorials/forecasting_quickstart>
   Backtesting Quickstart <tutorials/backtesting_quickstart>
   Understanding Datasets <tutorials/datasets>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Training & Forecasting

   Custom Pipeline <tutorials/custom_pipeline>
   Feature Engineering <tutorials/feature_engineering>
   Ensemble Forecasting <tutorials/ensemble_forecasting>
   Hyperparameter Tuning <tutorials/hyperparameter_tuning_with_optuna>
   Foundation Model Forecasting Quickstart <tutorials/foundation_model_forecasting_quickstart>
   Foundation Model Forecasting <tutorials/foundation_model_forecasting>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Evaluation & Analysis

   Model Explainability <tutorials/model_explainability>
   Quantile Calibration <tutorials/quantile_calibration>

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Benchmarks

   Benchmarking Guide <benchmarks/README>
   Liander 2024 <benchmarks/liander2024/README>
   Build Your Own <benchmarks/custom/README>
