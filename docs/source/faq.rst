FAQ
===

This page answers common questions from new users about OpenSTEF — what it does,
how to set it up, and how to get the most out of it. Click any question to expand
the answer.

General
-------

.. dropdown:: What is OpenSTEF?
   :icon: question

   OpenSTEF (Open Short-Term Energy Forecasting) is an open-source Python library
   that provides a complete machine learning framework for short-term energy load
   forecasting. It covers the full pipeline: data preprocessing, feature engineering,
   model training, probabilistic forecasting, evaluation, and post-processing.

   OpenSTEF is **not** a single model — it is a model-agnostic framework that lets
   you plug in different ML algorithms while handling all the domain-specific logic
   for energy forecasting.

.. dropdown:: What does "short-term" mean in this context?
   :icon: question

   Short-term means predicting energy load **hours to approximately 7 days** ahead.
   The practical limit is around 7 days because:

   - Weather forecasts beyond 7 days lack the 15-minute resolution needed for accurate predictions
   - Solar and wind peaks become unpredictable (cloudy vs sunny days)
   - Forecast quality degrades significantly beyond this horizon

.. dropdown:: What makes OpenSTEF different from a general ML library?
   :icon: question

   OpenSTEF includes domain knowledge specific to energy forecasting that you would
   otherwise need to build yourself:

   - **Energy-specific feature engineering** — e.g., converting solar radiation into PV generation estimates
   - **Probabilistic forecasts** — generates prediction intervals (uncertainty bandwidths), not just point predictions
   - **Complete pipelines** — handles the full workflow from raw data to deployable forecasts
   - **Built-in evaluation** — metrics and benchmarking tools designed for energy forecasting use cases

   A general ML library gives you models; OpenSTEF gives you a forecasting system.

.. dropdown:: What are the primary use cases?
   :icon: question

   OpenSTEF was originally developed at Alliander (a Dutch grid operator) for
   **congestion management** — forecasting load at grid points to identify when
   equipment limits will be exceeded. Other use cases include:

   - Transport capacity forecasting
   - EV charging capacity estimation
   - Grid loss prediction
   - Any scenario where you need accurate short-term load predictions

Installation & Setup
--------------------

.. dropdown:: What are the system requirements?
   :icon: question

   - **Python**: >=3.12, <4.0
   - **OS**: Linux, macOS, or Windows

   Install the complete framework:

   .. code-block:: bash

      pip install openstef

   This meta-package installs all sub-packages: ``openstef-beam``, ``openstef-core``,
   ``openstef-meta``, and ``openstef-models``.

.. dropdown:: Can I install only the parts I need?
   :icon: question

   Yes. OpenSTEF is modular — install individual packages for specific functionality:

   .. code-block:: bash

      # Core data processing and pipelines
      pip install openstef-core

      # Models (XGBoost, LightGBM, etc.)
      pip install openstef-models

      # Backtesting, Evaluation, Analysis and Metrics
      pip install openstef-beam

      # Meta-learning models
      pip install openstef-meta

   Each package also has optional extras. For example:

   .. code-block:: bash

      # Install models with LightGBM support
      pip install openstef-models[lgbm]

      # Install models with hyperparameter tuning
      pip install openstef-models[tuning]

      # Install beam with baseline models
      pip install openstef-beam[baselines]

.. dropdown:: What are the key dependencies?
   :icon: question

   The core dependencies include:

   - **numpy**, **pandas** — data manipulation
   - **pydantic** — configuration and validation
   - **pyarrow** — efficient data serialization
   - **joblib** — parallel processing

   Model-specific dependencies:

   - **xgboost** — XGBoost gradient boosting (via ``openstef-models[xgb-cpu]`` or ``openstef-models[xgb-gpu]``)
   - **lightgbm** — LightGBM gradient boosting (via ``openstef-models[lgbm]``)
   - **optuna** — hyperparameter tuning (via ``openstef-models[tuning]``)
   - **mlflow-skinny** — experiment tracking
   - **pvlib** — solar energy calculations

Models & Forecasting
--------------------

.. dropdown:: Which ML models are supported?
   :icon: question

   OpenSTEF currently supports:

   - **XGBoost** — gradient boosting trees, the default choice for most use cases
   - **LightGBM** — an alternative gradient boosting implementation

   Both models support probabilistic forecasting via quantile regression. The framework
   is model-agnostic, so additional model types (including foundation models) are being
   developed.

   See :doc:`user_guides/models` for details on model configuration.

.. dropdown:: What input data does OpenSTEF need?
   :icon: question

   At minimum, you need:

   - **Load measurements** — historical energy consumption time series
   - **Weather forecasts** — temperature, wind speed, solar radiation, humidity, pressure

   Optional but beneficial:

   - **Electricity market prices** (e.g., EPEX day-ahead prices)
   - **Load profiles** — typical daily/weekly consumption patterns

   Here's how to configure weather columns in a workflow:

   .. code-block:: python

      from openstef_models.presets import ForecastingWorkflowConfig

      config = ForecastingWorkflowConfig(
          model_id="my_forecast",
          model="xgboost",
          target_column="load",
          temperature_column="temperature_2m",
          relative_humidity_column="relative_humidity_2m",
          wind_speed_column="wind_speed_10m",
          radiation_column="shortwave_radiation",
          pressure_column="surface_pressure",
      )

.. dropdown:: What are probabilistic forecasts and why do they matter?
   :icon: question

   Instead of predicting a single value ("load will be 500 kW"), OpenSTEF produces
   **prediction intervals** — e.g., "load will be between 450 and 550 kW with 80%
   confidence."

   This is critical for grid operations because decisions (like curtailing customers)
   have real costs. Knowing the uncertainty helps operators make better risk-adjusted
   decisions.

   You configure quantiles when setting up a workflow:

   .. code-block:: python

      from openstef_core.types import LeadTime, Q

      config = ForecastingWorkflowConfig(
          # ...
          horizons=[LeadTime.from_string("PT36H")],
          quantiles=[Q(0.5), Q(0.1), Q(0.9)],  # Median + 80% prediction interval
      )

.. dropdown:: Can I tune hyperparameters automatically?
   :icon: question

   Yes. OpenSTEF integrates with Optuna for hyperparameter tuning. You specify
   parameter ranges and the tuner searches for optimal values:

   .. code-block:: python

      from openstef_core.mixins.param_ranges import FloatRange, IntRange
      from openstef_models.models.forecasting.xgboost_forecaster import XGBoostHyperParams

      hyperparams = XGBoostHyperParams(
          learning_rate=FloatRange(0.01, 0.3, log=True, tune=True),
          n_estimators=IntRange(50, 500, tune=True),
          max_depth=IntRange(3, 10, tune=True),
      )

   Install the tuning extra to enable this: ``pip install openstef-models[tuning]``.

   See :doc:`user_guides/hyperparameter_tuning` for a complete guide.

Getting Started
---------------

.. dropdown:: Where should I start as a new user?
   :icon: question

   Follow this path:

   1. Install OpenSTEF: ``pip install openstef``
   2. Work through the :doc:`tutorials/index` — start with the introductory tutorial
   3. Try the built-in benchmark dataset to see a full workflow in action

   The quickest way to see OpenSTEF in action:

   .. code-block:: python

      from openstef_core.testing import load_liander_dataset

      dataset = load_liander_dataset()
      print(f"Dataset shape: {dataset.data.shape}")
      print(f"Date range: {dataset.data.index.min()} to {dataset.data.index.max()}")

.. dropdown:: Is there a benchmark dataset I can use to test things?
   :icon: question

   Yes. The **Liander 2024 Energy Forecasting Benchmark** dataset is available from
   HuggingFace Hub and can be loaded directly:

   .. code-block:: python

      from openstef_core.testing import load_liander_dataset, prepare_tutorial_datasets

      dataset = load_liander_dataset()

   This dataset contains load measurements from MV feeders and transformers, versioned
   weather forecasts, EPEX electricity prices, and typical load profiles. Install the
   benchmark extra for access: ``pip install openstef-core[benchmark]``.

.. dropdown:: How do I evaluate forecast accuracy?
   :icon: question

   Accuracy depends on your use case — congestion management cares about peak detection,
   while other applications may focus on overall error metrics like RMSE.

   OpenSTEF provides evaluation tools in the ``openstef-beam`` package:

   .. code-block:: python

      from openstef_beam.evaluation.metric_providers import (
          ObservedProbabilityProvider,
          RMAEProvider,
      )

   The recommendation is to run the Alliander benchmark with ``openstef-beam`` to see
   performance metrics and plots for your specific setup.

   See :doc:`user_guides/evaluation` for details on available metrics.