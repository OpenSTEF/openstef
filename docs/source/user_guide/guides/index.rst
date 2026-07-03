.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _guides:

======
Guides
======

Step-by-step instructions for common OpenSTEF tasks.

.. grid:: 1 1 2 2
    :gutter: 4
    :padding: 2 2 0 0

    .. grid-item-card:: :fa:`bullseye` Forecasting
        :link: forecasting
        :link-type: doc

        Build and run forecasting pipelines end-to-end.

    .. grid-item-card:: :fa:`gears` Feature Engineering
        :link: feature_engineering_tutorial
        :link-type: doc

        Create, select, and manage features for your models.

    .. grid-item-card:: :fa:`database` Datasets
        :link: datasets_tutorial
        :link-type: doc

        Prepare and validate input data for training and prediction.

    .. grid-item-card:: :fa:`chart-area` Probabilistic Forecasting
        :link: probabilistic_forecasting
        :link-type: doc

        Generate prediction intervals and quantile forecasts.

    .. grid-item-card:: :fa:`shield-halved` Reliability & Fallback
        :link: reliability_fallback
        :link-type: doc

        Handle missing data, detect flatlining, and configure fallback strategies.

    .. grid-item-card:: :fa:`clock-rotate-left` Backtesting
        :link: backtesting_tutorial
        :link-type: doc

        Evaluate model performance on historical data with rolling windows.

    .. grid-item-card:: :fa:`ranking-star` Benchmark Results
        :link: benchmark_results
        :link-type: doc

        Reference accuracy of each model on the public Liander 2024 benchmark.

    .. grid-item-card:: :fa:`server` Deployment
        :link: deployment
        :link-type: doc

        Deploy OpenSTEF models in production environments.

    .. grid-item-card:: :fa:`wand-sparkles` Foundation Model Forecasting
        :link: foundation_model_forecasting
        :link-type: doc

        Checkpoints, execution providers for your hardware, batching, and backtesting.

    .. grid-item-card:: :fa:`file-lines` Logging
        :link: /user_guide/logging
        :link-type: doc

        Configure logging and monitor pipeline execution.

.. toctree::
    :maxdepth: 1
    :hidden:

    forecasting
    Feature Engineering <feature_engineering_tutorial>
    Datasets <datasets_tutorial>
    probabilistic_forecasting
    reliability_fallback
    Backtesting <backtesting_tutorial>
    Benchmark Results <benchmark_results>
    deployment
    Foundation Model Forecasting <foundation_model_forecasting>
    /user_guide/logging
