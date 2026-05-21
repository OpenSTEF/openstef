Intro to Energy Forecasting
===========================

This page introduces the problem domain that OpenSTEF addresses: short-term energy load forecasting for electricity grid operators. It explains what the problem is, why it matters, and what makes it difficult. For the practical steps of producing a forecast with OpenSTEF, see :ref:`guide_forecasting`.

What is Short-Term Load Forecasting?
-------------------------------------

Short-term load forecasting (STLF) is the prediction of electrical energy demand or generation at specific points in the grid, from 15 minutes to approximately 7 days into the future. Grid operators rely on these forecasts to:

- Manage congestion at substations and cables before it occurs
- Communicate planned energy transport to upstream and downstream operators
- Optimize grid losses against market prices
- Plan maintenance windows safely

Unlike long-term planning (months or years), short-term forecasting operates at high temporal resolution (typically 15-minute intervals) and must be refreshed frequently as new measurements and weather forecasts become available.

Input Signals
-------------

Accurate load forecasts depend on combining several categories of input data, each contributing different information:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Signal Category
     - What It Provides
     - Why It Matters
   * - Load history
     - Recent and historical measurements of the target variable
     - Energy demand is highly auto-correlated; last Monday's load is the best predictor of this Monday's load
   * - Weather forecasts
     - Temperature, solar irradiance, wind speed, cloud cover
     - Drives heating/cooling demand, solar generation, and wind generation
   * - Calendar features
     - Hour of day, day of week, holidays, school vacations
     - Captures recurring human behaviour patterns (commuting, industry schedules)
   * - Market prices
     - Day-ahead and intraday electricity prices
     - Influences behaviour of price-responsive loads and generation (e.g., wind parks curtailing at negative prices)

OpenSTEF provides transforms to extract features from these raw signals. For example, :class:`~openstef_models.transforms.time_domain.DatetimeFeaturesAdder` extracts calendar integers, and :class:`~openstef_models.transforms.time_domain.LagsAdder` creates historical lag features that respect the forecast horizon. However, the library cannot compensate for missing or poor-quality input data.

Forecast Horizons
-----------------

The "horizon" (or lead time) is the time gap between when a forecast is issued and the moment being predicted. A forecast issued now for 36 hours from now has a 36-hour horizon.

Accuracy degrades with increasing lead time for two reasons:

- **Information loss:** Lag features from recent measurements become unavailable. If you are forecasting 36 hours ahead, you cannot use data from 24 hours ago because it does not yet exist at the time the forecast must be issued.
- **Weather forecast uncertainty:** Numerical weather predictions lose resolution and reliability beyond a few days. Solar and wind peaks become unpredictable as cloud and wind patterns diverge from forecasts.

OpenSTEF's ``LagsAdder`` automatically selects only lags that are valid for a given horizon, ensuring the model never trains on information that would be unavailable at prediction time.

Practical Difficulties
----------------------

Several real-world factors make energy forecasting harder than textbook time-series prediction:

**Unpredictable behaviour at low aggregation levels.** Individual customers or small groups of assets exhibit high variability. A single industrial customer changing shift patterns, or a wind park shutting down in response to negative market prices, can dominate the signal at that grid point.

**Aggregation helps, but is not always available.** Forecasting a substation serving thousands of households is substantially easier than forecasting a single customer. The law of large numbers smooths individual variability. Grid operators often need forecasts at both levels.

**Capacity changes over time.** New solar installations connect, factories relocate, and electric vehicle charging infrastructure grows. A model trained on last year's data may face a fundamentally different load profile today. Feature engineering and retraining strategies must account for non-stationarity.

**Weather forecast versioning.** The weather forecast for Monday issued on Saturday differs from the one issued on Sunday. Using the wrong version during training (one that was not actually available at the time) leads to overly optimistic accuracy estimates. OpenSTEF's :class:`~openstef_models.transforms.time_domain.VersionedLagsAdder` handles this distinction explicitly.

**Data quality issues.** Missing measurements, sensor errors, communication outages, and delayed data feeds are routine in operational settings. Robust forecasting pipelines must detect and handle these gracefully.

The Role of OpenSTEF
--------------------

OpenSTEF does not eliminate these difficulties, but it provides a structured approach to managing them:

- Feature transforms that encode domain knowledge (lags, calendar features, weather interactions)
- Models suited to the characteristics of energy data (see :doc:`/user_guide/concepts/models`)
- Probabilistic outputs that quantify forecast uncertainty (see :doc:`/user_guide/guides/probabilistic_forecasting`)
- Backtesting tools to measure real-world performance honestly (see :doc:`/user_guide/concepts/beam`)
- Reliability mechanisms for when forecasts cannot be produced (see :doc:`/user_guide/guides/reliability_fallback`)

Good input data remains essential. OpenSTEF's transforms extract signal from the data you provide, but they cannot invent information that is not there. Investing in data quality, appropriate weather forecast sources, and correct historical measurements pays dividends in forecast accuracy.

For a practical walkthrough of producing forecasts with OpenSTEF, see :ref:`guide_forecasting`.