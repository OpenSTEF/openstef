.. SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _concept_foundation_models:

Foundation Models
=================

A foundation model forecasts a time series it has never seen, with no training. It is
pretrained once on a large collection of time series, and from then on it can predict a new
meter directly from that meter's recent history. This is the opposite of how the rest of
OpenSTEF works, where you fit an XGBoost or LightGBM model on one meter's history before it
can predict that meter.

That difference gives you three things:

- **No per-target training.** One pretrained model forecasts every meter. You do not train
  and store a separate model for each of thousands of targets.
- **Works from day one.** A meter with only a short history, or one you have never forecast
  before, can be predicted straight away.
- **A baseline to start from.** Before you invest in feature engineering and tuning, a
  foundation model gives you a reference forecast to measure against.

How a foundation model forecasts
--------------------------------

A foundation model is trained once on a large and varied collection of time series: electricity
loads, but also many other kinds of measurement. Across all of them it learns the shapes that
recur in time series in general, such as daily and weekly cycles, slow trends, and the way a
value responds to a change like a drop in temperature. When you give it a new meter, it matches
those learned shapes to the meter's recent history and continues the series forward.

The comparison that usually helps: a language model reads a large amount of text and can then
continue a sentence it has never seen before. A foundation model for forecasting has read a
large number of time series and can continue yours the same way, without being retrained on it.

Because the model is already trained, there is nothing to fit. You give it a window of a
meter's recent values and it returns a forecast for the horizon ahead, together with a spread
that expresses how uncertain it is. That spread is a set of quantiles, the same probabilistic
output the rest of OpenSTEF produces, so a foundation-model forecast drops into the same plots,
metrics, and pipelines.

Foundation model or trained model?
----------------------------------

Both kinds of model live in OpenSTEF, and they are good at different things. The table below is
a starting point; :doc:`beam` is how you settle the choice on your own data.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - Foundation model
     - Trained model (XGBoost, LightGBM)
   * - Training
     - None. Pretrained, used as is.
     - Fit on each meter's own history.
   * - Best when
     - A meter is new, you have many meters, or you want a quick baseline.
     - A meter has a long history and you can afford to tune it.
   * - Cost to run
     - Heavier: more compute per forecast.
     - Light: cheap to run once trained.
   * - Fit to one meter
     - Applies general patterns, not this meter's own.
     - Learns this meter's own calendar and weather response.

Where it fits in OpenSTEF
-------------------------

``openstef-foundation-models`` is the package that holds these models. Each one is wrapped
behind the same :class:`~openstef_models.models.forecasting.forecaster.Forecaster` interface
as every other OpenSTEF model, so it plugs into the same workflows and backtesting. Chronos-2
is the first model family in the package; others can be added behind the same interface.

.. note::

   For the trainable models this sits alongside, see :doc:`models`. To compare a foundation
   model against them, see :doc:`beam`. To run one yourself, start with the
   :doc:`Foundation Model Forecasting Quickstart </user_guide/getting_started/foundation_model_forecasting_quickstart>`
   and then the
   :doc:`Foundation Model Forecasting guide </user_guide/guides/foundation_model_forecasting>`.

Covariates
----------

Often you already know something about the days ahead, such as a weather forecast. A foundation
model can take these extra series, called covariates, alongside the target and use them in its
prediction. You pass them the same way you pass features to any other OpenSTEF model: keep the
column, and the model reads both its past values and its known future values. Given a
temperature forecast, the model can raise its prediction for a cold day it has been told is
coming.

Hardware requirements
---------------------

A trained XGBoost or LightGBM model is a handful of small decision trees. It runs on a plain
CPU in milliseconds and needs little memory. A foundation model is a neural network with
millions of parameters, and every forecast runs the whole network, so it needs more compute and
more memory per prediction.

That cost is why hardware matters here in a way it does not for the trained models. The same
network runs faster on a GPU or on Apple Silicon than on a CPU, because both speed up the matrix
arithmetic it is built from. It still runs on a plain CPU, only slower, so you can start without
special hardware and add an accelerator when you need more throughput. OpenSTEF detects the
machine and picks a sensible default; the
:doc:`Foundation Model Forecasting guide </user_guide/guides/foundation_model_forecasting>`
covers choosing the hardware yourself and matching model size to machine.

.. seealso::

   - :ref:`concept_models` for the trainable forecasters this sits alongside.
   - :ref:`concept_beam` for comparing a foundation model against them.
   - :doc:`Foundation Model Forecasting Quickstart </user_guide/getting_started/foundation_model_forecasting_quickstart>` to produce a forecast end-to-end.
   - :doc:`Foundation Model Forecasting guide </user_guide/guides/foundation_model_forecasting>` for checkpoints, hardware, batching, and backtesting.
   - :doc:`API reference </api/foundation_models>` for the ``openstef-foundation-models`` package.
