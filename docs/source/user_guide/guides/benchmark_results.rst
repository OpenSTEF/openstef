.. SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
..
.. SPDX-License-Identifier: MPL-2.0

.. _benchmark_results:

Benchmark Results
=================

How accurate are OpenSTEF's models in practice? This page reports reference
performance on the public **Liander 2024 STEF benchmark**, so you can compare models
before committing to one. Use it together with the :ref:`Model Selection Guide
<concept_models>` (which explains *why* each model behaves the way it does) and
:doc:`BEAM </user_guide/concepts/beam>` (which explains *how* these numbers are produced).

.. note::

   These numbers are a point-in-time snapshot and may change as models, defaults, and
   the benchmark evolve. To reproduce or extend them, run the notebooks under
   :doc:`Liander 2024 </benchmarks/liander2024/README>`, then compare runs.


At a Glance
-----------

.. image:: /images/guides/liander2024_rcrps_by_group.png
   :alt: Box plot of rCRPS per model and target group on the Liander 2024 benchmark.
         The ensemble has the lowest median rCRPS in every group, gblinear is close
         behind, and xgboost trails.
   :align: center

Each box shows the distribution of per-target ``rCRPS`` within a target group (one
point per target). Lower is better.

**Takeaways**

- The **ensemble** is the most accurate model across every target group, on both the
  unweighted and the peak-weighted metric.
- **GBLinear** is a strong, consistent second and a good single-model default —
  especially where extrapolation beyond the training range matters (congestion).
- **XGBoost** alone trails the other two on this benchmark. The ensemble does not
  use it: it blends GBLinear with a LightGBM learner, pairing GBLinear's linear
  extrapolation with complementary non-linear structure.
- The gap between models *widens* under the peak-weighted metric (see
  :ref:`rCRPS sample-weighted <metric_rcrps_weighted>`), most visibly for the highly
  intermittent solar and wind targets.


.. _metrics_explained:

The Metrics
-----------

All scores on this page are variants of the **Continuous Ranked Probability Score
(CRPS)**, the standard proper scoring rule for *probabilistic* forecasts. CRPS
generalizes the absolute error to a full predictive distribution: it rewards forecasts
whose quantiles are both sharp and well-calibrated, and it is expressed in the same
units as the load. A perfect forecast scores 0.

CRPS in raw load units cannot be compared across targets of different size (a feeder
peaking at 1 MW versus one at 50 MW). The benchmark therefore reports two *relative*
variants.

rCRPS
^^^^^

**Relative CRPS** normalizes the CRPS by the operating range of the observed load —
the gap between its 1st and 99th percentile:

.. math::

   \text{rCRPS} = \frac{\text{CRPS}}{P_{99}(y) - P_{1}(y)}

This makes the score **scale-invariant**: roughly, the average distributional error as
a fraction of how much the target moves. A value of ``0.05`` means the typical
probabilistic error is about 5% of the target's operating range. Every timestamp counts
equally. Lower is better.

.. _metric_rcrps_weighted:

rCRPS (sample-weighted)
^^^^^^^^^^^^^^^^^^^^^^^^^

For grid operations the moments that matter most are **high-load periods** — that is
when congestion risk is highest. The sample-weighted variant computes the same rCRPS
but weights each timestamp by its load magnitude, so peaks dominate the score and
near-zero load is de-emphasized (down to a floor weight):

.. math::

   w_i = \operatorname{clip}\!\left(
   \left| \frac{y_i}{P_{99}(|y|)} \right|,\; 0.1,\; 1.0 \right)

Use this metric when peak accuracy is the priority. Intermittent targets (solar, wind)
score noticeably *worse* here than on the unweighted metric: they sit near zero much of
the time, so up-weighting their large, hard-to-predict peaks raises the relative error.

.. tip::

   For a single, intuitive accuracy number prefer **rCRPS**. When your use case is
   congestion management or peak shaving, lead with **rCRPS (sample-weighted)**.

.. _metric_rmae:

rMAE
^^^^

**Relative Mean Absolute Error** measures the accuracy of the **median (P50) forecast**
alone, normalized by the same operating-range denominator as rCRPS:

.. math::

   \text{rMAE} = \frac{\text{MAE}_{P50}}{P_{99}(y) - P_{1}(y)}

Use this when you care about point-forecast accuracy at the median rather than the
full probabilistic distribution.


.. _benchmark_tables:

Results by Model and Target Group
---------------------------------

Rows are models; columns are the benchmark's target groups plus the **Global** average
across all 55 targets. Each cell is the **mean metric value over the targets in that
group** (each target weighted equally). **Lower is better**; the best model per
column is in bold.

.. list-table:: rCRPS — unweighted (lower is better)
   :header-rows: 1
   :stub-columns: 1
   :widths: 18 12 12 16 14 12 12

   * - Model
     - Global
     - MV feeder
     - Station inst.
     - Transformer
     - Solar park
     - Wind park
   * - XGBoost
     - 0.065
     - 0.052
     - 0.062
     - 0.075
     - 0.052
     - 0.089
   * - GBLinear
     - 0.051
     - 0.041
     - 0.049
     - 0.059
     - 0.044
     - 0.070
   * - Ensemble
     - **0.049**
     - **0.039**
     - **0.047**
     - **0.058**
     - **0.037**
     - **0.066**

.. list-table:: rCRPS — sample-weighted / peak-focused (lower is better)
   :header-rows: 1
   :stub-columns: 1
   :widths: 18 12 12 16 14 12 12

   * - Model
     - Global
     - MV feeder
     - Station inst.
     - Transformer
     - Solar park
     - Wind park
   * - XGBoost
     - 0.082
     - 0.056
     - 0.068
     - 0.085
     - 0.113
     - 0.156
   * - GBLinear
     - 0.063
     - 0.045
     - 0.054
     - 0.069
     - 0.077
     - 0.107
   * - Ensemble
     - **0.059**
     - **0.042**
     - **0.053**
     - **0.067**
     - **0.069**
     - **0.096**

.. list-table:: rMAE — median point forecast (lower is better)
   :header-rows: 1
   :stub-columns: 1
   :widths: 18 12 12 16 14 12 12

   * - Model
     - Global
     - MV feeder
     - Station inst.
     - Transformer
     - Solar park
     - Wind park
   * - XGBoost
     - 0.084
     - 0.067
     - 0.079
     - 0.095
     - 0.067
     - 0.111
   * - GBLinear
     - 0.084
     - 0.067
     - 0.079
     - 0.094
     - 0.070
     - 0.110
   * - Ensemble
     - **0.078**
     - **0.063**
     - **0.074**
     - **0.089**
     - **0.062**
     - **0.103**


How These Numbers Were Produced
-------------------------------

.. list-table::
   :stub-columns: 1
   :widths: 30 70

   * - Dataset
     - `Liander 2024 STEF benchmark <https://huggingface.co/datasets/OpenSTEF/liander2024-stef-benchmark>`_
       — 55 real grid targets across 5 groups (MV feeders, station installations,
       transformers, solar parks, wind parks).
   * - Models
     - ``xgboost`` (:class:`~openstef_models.models.forecasting.xgboost_forecaster.XGBoostForecaster`),
       ``gblinear`` (:class:`~openstef_models.models.forecasting.gblinear_forecaster.GBLinearForecaster`),
       and ``ensemble`` (an openstef-meta learned-weight combination of
       LightGBM and GBLinear base models).
   * - Forecast moment
     - Day-ahead, with all inputs restricted to what was available at **D-1 06:00**
       (no future data leakage).
   * - Evaluation
     - Sequential :doc:`BEAM </user_guide/concepts/beam>` backtest over 2024. rCRPS is computed per target
       from quantile forecasts (normalization range :math:`P_1`–:math:`P_{99}`), then
       averaged within each group.

For the full methodology — how the backtest prevents leakage and how metrics are
segmented — see :doc:`BEAM </user_guide/concepts/beam>`. To benchmark your *own* model or data on the same
footing, see the :doc:`Build Your Own </benchmarks/custom/README>` benchmarks.
