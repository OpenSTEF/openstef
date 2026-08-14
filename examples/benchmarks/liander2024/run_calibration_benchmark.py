# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Liander 2024 benchmark for quantile calibration methods.

This development benchmark trains one GBLinear forecaster, reserves a held-out
calibration window, and compares uncalibrated, isotonic, and reference-aligned
conformalized forecasts on a separate holdout.
"""

# %% [markdown]
# # Quantile calibration benchmark
#
# Compare `IsotonicQuantileCalibrator` and
# `ConformalizedQuantileCalibrator` on the Liander 2024 STEF benchmark dataset.
# This notebook trains one GBLinear forecaster and keeps the calibration window
# separate from the final holdout.
#
# This is a development-only benchmark and is intentionally not linked from the
# committed documentation.

# %% [markdown]
# ## Benchmark protocol
#
# 1. Fit GBLinear on the training window.
# 2. Predict a held-out calibration window immediately after training.
# 3. Fit isotonic and reference-aligned conformalized calibration independently
#    on that calibration window.
# 4. Evaluate raw, isotonic, and conformalized forecasts on a later holdout.
# 5. Optionally apply the downstream `QuantileSorter` to the conformalized output
#    and report the direct and sorted results.
#
# The reported metrics include P50 MAE, mean absolute calibration error (MACE),
# observed quantile levels, P10-P90 coverage and width, and quantile ordering.

# %%
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt

from openstef_core.datasets import ForecastDataset
from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster
from openstef_models.transforms.postprocessing import (
    ConformalizedQuantileCalibrator,
    IsotonicQuantileCalibrator,
    QuantileSorter,
)

# %%
QUANTILES = [Q(0.1), Q(0.5), Q(0.9)]
TRAIN_START = datetime.fromisoformat("2024-03-01T00:00:00+00:00")
TRAIN_DAYS = 45
CALIBRATION_DAYS = 7
HOLDOUT_DAYS = 7
OUTPUT_DIR = Path("benchmark_results/calibration")
USE_LOCAL_ISOTONIC = True
SORT_QUANTILES = True


def make_config() -> ForecastingWorkflowConfig:
    """Create the small GBLinear configuration used by both methods."""
    return ForecastingWorkflowConfig(
        model_id="liander2024_calibration_benchmark",
        model="gblinear",
        horizons=[LeadTime.from_string("PT36H")],
        quantiles=QUANTILES,
        target_column="load",
        temperature_column="temperature_2m",
        relative_humidity_column="relative_humidity_2m",
        wind_speed_column="wind_speed_10m",
        radiation_column="shortwave_radiation",
        pressure_column="surface_pressure",
        verbosity=0,
        mlflow_storage=None,
        gblinear_hyperparams=GBLinearForecaster.HyperParams(n_steps=50),
    )


def column(quantile: Q) -> str:
    """Return the forecast column name for a quantile."""
    return f"quantile_P{int(float(quantile) * 100)}"


def window(forecast: pd.DataFrame, actuals: pd.Series, start: datetime, end: datetime) -> ForecastDataset:
    """Create a calibration or holdout dataset from aligned forecasts and actuals."""
    data = forecast.drop(columns=["load"], errors="ignore").join(actuals.rename("load"), how="inner")
    return ForecastDataset(data=data.loc[start:end], forecast_start=start, target_column="load")


def score(method: str, dataset: ForecastDataset) -> dict[str, float | int | str]:
    """Calculate point, interval, coverage, and quantile-ordering metrics."""
    data = dataset.data.dropna()
    actual = data["load"]
    lower, median, upper = (data[column(q)] for q in QUANTILES)
    observed = [(actual <= data[column(q)]).mean() for q in QUANTILES]
    expected = [float(q) for q in QUANTILES]
    return {
        "method": method,
        "rows": len(data),
        "p50_mae": float((actual - median).abs().mean()),
        "mace": float(
            sum(
                abs(observed_level - expected_level)
                for observed_level, expected_level in zip(observed, expected, strict=True)
            )
            / len(expected)
        ),
        "p10_observed": float(observed[0]),
        "p50_observed": float(observed[1]),
        "p90_observed": float(observed[2]),
        "p10_p90_coverage": float(((actual >= lower) & (actual <= upper)).mean()),
        "p10_p90_width": float((upper - lower).mean()),
        "quantile_order_rate": float(((lower <= median) & (median <= upper)).mean()),
    }


# %%
def plot_forecasts(
    holdout: ForecastDataset,
    calibrators: dict[str, IsotonicQuantileCalibrator | ConformalizedQuantileCalibrator],
    output_path: Path,
) -> None:
    """Save and display a representative holdout time-series plot."""
    conformalized = calibrators["conformalized"].transform(holdout)
    selected = QuantileSorter().transform(conformalized) if SORT_QUANTILES else conformalized
    selected_label = "Conformalized (sorted)" if SORT_QUANTILES else "Conformalized (direct)"
    plot_data = holdout.data.join(
        calibrators["isotonic"].transform(holdout).data.add_suffix("_isotonic"),
        how="inner",
    ).join(selected.data.add_suffix("_conformalized"), how="inner")
    plot_data = plot_data.iloc[: min(len(plot_data), 7 * 96)]

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.plot(plot_data.index, plot_data["load"], color="black", linewidth=1.5, label="Actual")
    ax.plot(plot_data.index, plot_data["quantile_P50"], color="#7f8c8d", label="Raw P50")
    ax.plot(plot_data.index, plot_data["quantile_P50_isotonic"], color="#1f77b4", label="Isotonic P50")
    ax.plot(plot_data.index, plot_data["quantile_P50_conformalized"], color="#d62728", label=f"{selected_label} P50")
    ax.fill_between(
        plot_data.index,
        plot_data["quantile_P10_isotonic"],
        plot_data["quantile_P90_isotonic"],
        color="#1f77b4",
        alpha=0.12,
        label="Isotonic P10-P90",
    )
    ax.fill_between(
        plot_data.index,
        plot_data["quantile_P10_conformalized"],
        plot_data["quantile_P90_conformalized"],
        color="#d62728",
        alpha=0.12,
        label=f"{selected_label} P10-P90",
    )
    ax.set_title("Liander 2024 quantile calibration on the holdout period")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Load")
    ax.grid(visible=True, alpha=0.25)
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(output_path, dpi=150)
    if "get_ipython" in globals():
        plt.show()
    plt.close(fig)


# %%
def run() -> pd.DataFrame:
    """Run the calibration comparison and save its results."""
    dataset = load_liander_dataset()
    calibration_start = TRAIN_START + timedelta(days=TRAIN_DAYS)
    holdout_start = calibration_start + timedelta(days=CALIBRATION_DAYS)
    holdout_end = holdout_start + timedelta(days=HOLDOUT_DAYS)

    train = dataset.filter_by_range(start=TRAIN_START, end=calibration_start)
    prediction_input = dataset.filter_by_range(
        start=calibration_start - timedelta(days=14),
        end=holdout_end,
    )
    workflow = create_forecasting_workflow(config=make_config())
    workflow.fit(train)
    raw_forecast = workflow.predict(prediction_input, forecast_start=calibration_start).data
    actuals = dataset.data["load"].reindex(raw_forecast.index)

    calibration = window(raw_forecast, actuals, calibration_start, holdout_start - timedelta(minutes=15))
    holdout = window(raw_forecast, actuals, holdout_start, holdout_end)
    results = [score("raw", holdout)]

    calibrators = {
        "isotonic": IsotonicQuantileCalibrator(
            quantiles=QUANTILES,
            use_local_quantile_estimation=USE_LOCAL_ISOTONIC,
        ),
        "conformalized": ConformalizedQuantileCalibrator(quantiles=QUANTILES),
    }
    for name, calibrator in calibrators.items():
        calibrator.fit(calibration)
        transformed = calibrator.transform(holdout)
        results.append(score(name, transformed))
        if name == "conformalized" and SORT_QUANTILES:
            results.append(score("conformalized_sorted", QuantileSorter().transform(transformed)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_forecasts(holdout, calibrators, OUTPUT_DIR / "timeseries.png")
    metrics = pd.DataFrame(results)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": "OpenSTEF/liander2024-stef-benchmark",
                "train_start": TRAIN_START.isoformat(),
                "calibration_start": calibration_start.isoformat(),
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "train_days": TRAIN_DAYS,
                "calibration_days": CALIBRATION_DAYS,
                "holdout_days": HOLDOUT_DAYS,
                "isotonic_use_local_quantile_estimation": USE_LOCAL_ISOTONIC,
                "sort_quantiles": SORT_QUANTILES,
                "quantiles": [float(q) for q in QUANTILES],
            },
            indent=2,
        )
    )
    return metrics


# %%
def main() -> None:
    """Run the benchmark and print its metrics."""
    metrics = run()
    print(metrics.to_string(index=False))


# %% [markdown]
# ## Interpret the result
#
# Lower P50 MAE and MACE are generally better. P10 and P90 observed levels
# should be close to 0.1 and 0.9, while P10-P90 coverage should be close to
# 0.8. Compare interval width together with coverage; a wider interval can
# improve coverage without improving sharpness. When `SORT_QUANTILES` is true,
# the `conformalized_sorted` row reports the downstream-sorted output. The output
# directory contains `metrics.csv`, `metadata.json`, and `timeseries.png`.

# %%
if __name__ == "__main__":
    main()
