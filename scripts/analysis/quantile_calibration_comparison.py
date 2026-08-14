# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Benchmark isotonic and MAPIE quantile calibration on a holdout period.

This is an exploratory follow-up benchmark, separate from the documentation
Tutorial. It keeps calibration rows separate from the final holdout and writes
machine-readable metrics without changing repository state.
"""

# ruff: noqa: INP001

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import pandas as pd

mpl.use("Agg")

import matplotlib.pyplot as plt

from openstef_core.datasets import ForecastDataset
from openstef_core.testing import load_liander_dataset
from openstef_core.types import LeadTime, Q
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow
from openstef_models.presets.forecasting_workflow import GBLinearForecaster
from openstef_models.transforms.postprocessing import (
    ConformalizedQuantileCalibrator,
    IsotonicQuantileCalibrator,
    MapieQuantileCalibrator,
    QuantileSorter,
)

QUANTILES = [Q(0.1), Q(0.5), Q(0.9)]


def parse_args() -> argparse.Namespace:
    """Parse benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("quantile_calibration_comparison_output"),
    )
    parser.add_argument("--train-days", type=int, default=45)
    parser.add_argument("--calibration-days", type=int, default=7)
    parser.add_argument("--holdout-days", type=int, default=7)
    parser.add_argument("--local-isotonic", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def make_config() -> ForecastingWorkflowConfig:
    """Create the benchmark forecasting configuration."""
    return ForecastingWorkflowConfig(
        model_id="quantile_calibration_benchmark",
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


def quantile_column(quantile: Q) -> str:
    """Return the forecast column name for a quantile."""
    return f"quantile_P{int(float(quantile) * 100)}"


def make_calibration_dataset(
    forecast: pd.DataFrame,
    actuals: pd.Series,
    start: datetime,
    end: datetime,
) -> ForecastDataset:
    """Build a ForecastDataset for fitting one calibrator."""
    forecast_columns = forecast.drop(columns=["load"], errors="ignore")
    data = forecast_columns.join(actuals.rename("load"), how="inner").loc[start:end]
    return ForecastDataset(data=data, forecast_start=start, target_column="load")


def score(name: str, data: pd.DataFrame) -> dict[str, float | str]:
    """Calculate point, quantile, interval, and ordering metrics."""
    aligned = data.dropna()
    p10 = aligned[quantile_column(Q(0.1))]
    p50 = aligned[quantile_column(Q(0.5))]
    p90 = aligned[quantile_column(Q(0.9))]
    actual = aligned["load"]
    expected = [0.1, 0.5, 0.9]
    observed = [(actual <= aligned[quantile_column(q)]).mean() for q in QUANTILES]
    calibration_errors = [
        observed_level - expected_level for observed_level, expected_level in zip(observed, expected, strict=True)
    ]
    return {
        "method": name,
        "rows": len(aligned),
        "p50_mae": float((actual - p50).abs().mean()),
        "mace": float(pd.Series(calibration_errors).abs().mean()),
        "p10_observed": float(observed[0]),
        "p50_observed": float(observed[1]),
        "p90_observed": float(observed[2]),
        "p10_90_coverage": float(((actual >= p10) & (actual <= p90)).mean()),
        "p10_90_width": float((p90 - p10).mean()),
        "quantile_order_rate": float((p10 <= p50).mul(p50 <= p90).mean()),
    }


def next_output_dir(root: Path) -> tuple[str, Path]:
    """Return the next versioned output directory below ``root``."""
    versions = [int(match.group(1)) for path in root.glob("v*") if (match := re.fullmatch(r"v(\d+)", path.name))]
    version = f"v{max(versions, default=0) + 1:03d}"
    return version, root / version


def plot_comparison(
    holdout: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    output_path: Path,
) -> None:
    """Plot actuals, median forecasts, and P10-P90 intervals."""
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(holdout.index, holdout["load"] / 1e6, color="black", linewidth=1.5, label="Actual load")
    colors = {
        "raw": "tab:blue",
        "isotonic": "tab:orange",
        "mapie": "tab:green",
        "conformalized": "tab:red",
    }
    for name, forecast in forecasts.items():
        color = colors[name]
        axis.plot(
            forecast.index,
            forecast[quantile_column(Q(0.5))] / 1e6,
            color=color,
            linewidth=1.2,
            label=f"{name.title()} P50",
        )
        if name != "raw":
            axis.fill_between(
                forecast.index,
                forecast[quantile_column(Q(0.1))] / 1e6,
                forecast[quantile_column(Q(0.9))] / 1e6,
                color=color,
                alpha=0.12,
                label=f"{name.title()} P10-P90",
            )
    axis.set_title("Quantile calibration comparison")
    axis.set_xlabel("Timestamp")
    axis.set_ylabel("Load (MW)")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run(args: argparse.Namespace) -> pd.DataFrame:
    """Run the comparison benchmark."""
    dataset = load_liander_dataset()
    train_start = datetime.fromisoformat("2024-03-01T00:00:00Z")
    calibration_start = train_start + timedelta(days=args.train_days)
    holdout_start = calibration_start + timedelta(days=args.calibration_days)
    holdout_end = holdout_start + timedelta(days=args.holdout_days)

    train = dataset.filter_by_range(start=train_start, end=calibration_start)
    application = dataset.filter_by_range(
        start=calibration_start - timedelta(days=14),
        end=holdout_end,
    )
    workflow = create_forecasting_workflow(config=make_config())
    workflow.fit(train)
    raw_forecast = workflow.predict(application, forecast_start=calibration_start).data
    actuals = dataset.data["load"].reindex(raw_forecast.index)

    calibration = make_calibration_dataset(
        raw_forecast,
        actuals,
        calibration_start,
        holdout_start - timedelta(minutes=15),
    )
    holdout = make_calibration_dataset(raw_forecast, actuals, holdout_start, holdout_end)
    results = [score("raw", holdout.data)]
    plot_forecasts = {"raw": holdout.data}

    calibrators = {
        "isotonic": IsotonicQuantileCalibrator(
            quantiles=QUANTILES,
            use_local_quantile_estimation=args.local_isotonic,
        ),
        "mapie": MapieQuantileCalibrator(quantiles=QUANTILES),
        "conformalized": ConformalizedQuantileCalibrator(quantiles=QUANTILES),
    }
    for name, calibrator in calibrators.items():
        calibrator.fit(calibration)
        calibrated = calibrator.transform(holdout)
        results.append(score(name, calibrated.data))
        sorted_calibrated = QuantileSorter().transform(calibrated)
        results.append(score(f"{name}_sorted", sorted_calibrated.data))
        plot_forecasts[name] = sorted_calibrated.data

    version, output_dir = next_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics = pd.DataFrame(results)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    plot_comparison(holdout.data, plot_forecasts, output_dir / "comparison.png")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": version,
                "train_start": train_start.isoformat(),
                "calibration_start": calibration_start.isoformat(),
                "holdout_start": holdout_start.isoformat(),
                "holdout_end": holdout_end.isoformat(),
                "train_days": args.train_days,
                "calibration_days": args.calibration_days,
                "holdout_days": args.holdout_days,
                "isotonic_use_local_quantile_estimation": args.local_isotonic,
                "quantiles": [float(q) for q in QUANTILES],
            },
            indent=2,
        )
    )
    return metrics


def main() -> None:
    """Run the benchmark and print its metrics."""
    args = parse_args()
    output = run(args).to_string(index=False)
    sys.stdout.write(f"{output}\n")


if __name__ == "__main__":
    main()
