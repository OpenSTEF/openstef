# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for ModelContributionsPlotter."""

from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from openstef_core.datasets import ForecastDataset, TimeSeriesDataset
from openstef_core.types import Q
from openstef_models.explainability import ModelContributionsPlotter

SAMPLE_INTERVAL = timedelta(minutes=15)


@pytest.fixture
def contributions_dataset() -> TimeSeriesDataset:
    """Two model weights across two quantiles, summing to 1.0 per row."""
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    gb_p50 = np.linspace(0.6, 0.4, 8)
    data = pd.DataFrame(
        {
            "gblinear__quantile_P50": gb_p50,
            "lgbm__quantile_P50": 1.0 - gb_p50,
            "gblinear__quantile_P90": np.full(8, 0.5),
            "lgbm__quantile_P90": np.full(8, 0.5),
            "load": np.linspace(16, 30, 8),
        },
        index=index,
    )
    return TimeSeriesDataset(data=data, sample_interval=SAMPLE_INTERVAL)


@pytest.fixture
def forecast_dataset() -> ForecastDataset:
    """Ensemble forecast with P50 and P90 quantile columns."""
    index = pd.date_range("2025-01-01", periods=8, freq="15min")
    data = pd.DataFrame(
        {
            "load": np.linspace(16, 30, 8),
            "quantile_P10": np.linspace(10, 20, 8),
            "quantile_P50": np.linspace(15, 28, 8),
            "quantile_P90": np.linspace(18, 32, 8),
        },
        index=index,
    )
    return ForecastDataset(data=data, sample_interval=SAMPLE_INTERVAL)


def test_returns_figure(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    result = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset)
    assert isinstance(result, go.Figure)


def test_has_forecast_and_stacked_weights_in_separate_panels(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset, quantile=Q(0.5))
    assert len(fig.data) == 3
    assert fig.data[0].xaxis == "x"
    assert all(trace.xaxis == "x2" for trace in fig.data[1:])
    assert all(trace.stackgroup == "contrib" for trace in fig.data[1:])


def test_selects_requested_quantile(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset, quantile=Q(0.5))
    assert [trace.name for trace in fig.data[1:]] == ["gblinear", "lgbm"]
    assert all("P90" not in trace.name for trace in fig.data[1:] if trace.name is not None)


def test_selected_weights_and_forecast_use_the_same_quantile(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(
        contributions_dataset,
        forecast=forecast_dataset,
        quantile=Q(0.9),
    )

    assert [trace.name for trace in fig.data[1:]] == ["gblinear", "lgbm"]
    np.testing.assert_array_almost_equal(fig.data[0].y, forecast_dataset.data["quantile_P90"].to_numpy())
    expected_weights = contributions_dataset.data["gblinear__quantile_P90"].to_numpy()
    np.testing.assert_array_almost_equal(fig.data[1].y, expected_weights)
    assert fig.layout.yaxis.title.text == "Prediction (P90)"
    assert fig.layout.yaxis2.title.text == "Model weight (P90)"


def test_top_panel_shows_prediction(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset, quantile=Q(0.5))
    top_trace = fig.data[0]
    expected = forecast_dataset.data["quantile_P50"].to_numpy()
    np.testing.assert_array_almost_equal(top_trace.y, expected)


def test_weights_sum_to_one(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset, quantile=Q(0.5))
    weight_traces = fig.data[1:]
    stacked = np.sum([np.array(t.y, dtype=float) for t in weight_traces], axis=0)
    np.testing.assert_array_almost_equal(stacked, 1.0)


def test_missing_quantile_raises(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    with pytest.raises(ValueError, match="quantile_P25"):
        ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast=forecast_dataset, quantile=Q(0.25))


def test_default_quantile_is_p50(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
) -> None:
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast_dataset)

    np.testing.assert_array_almost_equal(fig.data[0].y, forecast_dataset.data["quantile_P50"].to_numpy())
    np.testing.assert_array_almost_equal(fig.data[1].y, contributions_dataset.data["gblinear__quantile_P50"].to_numpy())


def test_figure_exports_as_standalone_html(
    contributions_dataset: TimeSeriesDataset,
    forecast_dataset: ForecastDataset,
    tmp_path: Path,
) -> None:
    html_path = tmp_path / "model-contributions.html"
    fig = ModelContributionsPlotter.plot_stacked_area(contributions_dataset, forecast_dataset)

    fig.write_html(html_path, include_plotlyjs=True)

    html = html_path.read_text()
    assert "plotly.js" in html
    assert "gblinear" in html
