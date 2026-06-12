# SPDX-FileCopyrightText: 2026 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Tests for ContributionsPlotter."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_models.explainability.plotters.contributions_plotter import ContributionsPlotter

TARGET_COLUMN = "load"
BIAS_COLUMN = "bias"


@pytest.fixture
def contributions_dataset() -> TimeSeriesDataset:
    """5 features with deliberately different magnitudes, plus bias and target."""
    rng = np.random.default_rng(42)
    index = pd.date_range("2025-01-01", periods=10, freq="15min")
    data = pd.DataFrame(
        {
            "feat_a": rng.normal(10, 1, 10),  # largest
            "feat_b": rng.normal(5, 1, 10),
            "feat_c": rng.normal(2, 0.5, 10),
            "feat_d": rng.normal(1, 0.3, 10),
            "feat_e": rng.normal(0.1, 0.05, 10),  # smallest
            BIAS_COLUMN: np.full(10, 100.0),
            TARGET_COLUMN: rng.normal(120, 5, 10),
        },
        index=index,
    )
    return TimeSeriesDataset(data=data, sample_interval=timedelta(minutes=15))


ALL_PLOT_METHODS = ["plot_heatmap", "plot_waterfall", "plot_bar"]


@pytest.mark.parametrize("method", ALL_PLOT_METHODS)
def test_returns_figure(
    contributions_dataset: TimeSeriesDataset,
    method: str,
) -> None:
    # Act
    result = getattr(ContributionsPlotter, method)(contributions_dataset)

    # Assert
    assert isinstance(result, go.Figure)


@pytest.mark.parametrize("method", ALL_PLOT_METHODS)
def test_no_bias_column(method: str) -> None:
    # Arrange
    index = pd.date_range("2025-01-01", periods=5, freq="15min")
    data = pd.DataFrame({"feat_x": [1, 2, 3, 4, 5], "feat_y": [5, 4, 3, 2, 1]}, index=index)
    ds = TimeSeriesDataset(data=data, sample_interval=timedelta(minutes=15))

    # Act
    fig = getattr(ContributionsPlotter, method)(ds)

    # Assert
    assert isinstance(fig, go.Figure)


def test_heatmap_with_prediction_has_two_traces(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, show_prediction=True)

    # Assert
    assert len(fig.data) == 2


def test_heatmap_without_prediction_has_one_trace(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, show_prediction=False)

    # Assert
    assert len(fig.data) == 1


def test_heatmap_excludes_target_and_bias(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, show_prediction=False)
    y_labels = list(fig.data[0].y)

    # Assert
    assert TARGET_COLUMN not in y_labels
    assert BIAS_COLUMN not in y_labels


def test_heatmap_features_sorted_by_importance(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, top_n=5, show_prediction=False)
    y_labels = list(fig.data[0].y)

    # Assert
    assert y_labels[-1] == "feat_a"  # most important at top (last in list)
    assert y_labels[0] == "feat_e"  # least important at bottom


def test_heatmap_top_n_limits_features(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, top_n=3, show_prediction=False)
    y_labels = list(fig.data[0].y)

    # Assert
    assert len(y_labels) == 3
    assert set(y_labels) == {"feat_a", "feat_b", "feat_c"}


def test_heatmap_prediction_line_values(contributions_dataset: TimeSeriesDataset) -> None:
    """Prediction = sum(feature contributions) + bias."""
    # Arrange
    df = contributions_dataset.data
    feature_cols = [c for c in df.columns if c not in {TARGET_COLUMN, BIAS_COLUMN}]
    expected_prediction = df[feature_cols].sum(axis=1) + df[BIAS_COLUMN]

    # Act
    fig = ContributionsPlotter.plot_heatmap(contributions_dataset, show_prediction=True)
    prediction_trace = fig.data[0]

    # Assert
    np.testing.assert_array_almost_equal(np.array(prediction_trace.y), expected_prediction.to_numpy())


def test_waterfall_starts_with_bias(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_waterfall(contributions_dataset)
    wf = fig.data[0]

    # Assert
    assert wf.x[0] == BIAS_COLUMN
    assert wf.y[0] == pytest.approx(100.0)
    assert wf.measure[0] == "absolute"


def test_waterfall_ends_with_prediction_total(contributions_dataset: TimeSeriesDataset) -> None:
    # Arrange
    df = contributions_dataset.data
    feature_cols = [c for c in df.columns if c not in {TARGET_COLUMN, BIAS_COLUMN}]
    expected_prediction = float(df[BIAS_COLUMN].iloc[0]) + float(df[feature_cols].iloc[0].sum())

    # Act
    fig = ContributionsPlotter.plot_waterfall(contributions_dataset, timestep=0)
    wf = fig.data[0]

    # Assert
    assert wf.x[-1] == "Prediction"
    assert wf.measure[-1] == "total"
    assert wf.y[-1] == pytest.approx(expected_prediction)


def test_waterfall_top_n_limits_bars(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_waterfall(contributions_dataset, top_n=2)
    wf = fig.data[0]
    names = list(wf.x)

    # Assert
    assert len(names) == 5  # bias + 2 features + "other (...)" + Prediction
    assert any("other" in n for n in names)


def test_waterfall_title_contains_timestamp(contributions_dataset: TimeSeriesDataset) -> None:
    # Arrange
    expected_ts = str(contributions_dataset.data.index[3])

    # Act
    fig = ContributionsPlotter.plot_waterfall(contributions_dataset, timestep=3)

    # Assert
    assert expected_ts in fig.layout.title.text


def test_bar_excludes_target_and_bias(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_bar(contributions_dataset)
    y_labels = list(fig.data[0].y)

    # Assert
    assert TARGET_COLUMN not in y_labels
    assert BIAS_COLUMN not in y_labels


def test_bar_top_n_limits_bars(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_bar(contributions_dataset, top_n=3)
    y_labels = list(fig.data[0].y)

    # Assert
    assert len(y_labels) == 3


def test_bar_values_are_mean_absolute(contributions_dataset: TimeSeriesDataset) -> None:
    # Arrange
    df = contributions_dataset.data
    feature_cols = [c for c in df.columns if c not in {TARGET_COLUMN, BIAS_COLUMN}]
    expected = df[feature_cols].abs().mean().sort_values(ascending=True)  # reversed for plotly

    # Act
    fig = ContributionsPlotter.plot_bar(contributions_dataset, top_n=5)
    bar = fig.data[0]

    # Assert
    np.testing.assert_array_almost_equal(np.array(bar.x), expected.to_numpy())


def test_bar_most_important_feature_at_top(contributions_dataset: TimeSeriesDataset) -> None:
    # Act
    fig = ContributionsPlotter.plot_bar(contributions_dataset)
    y_labels = list(fig.data[0].y)

    # Assert plotly renders bottom-to-top, so last y label is at the top
    assert y_labels[-1] == "feat_a"
