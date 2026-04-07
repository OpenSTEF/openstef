# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_models.transforms.general import Clipper  # alias still works
from openstef_models.utils.feature_selection import FeatureSelection


@pytest.fixture
def train_dataset() -> TimeSeriesDataset:
    """Training dataset with three features A, B, C."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]},
            index=pd.date_range("2025-01-01", periods=3, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


@pytest.fixture
def test_dataset() -> TimeSeriesDataset:
    """Test dataset with values outside training ranges."""
    return TimeSeriesDataset(
        data=pd.DataFrame(
            {"A": [0.5, 4.0], "B": [5.0, 35.0], "C": [150.0, 350.0]},
            index=pd.date_range("2025-01-06", periods=2, freq="1h"),
        ),
        sample_interval=timedelta(hours=1),
    )


# ---------------------------
# CLIP BEHAVIOR (DEFAULT)
# ---------------------------

@pytest.mark.parametrize(
    ("mode", "expected_a", "expected_b"),
    [
        pytest.param("minmax", [1.0, 3.0], [10.0, 30.0], id="minmax_clip"),
        pytest.param("standard", [0.5, 4.0], [5.0, 35.0], id="standard_clip"),
    ],
)
def test_clipper__clip_behavior(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
    expected_a: list[float],
    expected_b: list[float],
):
    clipper = Clipper(
        selection=FeatureSelection(include={"A", "B"}),
        mode=mode,
        n_std=2.0,
        outlier_action="clip",
    )

    clipper.fit(train_dataset)
    transformed = clipper.transform(test_dataset)

    expected_df = pd.DataFrame(
        {
            "A": expected_a,
            "B": expected_b,
            "C": [150.0, 350.0],
        },
        index=test_dataset.index,
    )

    pd.testing.assert_frame_equal(transformed.data, expected_df)


# ---------------------------
# NAN BEHAVIOR (NEW FEATURE)
# ---------------------------

@pytest.mark.parametrize(
    ("mode", "expected_a", "expected_b"),
    [
        pytest.param("minmax", [np.nan, np.nan], [np.nan, np.nan], id="minmax_nan"),
        pytest.param("standard", [0.5, 4.0], [5.0, 35.0], id="standard_nan_no_clip"),
    ],
)
def test_clipper__nan_behavior(
    train_dataset: TimeSeriesDataset,
    test_dataset: TimeSeriesDataset,
    mode: str,
    expected_a: list[float],
    expected_b: list[float],
):
    clipper = Clipper(
        selection=FeatureSelection(include={"A", "B"}),
        mode=mode,
        n_std=2.0,
        outlier_action="nan",
    )

    clipper.fit(train_dataset)
    transformed = clipper.transform(test_dataset)

    expected_df = pd.DataFrame(
        {
            "A": expected_a,
            "B": expected_b,
            "C": [150.0, 350.0],
        },
        index=test_dataset.index,
    )

    pd.testing.assert_frame_equal(transformed.data, expected_df)


# ---------------------------
# MISSING FEATURES
# ---------------------------

def test_clipper__handles_missing_features():
    train_data = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0], "C": [100.0, 200.0, 300.0]},
        index=pd.date_range("2025-01-01", periods=3, freq="1h"),
    )
    train_dataset = TimeSeriesDataset(train_data, timedelta(hours=1))

    test_data = pd.DataFrame(
        {"A": [0.5, 4.0], "C": [150.0, 350.0]},
        index=pd.date_range("2025-01-06", periods=2, freq="1h"),
    )
    test_dataset = TimeSeriesDataset(test_data, timedelta(hours=1))

    clipper = Clipper(selection=FeatureSelection(include={"A", "B"}), mode="minmax")

    clipper.fit(train_dataset)
    transformed = clipper.transform(test_dataset)

    expected_df = pd.DataFrame(
        {"A": [1.0, 3.0], "C": [150.0, 350.0]},
        index=test_dataset.index,
    )

    pd.testing.assert_frame_equal(transformed.data, expected_df)


# ---------------------------
# NOT FITTED ERROR
# ---------------------------

def test_clipper__transform_without_fit(test_dataset: TimeSeriesDataset):
    """Test that transform raises error when called without fitting."""
    clipper = Clipper(selection=FeatureSelection(include={"A", "B"}))

    with pytest.raises(NotFittedError):
        clipper.transform(test_dataset)