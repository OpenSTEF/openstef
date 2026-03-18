# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from openstef_core.datasets.validated_datasets import EnsembleForecastDataset, ForecastInputDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.learned_weights_combiner import (
    LGBMCombinerHyperParams,
    LogisticCombinerHyperParams,
    RFCombinerHyperParams,
    WeightsCombiner,
    XGBCombinerHyperParams,
)


@pytest.fixture(params=["lgbm", "xgboost", "rf", "logistic"])
def classifier(request: pytest.FixtureRequest) -> str:
    """Fixture to provide different classifier types for LearnedWeightsCombiner tests."""
    return request.param


@pytest.fixture
def combiner(classifier: str) -> WeightsCombiner:
    """Fixture to create a WeightsCombiner based on the classifier type."""
    if classifier == "lgbm":
        hp = LGBMCombinerHyperParams(n_leaves=5, n_estimators=10)
    elif classifier == "xgboost":
        hp = XGBCombinerHyperParams(n_estimators=10)
    elif classifier == "rf":
        hp = RFCombinerHyperParams(n_estimators=10, n_leaves=5)
    elif classifier == "logistic":
        hp = LogisticCombinerHyperParams()
    else:
        msg = f"Unsupported classifier type: {classifier}"
        raise ValueError(msg)

    return WeightsCombiner(hyperparams=hp, quantiles=[Q(0.1), Q(0.5), Q(0.9)], horizons=[LeadTime(timedelta(days=1))])


def test_quantile_weights_combiner__fit_predict(
    ensemble_dataset: EnsembleForecastDataset,
    combiner: WeightsCombiner,
):
    """Test basic fit and predict workflow with comprehensive output validation."""
    # Arrange
    expected_quantiles = combiner.quantiles

    # Act
    combiner.fit(ensemble_dataset)
    result = combiner.predict(ensemble_dataset)

    # Assert
    assert combiner.is_fitted, "Model should be fitted after calling fit()"

    expected_columns = [q.format() for q in expected_quantiles]
    expected_columns.append("load")
    assert list(result.data.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(result.data.columns)}"
    )

    assert not result.data.isna().any().any(), "Forecast should not contain NaN or None values"

    stds = result.data.std()
    assert (stds > 0).all(), f"All columns should have variation, got stds: {dict(stds)}"


def test_weights_combiner_not_fitted_error(
    ensemble_dataset: EnsembleForecastDataset,
    combiner: WeightsCombiner,
):
    """Test that NotFittedError is raised when predicting before fitting."""
    with pytest.raises(NotFittedError):
        combiner.predict(ensemble_dataset)


def test_quantile_weights_combiner__fit_with_additional_features_shorter_index(
    ensemble_dataset: EnsembleForecastDataset,
) -> None:
    """Fit should succeed when additional_features has fewer rows than the ensemble dataset.

    Regression test: combine_forecast_input_datasets performs an inner join, which drops
    rows not present in additional_features. Labels must be reindexed to match the
    combined dataset to avoid a shape mismatch in sample_weight computation.
    """
    # Arrange — additional_features covers only a subset of ensemble timestamps
    full_index = ensemble_dataset.data.index
    subset_index = full_index[1:]  # drop the first timestamp

    rng = np.random.default_rng(42)
    additional_features = ForecastInputDataset(
        data=pd.DataFrame(
            {"extra_feature": rng.normal(size=len(subset_index)), "load": rng.normal(size=len(subset_index))},
            index=subset_index,
        ),
        sample_interval=ensemble_dataset.sample_interval,
        target_column="load",
    )

    combiner = WeightsCombiner(
        hyperparams=LGBMCombinerHyperParams(n_leaves=5, n_estimators=10),
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime(timedelta(days=1))],
    )

    # Act — should not raise ValueError about broadcast shapes
    combiner.fit(ensemble_dataset, additional_features=additional_features)

    # Assert
    assert combiner.is_fitted


def test_predict_renormalizes_weights_when_base_model_predictions_are_nan() -> None:
    """Predict should renormalize weights when a base model has NaN predictions.

    Regression test: when one base model cannot predict certain timestamps (e.g.
    gblinear limited to 2-day horizon while lgbm predicts 7 days), the combiner
    must redistribute the missing model's weight to the remaining models. Without
    renormalization, sum(axis=1, skipna=True) drops the NaN contribution, causing
    predictions to be systematically scaled down.
    """
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=100, freq="15min")

    # Two forecasters: lgbm has all values, gblinear is NaN for the last 50 rows
    lgbm_vals = rng.normal(1000, 100, 100)
    gblinear_vals = rng.normal(1000, 100, 100).copy()
    gblinear_vals[50:] = np.nan

    data = pd.DataFrame(
        {
            "LGBMForecaster__quantile_P10": lgbm_vals * 0.8,
            "LGBMForecaster__quantile_P50": lgbm_vals,
            "LGBMForecaster__quantile_P90": lgbm_vals * 1.2,
            "GBLinearForecaster__quantile_P10": gblinear_vals * 0.8,
            "GBLinearForecaster__quantile_P50": gblinear_vals,
            "GBLinearForecaster__quantile_P90": gblinear_vals * 1.2,
            "load": rng.normal(1000, 100, 100),
        },
        index=index,
    )
    dataset = EnsembleForecastDataset(data=data, sample_interval=timedelta(minutes=15))

    combiner = WeightsCombiner(
        hyperparams=LGBMCombinerHyperParams(n_leaves=5, n_estimators=10),
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime(timedelta(days=1))],
    )
    combiner.fit(dataset)

    # Act
    result = combiner.predict(dataset)

    # Assert — rows where gblinear is NaN should still produce valid (non-NaN) predictions
    nan_rows = result.data[["quantile_P10", "quantile_P50", "quantile_P90"]].iloc[50:]
    assert not nan_rows.isna().any().any(), (
        "Predictions should not be NaN when at least one base model has valid predictions"
    )
    # And the predictions should be in the ballpark of the lgbm values (not scaled down)
    assert nan_rows["quantile_P50"].mean() > 500, (
        "Predictions in the NaN region should not be systematically scaled down"
    )
