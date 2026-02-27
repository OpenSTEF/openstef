# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import timedelta

import pytest

from openstef_core.datasets import TimeSeriesDataset
from openstef_core.datasets.validated_datasets import EnsembleForecastDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.types import LeadTime, Q
from openstef_meta.models.forecast_combiners.stacking_combiner import StackingCombiner
from openstef_models.models.forecasting.gblinear_forecaster import GBLinearForecaster
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster


@pytest.fixture(params=["lgbm", "gblinear"])
def regressor(request: pytest.FixtureRequest) -> str:
    """Fixture to provide different regressor types for Stacking tests."""
    return request.param


def _make_template(regressor: str) -> GBLinearForecaster | LGBMForecaster:
    """Build a lightweight template forecaster for tests."""
    horizon = LeadTime(timedelta(days=1))
    if regressor == "lgbm":
        return LGBMForecaster(
            config=LGBMForecaster.Config(
                hyperparams=LGBMForecaster.HyperParams(num_leaves=5, n_estimators=10),
                horizons=[horizon],
                quantiles=[Q(0.5)],
            ),
        )
    if regressor == "gblinear":
        return GBLinearForecaster(
            config=GBLinearForecaster.Config(
                hyperparams=GBLinearForecaster.HyperParams(n_steps=10),
                horizons=[horizon],
                quantiles=[Q(0.5)],
            ),
        )
    msg = f"Unsupported regressor type: {regressor}"
    raise ValueError(msg)


@pytest.fixture
def combiner(regressor: str) -> StackingCombiner:
    """Fixture to create a StackingCombiner based on the regressor type."""
    return StackingCombiner(
        meta_forecaster=_make_template(regressor),
        quantiles=[Q(0.1), Q(0.5), Q(0.9)],
        horizons=[LeadTime(timedelta(days=1))],
    )


def test_stacking_combiner__fit_predict(
    ensemble_dataset: EnsembleForecastDataset,
    combiner: StackingCombiner,
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
    assert list(result.data.columns) == expected_columns, (
        f"Expected columns {expected_columns}, got {list(result.data.columns)}"
    )

    assert not result.data.isna().any().any(), "Forecast should not contain NaN or None values"


def test_stacking_combiner_not_fitted_error(
    ensemble_dataset: EnsembleForecastDataset,
    combiner: StackingCombiner,
):
    """Test that NotFittedError is raised when predicting before fitting."""
    with pytest.raises(NotFittedError):
        combiner.predict(ensemble_dataset)


def test_stacking_combiner_predict_contributions(
    ensemble_dataset: EnsembleForecastDataset,
    combiner: StackingCombiner,
):
    """Test that predict_contributions returns a TimeSeriesDataset with correct shape."""
    combiner.fit(ensemble_dataset)

    contributions = combiner.predict_contributions(ensemble_dataset)

    assert isinstance(contributions, TimeSeriesDataset), "Contributions should be a TimeSeriesDataset."
    # Expect (quantiles × forecasters) feature columns + 1 bias per quantile + target
    n_features = len(ensemble_dataset.forecaster_names)
    n_quantiles = len(ensemble_dataset.quantiles)
    # Each quantile model returns (features + bias) columns
    expected_cols = n_quantiles * (n_features + 1) + 1  # +1 for target column
    assert len(contributions.data.columns) == expected_cols
