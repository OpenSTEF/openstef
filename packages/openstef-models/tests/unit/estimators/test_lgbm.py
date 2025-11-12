# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import pytest
from numpy.random import default_rng

from openstef_models.estimators.lgbm import LGBMQuantileRegressor


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    n_samples = 100
    n_features = 5
    rng = default_rng()
    X = pd.DataFrame(rng.random((n_samples, n_features)))
    y = pd.Series(rng.random(n_samples))
    return X, y


def test_init_sets_quantiles_and_models():
    quantiles = [0.1, 0.5, 0.9]
    model = LGBMQuantileRegressor(quantiles=quantiles, linear_tree=False)
    assert model.quantiles == quantiles
    assert len(model._models) == len(quantiles)


def test_fit_and_predict_shape(dataset: tuple[pd.DataFrame, pd.Series]):
    quantiles = [0.1, 0.5, 0.9]
    X, y = dataset[0], dataset[1]
    model = LGBMQuantileRegressor(quantiles=quantiles, linear_tree=False, n_estimators=5)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0], len(quantiles))


def test_sklearn_is_fitted_true_after_fit(dataset: tuple[pd.DataFrame, pd.Series]):
    quantiles = [0.1, 0.5, 0.9]
    X, y = dataset[0], dataset[1]
    model = LGBMQuantileRegressor(quantiles=quantiles, linear_tree=False, n_estimators=2)
    model.fit(X, y)
    assert model.__sklearn_is_fitted__()
