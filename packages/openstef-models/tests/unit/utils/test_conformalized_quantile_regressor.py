# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from numpy.random import default_rng
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.linear_model import QuantileRegressor
from xgboost import XGBRegressor

from openstef_models.utils.conformalized_quantile_regressor import ConformalizedQuantileRegressor, ParamType

ParamDict = dict[str, ParamType]
BaseLearner = BaseEstimator


class BaseLearnerConfig(BaseModel):
    base_learner: type[BaseLearner]
    quantile_param: str
    hyperparams: ParamDict


@pytest.fixture
def dataset() -> tuple[pd.DataFrame, pd.Series]:
    n_samples = 100
    n_features = 5
    rng = default_rng()
    X = pd.DataFrame(rng.random((n_samples, n_features)))
    y = pd.Series(rng.random(n_samples))
    return X, y


@pytest.fixture(params=["sklearn_quantile", "lgbm", "xgboost"])
def baselearner_config(request: pytest.FixtureRequest) -> BaseLearnerConfig:  # type : ignore
    model: str = request.param
    if model == "sklearn_quantile":
        return BaseLearnerConfig(
            base_learner=QuantileRegressor,
            quantile_param="quantile",
            hyperparams={"alpha": 0.1, "solver": "highs", "fit_intercept": True},
        )
    if model == "lgbm":
        return BaseLearnerConfig(
            base_learner=LGBMRegressor,  # type: ignore
            quantile_param="alpha",
            hyperparams={
                "objective": "quantile",
                "n_estimators": 10,
                "learning_rate": 0.1,
                "max_depth": -1,
            },
        )
    return BaseLearnerConfig(
        base_learner=XGBRegressor,
        quantile_param="quantile_alpha",
        hyperparams={
            "objective": "reg:quantileerror",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "max_depth": 3,
        },
    )


def test_init_sets_quantiles_and_models(baselearner_config: BaseLearnerConfig):
    quantiles = [0.1, 0.5, 0.9]

    model = ConformalizedQuantileRegressor(
        base_learner=baselearner_config.base_learner,
        quantile_param=baselearner_config.quantile_param,
        quantiles=quantiles,
        hyperparams=baselearner_config.hyperparams,
    )

    assert model.quantiles == quantiles


def test_fit_and_predict_shape(dataset: tuple[pd.DataFrame, pd.Series], baselearner_config: BaseLearnerConfig):
    quantiles = [0.1, 0.5, 0.9]

    X, y = dataset[0], dataset[1]
    model = ConformalizedQuantileRegressor(
        base_learner=baselearner_config.base_learner,
        quantile_param=baselearner_config.quantile_param,
        quantiles=quantiles,
        hyperparams=baselearner_config.hyperparams,
    )

    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0], len(quantiles))


def test_is_fitted_true_after_fit(dataset: tuple[pd.DataFrame, pd.Series], baselearner_config: BaseLearnerConfig):
    quantiles = [0.1, 0.5, 0.9]
    X, y = dataset[0], dataset[1]
    model = ConformalizedQuantileRegressor(
        base_learner=baselearner_config.base_learner,
        quantile_param=baselearner_config.quantile_param,
        quantiles=quantiles,
        hyperparams=baselearner_config.hyperparams,
    )
    model.fit(X, y)
    assert model.is_fitted
