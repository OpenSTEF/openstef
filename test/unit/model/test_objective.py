# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
import optuna
from test.utils import BaseTestCase, TestData

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.objective import (
    XGBRegressorObjective,
    LGBRegressorObjective,
    XGBQuantileRegressor,
)
from openstf.model.model_creator import ModelCreator

input_data = TestData.load("reference_sets/307-train-data.csv")
input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
    input_data
)
input_data_with_features = input_data_with_features.iloc[::50, :]
pj = TestData.get_prediction_job(pid=307)
N_TRIALS = 1


class TestXGBRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        objective = XGBRegressorObjective(
            input_data_with_features,
            model,
        )
        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )
        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, XGBRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestLGBRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "lgb"
        model = ModelCreator.create_model(model_type)

        objective = LGBRegressorObjective(
            input_data_with_features,
            model,
        )
        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )
        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, LGBRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestXGBQRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "xgb_quantile"
        model = ModelCreator.create_model(model_type)

        objective = XGBQuantileRegressor(
            input_data_with_features,
            model,
        )
        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )
        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, XGBQuantileRegressor)
        self.assertEqual(len(study.trials), N_TRIALS)


class ColumnOrderTest(BaseTestCase):
    def test_call(self):
        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        objective = XGBRegressorObjective(
            input_data,
            model,
        )

        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )
        with self.assertRaises(RuntimeError):
            study.optimize(objective, n_trials=N_TRIALS)


if __name__ == "__main__":
    unittest.main()
