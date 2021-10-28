# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest

import optuna

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.model_creator import ModelCreator
from openstf.model.objective import (
    RegressorObjective,
    XGBRegressorObjective,
    LGBRegressorObjective,
    XGBQuantileRegressorObjective,
)
from test.utils import BaseTestCase, TestData

input_data = TestData.load("reference_sets/307-train-data.csv")
input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
    input_data
)
# Select 50 data points to speedup test
input_data_with_features = input_data_with_features.iloc[::50, :]

N_TRIALS = 2


class TestRegressorObjective(BaseTestCase):
    def test_call(self):
        pj = TestData.get_prediction_job(pid=307)
        model = ModelCreator.create_model(pj)

        objective = RegressorObjective(
            model,
            input_data_with_features,
        )

        study = optuna.create_study(
            study_name=pj["model"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, RegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestXGBRegressorObjective(BaseTestCase):
    def test_call(self):
        pj = TestData.get_prediction_job(pid=307)
        model = ModelCreator.create_model(pj)

        objective = XGBRegressorObjective(
            model,
            input_data_with_features,
        )
        study = optuna.create_study(
            study_name=pj["model"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, XGBRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestLGBRegressorObjective(BaseTestCase):
    def test_call(self):
        pj = TestData.get_prediction_job(pid=307)
        pj["model"] = "lgb"
        model = ModelCreator.create_model(pj)

        objective = LGBRegressorObjective(
            model,
            input_data_with_features,
        )
        study = optuna.create_study(
            study_name=pj["model"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, LGBRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestXGBQRegressorObjective(BaseTestCase):
    def test_call(self):
        pj = TestData.get_prediction_job(pid=307)
        pj["model"] = "xgb_quantile"
        model = ModelCreator.create_model(pj)

        objective = XGBQuantileRegressorObjective(
            model,
            input_data_with_features,
        )
        study = optuna.create_study(
            study_name=pj["model"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, XGBQuantileRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class ColumnOrderTest(BaseTestCase):
    def test_call(self):
        pj = TestData.get_prediction_job(pid=307)
        model = ModelCreator.create_model(pj)

        objective = XGBRegressorObjective(
            model,
            input_data,
        )

        study = optuna.create_study(
            study_name=pj["model"],
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        with self.assertRaises(RuntimeError):
            study.optimize(objective, n_trials=N_TRIALS)


if __name__ == "__main__":
    unittest.main()
