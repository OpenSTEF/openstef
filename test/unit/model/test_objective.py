# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
from openstef.enums import ModelType
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import optuna

from openstef.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstef.model.model_creator import ModelCreator
from openstef.model.objective import (
    LGBRegressorObjective,
    LinearRegressorObjective,
    RegressorObjective,
    XGBQuantileRegressorObjective,
    XGBRegressorObjective,
)

input_data = TestData.load("reference_sets/307-train-data.csv")
input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
    input_data
)
# Select 50 data points to speedup test
input_data_with_features = input_data_with_features.iloc[::50, :]

N_TRIALS = 2


class TestRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = ModelType.XGB
        model = ModelCreator.create_model(model_type)

        objective = RegressorObjective(
            model,
            input_data_with_features,
        )
        objective.model_type = model_type  # Setting model_type is done in subclasses

        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, RegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestXGBRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        objective = XGBRegressorObjective(
            model,
            input_data_with_features,
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
            model,
            input_data_with_features,
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

        objective = XGBQuantileRegressorObjective(
            model,
            input_data_with_features,
        )
        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, XGBQuantileRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class TestLinearRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "linear"
        model = ModelCreator.create_model(model_type)

        objective = LinearRegressorObjective(
            model,
            input_data_with_features,
        )
        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )
        study.optimize(objective, n_trials=N_TRIALS)

        self.assertIsInstance(objective, LinearRegressorObjective)
        self.assertEqual(len(study.trials), N_TRIALS)


class ColumnOrderTest(BaseTestCase):
    def test_call(self):
        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        objective = XGBRegressorObjective(
            model,
            input_data,
        )

        study = optuna.create_study(
            study_name=model_type,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
            direction="minimize",
        )

        with self.assertRaises(RuntimeError):
            study.optimize(objective, n_trials=N_TRIALS)
