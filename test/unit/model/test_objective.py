# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import unittest
import optuna
from test.utils import BaseTestCase, TestData
from unittest.mock import MagicMock, patch
from datetime import datetime

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.objective_creator import ObjectiveCreator
from openstf.model.objective import RegressorObjective
from openstf.model.model_creator import ModelCreator
from openstf.pipeline.optimize_hyperparameters import get_relevant_model_paramspace
input_data = TestData.load("reference_sets/307-train-data.csv")
input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24.0]).add_features(
            input_data
        )
pj = TestData.get_prediction_job(pid=60)

default_paramspace: dict = {
    # General parameters
    "learning_rate": ((0.01, 0.2), True),
    "alpha": ((1e-8, 1.0), True),
    "lambda": ((1e-8, 1.0), True),

    "subsample": ((0.5, 0.99), False),
    "min_child_weight": ((1, 6), False),
    "max_depth": ((3, 10), False),
    "colsample_bytree": ((0.5, 1.0), False),
    "max_delta_step": ((1, 10), False),
}
# Important parameters, model specific
# XGB specific
xgb_paramspace: dict = {
    "gamma": ((1e-8, 1.0), True),
    "booster": ["gbtree", "dart"],
    # , "gblinear" gives warnings because it doesn't use { colsample_bytree, gamma, max_delta_step, max_depth, min_child_weight, subsample }
}

# LGB specific
lgb_paramspace: dict = {
    "num_leaves": ((16, 62), False),
    "boosting_type": ['gbdt', 'dart', 'rf'],
    "tree_learner": ['serial', 'feature', 'data', 'voting'],
    "n_estimators": ((50, 150), False),
    "min_split_gain": ((1e-8, 1), True),
    "subsample_freq": ((1, 10), False),
}

#TODO: improve speed

class TestXGBRegressorObjective(BaseTestCase):
    def test_call(self):



        model_type = "xgb"
        model = ModelCreator.create_model(model_type)

        default_paramspace.update(**xgb_paramspace, **lgb_paramspace)
        model_params = get_relevant_model_paramspace(model, default_paramspace)
        start_time = datetime.utcnow()
        objective = ObjectiveCreator.create_objective(model_type)
        objective = objective(input_data_with_features, model, optuna.integration.XGBoostPruningCallback, model_params, start_time, observation_key = "validation_1-{}".format("mae"))
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
        )
        study.optimize( objective,
                       n_trials=2)

        self.assertIsInstance(study.best_trial.params,dict)


class TestLGBRegressorObjective(BaseTestCase):
    def test_call(self):
        model_type = "lgb"
        model = ModelCreator.create_model(model_type)

        default_paramspace.update(**xgb_paramspace, **lgb_paramspace)
        model_params = get_relevant_model_paramspace(model, default_paramspace)
        start_time = datetime.utcnow()
        objective = ObjectiveCreator.create_objective(model_type)
        objective = objective(input_data_with_features, model, optuna.integration.LightGBMPruningCallback, model_params,
                              start_time, metric = "l1", valid_name = "valid_1" )
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
        )
        study.optimize(objective,
                       n_trials=2)

        self.assertIsInstance(study.best_trial.params, dict)

class TestXGBQRegressorObjective(BaseTestCase):
    def test_call(self):



        model_type = "xgb_quantile"
        model = ModelCreator.create_model(model_type)

        default_paramspace.update(**xgb_paramspace, **lgb_paramspace)
        model_params = get_relevant_model_paramspace(model, default_paramspace)
        start_time = datetime.utcnow()
        objective = ObjectiveCreator.create_objective(model_type)
        objective = objective(input_data_with_features, model, optuna.integration.XGBoostPruningCallback, model_params, start_time, observation_key = "validation_1-{}".format("mae"))
        study = optuna.create_study(
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
        )
        study.optimize( objective,
                       n_trials=2)

        self.assertIsInstance(study.best_trial.params,dict)

if __name__ == "__main__":
    unittest.main()
