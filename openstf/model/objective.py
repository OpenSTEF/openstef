# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
import optuna

from openstf.metrics import metrics
from openstf.model.regressors.regressor_interface import OpenstfRegressorInterface
from openstf.enums import MLModelType
from datetime import datetime
from openstf.model_selection.model_selection import split_data_train_validation_test

EARLY_STOPPING_ROUNDS: int = 10
TEST_FRACTION: float = 0.1
VALIDATION_FRACTION: float = 0.1
# See https://xgboost.readthedocs.io/en/latest/parameter.html for all possibilities
EVAL_METRIC: str = "mae"

# https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args
class RegressorObjective:
    """Regressor optuna objective function.

    Use any of the derived classses for optimization using an optuna study.
    The constructor is used to set the "input_data" and optionally add some
    configuration. Next the instance will be called by he optuna study during
    optimization.

    Example:
        # initialize a (derived class) objective function
        objective = XGBRegressorObjective(input_data, test_fraction)
        # use the objective function
        study.optimize(objective)
    """

    def __init__(
        self,
        input_data,
        model: OpenstfRegressorInterface,
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        eval_metric=EVAL_METRIC,
        verbose=False,
    ):
        self.input_data = input_data
        self.model = model
        self.start_time = datetime.utcnow()
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.eval_metric = eval_metric
        self.eval_metric_function = metrics.get_eval_metric_function(eval_metric)
        self.verbose = verbose
        # Should be set on a derived classes
        self.model_type = None

    def __call__(self, trial: optuna.trial.FrozenTrial) -> float:
        """Optuna objective function.

        Args: trial

        Returns:
            float: Mean absolute error for this trial.
        """
        # Perform data preprocessing
        train_data, validation_data, test_data = split_data_train_validation_test(
            self.input_data,
            test_fraction=self.test_fraction,
            validation_fraction=self.validation_fraction,
            backtest=True,
        )

        # Test if first column is "load" and last column is "horizon"
        if train_data.columns[0] != "load" or train_data.columns[-1] != "horizon":
            raise RuntimeError(
                "Column order in train input data not as expected, "
                "could not train a model!"
            )

        # Split in x, y data (x are the features, y is the load)
        train_x, train_y = train_data.iloc[:, 1:-1], train_data.iloc[:, 0]
        valid_x, valid_y = validation_data.iloc[:, 1:-1], validation_data.iloc[:, 0]
        test_x, test_y = test_data.iloc[:, 1:-1], test_data.iloc[:, 0]

        # Configure evals for early stopping
        eval_set = [(train_x, train_y), (valid_x, valid_y)]

        # get the parameters used in this trial
        hyper_params = self.get_params(trial)

        # insert parameters into model
        self.model.set_params(**hyper_params)

        # create the specific pruning callback
        pruning_callback = self.get_pruning_callback(trial)
        if pruning_callback is None:
            callbacks = None
        else:
            callbacks = [pruning_callback]

        # validation_0 and validation_1 are available
        self.model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=self.verbose,
            eval_metric=self.eval_metric,
            callbacks=callbacks,
        )

        forecast_y = self.model.predict(test_x)

        return self.eval_metric_function(test_y, forecast_y)

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for objective without model specific get_params function.

        Args: trial

        Returns:
            dict: {parameter: hyperparameter_value}
        """
        default_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0),
            "subsample": trial.suggest_float("subsample", 0.5, 0.99),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 1, 10),
        }

        # Compare the list to the default parameter space
        model_parameters = self.model.get_params()
        keys = [x for x in model_parameters.keys() if x in default_params.keys()]
        # create a dictionary with the matching parameters
        params = {parameter: default_params[parameter] for parameter in keys}

        return params

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return None


class XGBRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.XGB

    # extend the parameters with the model specific ones per implementation
    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for XGB Regressor Objective
        with objective specific parameters.

            Args: trial

            Returns:
                dict: {parameter: hyperparameter_value}
        """
        # Filtered default parameters
        model_params = super().get_params(trial)

        # XGB specific parameters
        params = {
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        }
        return {**model_params, **params}

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return optuna.integration.XGBoostPruningCallback(
            trial, observation_key=f"validation_1-{self.eval_metric}"
        )


class LGBRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.LGB

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for LGB Regressor Objective
        with objective specific parameters.

            Args: trial

            Returns:
                dict: {parameter: hyperparameter_value}
        """
        # Filtered default parameters
        model_params = super().get_params(trial)

        # LGB specific parameters
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 62),
            "boosting_type": trial.suggest_categorical(
                "boosting_type", ["gbdt", "dart", "rf"]
            ),
            "tree_learner": trial.suggest_categorical(
                "tree_learner", ["serial", "feature", "data", "voting"]
            ),
            "n_estimators": trial.suggest_int("n_estimators", 50, 150),
            "min_split_gain": trial.suggest_float("min_split_gain", 1e-8, 1),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        }
        return {**model_params, **params}

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        metric = self.eval_metric
        if metric == "mae":
            metric = "l1"
        return optuna.integration.LightGBMPruningCallback(
            trial, metric=metric, valid_name="valid_1"
        )


class XGBQuantileRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.XGB_QUANTILE

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for XGBQuantile Regressor Objective
        with objective specific parameters.

            Args: trial

            Returns:
                dict: {parameter: hyperparameter_value}
        """
        # Filtered default parameters
        model_params = super().get_params(trial)

        # XGB specific parameters
        params = {
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0),
        }
        return {**model_params, **params}

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return optuna.integration.XGBoostPruningCallback(
            trial, observation_key=f"validation_1-{self.eval_metric}"
        )
