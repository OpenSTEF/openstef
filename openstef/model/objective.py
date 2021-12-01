# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import copy
from datetime import datetime

import optuna
import pandas as pd

from openstef.enums import MLModelType
from openstef.metrics import metrics
from openstef.metrics.reporter import Report, Reporter
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model.standard_deviation_generator import StandardDeviationGenerator
from openstef.model_selection.model_selection import split_data_train_validation_test

EARLY_STOPPING_ROUNDS: int = 10
TEST_FRACTION: float = 0.15
VALIDATION_FRACTION: float = 0.15
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
        model: OpenstfRegressor,
        input_data: pd.DataFrame,
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        eval_metric=EVAL_METRIC,
        verbose=False,
    ):
        self.input_data = input_data
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        self.model = model
        self.start_time = datetime.utcnow()
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.eval_metric = eval_metric
        self.eval_metric_function = metrics.get_eval_metric_function(eval_metric)
        self.verbose = verbose
        # Should be set on a derived classes
        self.model_type = None
        self.track_trials = {}

    def __call__(
        self,
        trial: optuna.trial.FrozenTrial,
    ) -> float:
        """Optuna objective function.

        Args: trial

        Returns:
            float: Mean absolute error for this trial.
        """
        # Perform data preprocessing
        if self.model_type == MLModelType.ProLoaf:
            stratification_min_max = False
        else:
            stratification_min_max = True
        (
            peaks,
            peaks_val_train,
            self.train_data,
            self.validation_data,
            self.test_data,
        ) = split_data_train_validation_test(
            self.input_data,
            test_fraction=self.test_fraction,
            validation_fraction=self.validation_fraction,
            stratification_min_max=stratification_min_max,
            back_test=True,
        )

        # Test if first column is "load" and last column is "horizon"
        if (
            self.train_data.columns[0] != "load"
            or self.train_data.columns[-1] != "horizon"
        ):
            raise RuntimeError(
                "Column order in train input data not as expected, "
                "could not train a model!"
            )

        # Split in x, y data (x are the features, y is the load)
        train_x, train_y = self.train_data.iloc[:, 1:-1], self.train_data.iloc[:, 0]
        valid_x, valid_y = (
            self.validation_data.iloc[:, 1:-1],
            self.validation_data.iloc[:, 0],
        )
        test_x, test_y = self.test_data.iloc[:, 1:-1], self.test_data.iloc[:, 0]

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

        self.model.feature_importance_dataframe = self.model.set_feature_importance()

        # Do confidence interval determination
        self.model = StandardDeviationGenerator(
            self.validation_data
        ).generate_standard_deviation_data(self.model)

        forecast_y = self.model.predict(test_x)
        score = self.eval_metric_function(test_y, forecast_y)

        self.track_trials[f" trial: {trial.number}"] = {
            "score": score,
            "params": hyper_params,
        }
        trial.set_user_attr(key="model", value=copy.deepcopy(self.model))
        return score

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for objective without model specific get_params function.

        Args: trial

        Returns:
            dict: {parameter: hyperparameter_value}
        """
        default_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5),
            "alpha": trial.suggest_float("alpha", 0, 1.0),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0),
            "subsample": trial.suggest_float("subsample", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 16),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        }

        # Compare the list to the default parameter space
        model_parameters = self.model.get_params()
        keys = [x for x in model_parameters.keys() if x in default_params.keys()]
        # create a dictionary with the matching parameters
        params = {parameter: default_params[parameter] for parameter in keys}

        return params

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return None

    def get_trial_track(self) -> dict:
        """Get a dictionary of al trials

        Returns:
            dict: dict with al trials and it's parameters

        """
        return self.track_trials

    def create_report(self, model: OpenstfRegressor) -> Report:
        """Generate a report from the data available inside the objective function

        Args:
            model: OpenstfRegressor, model to create a report on

        Returns:
            Report: report about the model
        """
        # Report about the training process
        reporter = Reporter(self.train_data, self.validation_data, self.test_data)
        report = reporter.generate_report(model)

        return report

    @classmethod
    def get_default_values(cls) -> dict:
        return {
            "learning_rate": 0.3,
            "alpha": 0.0,
            "lambda": 1.0,
            "subsample": 1.0,
            "min_child_weight": 1,
            "max_depth": 6,
            "colsample_bytree": 1,
            "max_delta_step": 0,
        }


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
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        }
        return {**model_params, **params}

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return optuna.integration.XGBoostPruningCallback(
            trial, observation_key=f"validation_1-{self.eval_metric}"
        )

    @classmethod
    def get_default_values(cls) -> dict:

        default_parameter_values = super().get_default_values()
        default_parameter_values.update({"gamma": 0.0, "booster": "gbtree"})
        return default_parameter_values


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


class ProLoafRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.ProLoaf

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for ProLoaf Regressor Objective
        with objective specific parameters.

            Args: trial

            Returns:
                dict: {parameter: hyperparameter_value}
        """
        # Filtered default parameters
        model_params = super().get_params(trial)

        # ProLoaf specific parameters
        params = {
            # TODO: look into optimizing this pipeline for proloaf
            # "relu_leak": trial.suggest_float("relu_leak", 0.1, 1.0),
            # "core_layers": trial.suggest_int("core_layers", 1, 3),
            # "rel_linear_hidden_size": trial.suggest_float(
            #    "rel_linear_hidden_size", 0.1, 1
            # ),
            # "rel_core_hidden_size": trial.suggest_float("rel_core_hidden_size", 0.1, 1),
            # "dropout_fc": trial.suggest_float("dropout_fc", 0.1, 0.9),
            # "dropout_core": trial.suggest_float("dropout_core", 0.1, 0.9),
            # "early_stopping_patience": trial.suggest_int(
            #    "early_stopping_patience", 5, 10
            # ),
            # "early_stopping_margin": trial.suggest_float(
            #    "early_stopping_margin", 0.1, 0.9
            # ),
            "max_epochs": trial.suggest_int(
                "max_epochs", 1, 1
            ),  # TODO: change after having availability to gpu resource
            "batch_size": trial.suggest_int("batch_size", 1, 24),
        }
        return {**model_params, **params}

    def get_pruning_callback(self, trial: optuna.trial.FrozenTrial):
        return optuna.integration.PyTorchLightningPruningCallback(
            trial, monitor="val_loss"
        )


class LinearRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.LINEAR

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        """get parameters for Linear Regressor Objective
        with objective specific parameters.
            Args: trial
            Returns:
                dict: {parameter: hyperparameter_value}
        """

        # Imputation strategy
        params = {
            "imputation_strategy": trial.suggest_categorical(
                "imputation_strategy", ["mean", "median", "most_frequent"]
            ),
        }
        return params
