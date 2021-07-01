# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import optuna

from openstf.enums import MLModelType
from openstf.model.model_creator import ModelCreator
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.metrics.metrics import mae

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
        test_fraction=TEST_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        eval_metric=EVAL_METRIC,
        verbose=False,
    ):
        self.input_data = input_data
        self.test_fraction = test_fraction
        self.validation_fraction = validation_fraction
        self.eval_metric = eval_metric
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

        model = ModelCreator.create_model(self.model_type)

        params = self.get_params(trial)

        model.set_params(**params)

        eval_metric = self.eval_metric

        # See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.XGBoostPruningCallback.html
        pruning_callback = optuna.integration.XGBoostPruningCallback(
            # Use validation_0 to prune on the train set
            # Use validation_1 to prune on the validation set
            trial,
            observation_key=f"validation_1-{eval_metric}",
        )
        # validation_0 and validation_1 are available
        model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=self.verbose,
            eval_metric=eval_metric,
            callbacks=[pruning_callback],
        )

        forecast_y = model.predict(test_x)

        return mae(test_y, forecast_y)


class XGBRegressorObjective(RegressorObjective):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = MLModelType.XGB

    def get_params(self, trial: optuna.trial.FrozenTrial) -> dict:
        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        return params
