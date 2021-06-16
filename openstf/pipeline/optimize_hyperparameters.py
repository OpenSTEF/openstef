import pandas as pd
import optuna
from sklearn.metrics import mean_absolute_error
import xgboost

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.model_creator import ModelCreator
from openstf.model_selection.model_selection import split_data_train_validation_test

from sklearn.model_selection import train_test_split


EARLY_STOPPING_ROUNDS: int = 10

from openstf.enums import MLModelType

# https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args
class XGBRegressorObjective:

    EARLY_STOPPING_ROUNDS: int = 10

    def __init__(self, input_data):
        self.input_data = input_data
        # self.test_size = test_size

    def __call__(self, trial):
        # Perform data preprocessing
        train_data, validation_data, _ = split_data_train_validation_test(
            self.input_data
        )

        train_x, train_y = train_data.iloc[:, 1:], train_data.iloc[:, 0]
        valid_x, valid_y = validation_data.iloc[:, 1:], validation_data.iloc[:, 0]
        # data, target = self.input_data.iloc[:, 1:], self.input_data.iloc[:, 0]

        # train_x, valid_x, train_y, valid_y = train_test_split(
        #     data, target, test_size=self.test_size
        # )

        # Configure evals for early stopping
        eval_set = [(train_x, train_y), (valid_x, valid_y)]

        model = ModelCreator.create_model(MLModelType.XGB)

        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model.set_params(**params)

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            # trial, "validation-auc"
            trial, observation_key="validation_1-mae"
        )
        # validation_0 and validation_1 are available
        model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose=True,
            eval_metric="mae",
            callbacks=[pruning_callback]
            # fit() got an unexpected keyword argument 'fit_params'
            # fit_params={'callbacks': [pruning_callback]}
        )
        # model = xgb.train(
        #     params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback]
        # )
        forecast_y = model.predict(valid_x)

        return mean_absolute_error(valid_y, forecast_y)


class ObjectiveCreator:
    def create_objective(model_type: Union[MLModelType, str]):
        # if type(model) is XGBRegressor:
        if model_type == "xgb":
            return XGBRegressorObjective


def optimize_hyperparameters_task():
    from ktpbase.database import DataBase
    from datetime import datetime, timedelta

    db = DataBase()

    pj = db.get_prediction_job(pid=307)

    pj["hyper_params"] = db.get_hyper_params(pj)

    datetime_start = datetime.utcnow() - timedelta(
        # days=int(pj["hyper_params"]["training_period_days"])
        days=2
    )
    datetime_end = datetime.utcnow()

    input_data = db.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    optimize_hyperparameters_pipeline(pj, input_data)


def optimize_hyperparameters_pipeline(
    pj: dict,
    input_data: pd.DataFrame,
):
    input_data_with_features = TrainFeatureApplicator(horizons=[0.25, 24]).add_features(
        input_data
    )

    # Create relevant model
    Objective = ObjectiveCreator.create_objective(model_type=pj["model"])

    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), direction="minimize"
    )

    study.optimize(
        Objective(input_data_with_features),
        n_trials=8,
        timeout=200,
        show_progress_bar=True
    )

    optimized_hyperparams = study.best_params
    optimized_error = study.best_value

    print(optimized_error)
    print(optimized_hyperparams)

    return optimized_hyperparams