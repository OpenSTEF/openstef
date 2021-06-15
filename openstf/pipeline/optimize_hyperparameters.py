import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_absolute_error

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.model_creator import ModelCreator


from sklearn.model_selection import train_test_split

# https://optuna.readthedocs.io/en/stable/faq.html#objective-func-additional-args
class XGBRegressorObjective:

    def __init__(self, model_type, input_data):
        self.model_type = model_type
        self.input_data = input_data

    def __call__(self, trial):
        # Perform data preprocessing
        data = self.input_data

        train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.25)


        model = ModelCreator.create_model(model_type=self.model_type)

        params = {
            "eta": trial.suggest_float("eta", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        pruning_callback = optuna.integration.XGBoostPruningCallback(
            trial, "validation-auc"
        )

        model.set_params(params)
        model.fit(
            train_x,
            train_y,
            eval_set=eval_set,
            early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
            verbose=False,
        )
        # model = xgb.train(
        #     params, dtrain, evals=[(dvalid, "validation")], callbacks=[pruning_callback]
        # )
        # prediction = model.predict(dvalid)

        mean_absolute_error




# optuna.integration.XGBoostPruningCallback

class ObjectiveCreator:
    def create_objective(model_type):
        # if type(model) is XGBRegressor:
        if model_type == "xgb":
            return XGBRegressorObjective


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

    study.optimize(Objective(input_data))