from openstf.model.objective_creator import ObjectiveCreator
import pandas as pd
import optuna

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator


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