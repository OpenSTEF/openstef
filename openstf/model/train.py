# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import os
from datetime import datetime, timedelta
from pathlib import Path

from ktpbase.database import DataBase
from ktpbase.log import logging
from ktpbase.config.config import ConfigManager

from openstf.feature_engineering.general import calc_completeness
from openstf.model.figure import (
    convert_to_base64_data_uri,
    plot_data_series,
    plot_feature_importance,
)
from openstf.model.general import (
    ForecastType,
    pre_process_data,
    split_data_train_validation_test,
)
from openstf.model.prediction.creator import PredictionModelCreator
from openstf.model.trainer.creator import ModelTrainerCreator
from openstf.monitoring.teams import send_report_teams_better, send_report_teams_worse

# TODO make this config more central
# Set thresholds
COMPLETENESS_THRESHOLD = 0.5
MINIMAL_TABLE_LENGTH = 100
MAX_AGE_YOUNG_MODEL = 7
TRAINING_PERIOD_DAYS = 90  # Note, if hyperparam is available, this value is overwritten


def is_data_sufficient(data):
    """Check if enough data is left after validation and cleaning to continue
        with model training.

    Args:
        data: pd.DataFrame() with cleaned input data.

    Returns:
        (bool): True if amount of data is sufficient, False otherwise.

    """
    logger = logging.get_logger(__name__)
    # Set output variable
    is_sufficient = True

    # Calculate completeness
    completeness = calc_completeness(data, time_delayed=True, homogenise=False)
    table_length = data.shape[0]

    # Check if completeness is up to the standards
    if completeness < COMPLETENESS_THRESHOLD:
        logger.warning(
            "Input data is not sufficient, completeness too low",
            completeness=completeness,
            completeness_threshold=COMPLETENESS_THRESHOLD,
        )
        is_sufficient = False

    # Check if absolute amount of rows is sufficient
    if table_length < MINIMAL_TABLE_LENGTH:
        logger.warning(
            "Input data is not sufficient, table length too short",
            table_length=table_length,
            table_length_threshold=MINIMAL_TABLE_LENGTH,
        )
        is_sufficient = False

    return is_sufficient


def train_model_for_specific_pj(
    pj, context, retrain_young_models=False, compare_to_old=True
):
    """Main function that controls the training process of one prediction job.

    Here, on a high level, the input data is collected, features are applied and the
    model is trained and stored. Additionally models for energy components are also
    trained if requested. Model is saved if better than the old model unless specified
    otherwise.

    Args:
        pj (dict): Prediction job
        context (openstf.task.utils.TaskContext): Task context for logging
        retrain_young_models (bool): Retrain models younger then 7 days if True
        compare_to_old (bool): Compare new trained model to old trained model if True

    Returns:
        Trained model (FIXME can be various datatypes at present)
    """
    # TODO Maybe this whole function can be replaced with a scikit learn pipeline?
    # Make model trainer creator
    mc = ModelTrainerCreator(pj)
    config = ConfigManager.get_instance()

    # Make model trainer
    model_trainer = mc.create_model_trainer()

    # if model is too young AND retrain_young_models it not set, skip training
    if model_trainer.old_model_age < MAX_AGE_YOUNG_MODEL:
        if not retrain_young_models:
            context.logger.info(
                "Skip training of model",
                model_age_days=model_trainer.old_model_age,
                retrain_young_models=retrain_young_models,
            )
            return

    # Set optimized hyper parameters if available
    model_trainer.hyper_parameters.update(context.database.get_hyper_params(pj))

    context.perf_meter.checkpoint("model-creation")

    # Specify training period
    # use hyperparam training_period_days if available
    lookback = float(
        model_trainer.hyper_parameters.get("training_period_days", TRAINING_PERIOD_DAYS)
    )
    datetime_start = datetime.utcnow() - timedelta(days=lookback)
    datetime_end = datetime.utcnow()

    # Get data from database
    data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )
    featureset_name = model_trainer.hyper_parameters["featureset_name"]
    featureset = context.database.get_featureset(featureset_name)

    # Pre-process data
    clean_data_with_features = pre_process_data(data, featureset)

    # Check if we have enough data left to continue
    if is_data_sufficient(clean_data_with_features) is False:
        raise ValueError("Input data quality insufficient, aborting!")

    # Split data in train, test and validation sets
    train_data, validation_data, test_data = split_data_train_validation_test(
        clean_data_with_features
    )

    context.perf_meter.checkpoint("preprocessing")

    # Train model
    model_trainer.train(train_data, validation_data)

    context.perf_meter.checkpoint("training")

    # Build predictor and create predictions
    predictor = PredictionModelCreator.create_prediction_model(
        pj,
        ForecastType.DEMAND,
        model_trainer.trained_model,
        model_trainer.confidence_interval,
    )
    train_predict = predictor.make_forecast(
        train_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )
    validation_predict = predictor.make_forecast(
        validation_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )
    test_predict = predictor.make_forecast(
        test_data.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )

    context.perf_meter.checkpoint("predicting")

    # Create figures
    figure_features = plot_feature_importance(model_trainer.feature_importance)
    figure_series = {
        f"Predictor{horizon}": plot_data_series(
            [train_data, validation_data, test_data],
            [train_predict, validation_predict, test_predict],
            horizon,
        )
        for horizon in train_data.Horizon.unique()
    }

    # Combine validation + train data
    combined = train_data.append(validation_data)

    # Evaluate model and store if better than old model
    # TODO Use predicted data series created above instead of predicting again
    # in better_than_old_model.
    if not compare_to_old or model_trainer.better_than_old_model(combined):
        context.logger.info("New model is better then old model", prediction_job=pj)
        model_trainer.store_model()

        # Save figures to disk
        save_loc = Path(config.paths.trained_models) / str(pj["id"])
        os.makedirs(save_loc, exist_ok=True)
        figure_features.write_html(str(save_loc / "weight_plot.html"))
        for key, fig in figure_series.items():
            fig.write_html(str(save_loc / f"{key}.html"), auto_open=False)

        send_report_teams_better(pj, model_trainer.feature_importance)
    else:
        context.logger.warning("Old model is better then new model", prediction_job=pj)

        # Save figures to disk
        save_loc = Path(config.paths.trained_models) / str(pj["id"]) / "worse_model"
        os.makedirs(save_loc, exist_ok=True)
        figure_features.write_html(str(save_loc / "weight_plot.html"))
        for key, fig in figure_series.items():
            fig.write_html(str(save_loc / "f{key}.html"), auto_open=False)

            fig.update_layout(
                margin=dict(l=10, r=20, t=35, b=0),
                title=None,
                showlegend=False,
                grid=None,
                meta=None,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            fig.write_image(save_loc / "f{key}.jpg", format="jpg", scale=0.3)
            convert_to_base64_data_uri(
                save_loc / "f{key}.jpg",
                save_loc / "f{key}.datauri",
                content_type="image/jpg",
            )

        send_report_teams_worse(pj)

    context.perf_meter.checkpoint("evaluation")


def train_specific_model(context, pid):
    """Train model for given prediction id.

    Tracy-compatible function to train a specific model based on the prediction id (pid).
    Should not be used outside of Tracy, preferred alternative:
        train_model_for_specific_pj

    Args:
        pid (int): Prediction id of the corresponding prediction job.

    Returns:
        Trained model (FIXME can be various datatypes at present)
    """
    # Get DataBase instance:
    db = DataBase()

    # Get prediction job based on the given prediction ID (pid)
    pj = db.get_prediction_job(pid)

    # Train model for pj
    train_model_for_specific_pj(
        pj, context, compare_to_old=False, retrain_young_models=True
    )
