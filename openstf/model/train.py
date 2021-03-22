# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import namedtuple

from ktpbase.database import DataBase
from ktpbase.log import logging

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

split_model_data = namedtuple("model_data", ["train", "validation", "test"])


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


def train_model_pipeline(pj, context, retrain_young_models=False, compare_to_old=True):
    """Main function that controls the training process of one prediction job.

    On a high level: first, the input data is collected, features are applied and the
    model is trained and stored. Additionally models for energy components are also
    trained if requested. Model is saved if better than the old model unless specified
    otherwise.

    Args:
        pj (dict): Prediction job
        context (openstf.task.utils.TaskContext): Task context for logging
        retrain_young_models (bool): Retrain models younger then 7 days if True.
            Defaults to False.
        compare_to_old (bool): Only save new model when its better than the old model.
            Defaults to True.
    """

    # TODO Maybe this whole function can be replaced with a scikit learn pipeline?
    model_trainer = create_model_trainer(pj, context, retrain_young_models)

    if model_trainer is None:
        # previous young model is not retrained, end function
        return

    context.perf_meter.checkpoint("model-creation")

    split_input_data = preprocess_for_model_training(pj, context)

    context.perf_meter.checkpoint("preprocessing")

    train_model(model_trainer, split_input_data)

    context.perf_meter.checkpoint("training")

    split_predicted_data = predict_after_model_training(
        pj,
        model_trainer,
        split_input_data,
    )

    context.perf_meter.checkpoint("predicting")

    new_model_is_better = is_new_model_better(
        pj,
        context,
        model_trainer,
        split_input_data,
        compare_to_old,
    )

    context.perf_meter.checkpoint("evaluation")

    path_to_save = context.config.paths.trained_models
    write_results_new_model(
        pj,
        model_trainer,
        new_model_is_better,
        split_input_data,
        split_predicted_data,
        path_to_save,
    )

    send_report_teams(pj, model_trainer, new_model_is_better)

    context.perf_meter.checkpoint("writing results")


def create_model_trainer(pj, context, retrain_young_models=False):
    """Create model trainer and set hyperparams.

    Args:
        pj (dict): Prediction job
        context (openstf.task.utils.TaskContext): Task context for logging
        retrain_young_models (bool): Retrain models younger then 7 days if True.
            Defaults to False.

    Returns:
        Implementation of the AbstractModelTrainer class tailored to
            a specific algorithm.
    """
    # Make model trainer creator
    mc = ModelTrainerCreator(pj)

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

    return model_trainer


def preprocess_for_model_training(pj, context):
    """Pre-process model data. Clean data and apply features

    Args:
        pj (dict): Prediction job
        context (openstf.task.utils.TaskContext): Task context for logging

    Raises:
        ValueError: when data quality is insufficient

    Returns:
        dict: dict of cleaned data with features, split in train, validation and test
    """
    # Specify training period
    # use hyperparam training_period_days if available
    hyperparams = {"training_period_days": TRAINING_PERIOD_DAYS, "featureset_name": "D"}
    hyperparams.update(context.database.get_hyper_params())

    featureset = context.database.get_featureset(hyperparams["featureset_name"])
    datetime_start = datetime.utcnow() - timedelta(
        days=hyperparams["training_period_days"]
    )
    datetime_end = datetime.utcnow()

    # Get data from database
    data = context.database.get_model_input(
        pid=pj["id"],
        location=[pj["lat"], pj["lon"]],
        datetime_start=datetime_start,
        datetime_end=datetime_end,
    )

    # Pre-process data
    clean_data_with_features = pre_process_data(data, featureset)

    # Check if we have enough data left to continue
    if is_data_sufficient(clean_data_with_features) is False:
        raise ValueError("Input data quality insufficient, aborting!")

    # Split data in train, test and validation sets
    train_data, validation_data, test_data = split_data_train_validation_test(
        clean_data_with_features
    )
    return split_model_data(
        train=train_data,
        validation=validation_data,
        test=test_data,
    )


def train_model(model_trainer, split_input_data):
    """Train model.

    Args:
        model_trainer (): Implementation of the AbstractModelTrainer class tailored to
            a specific algorithm.
        split_input_data (dict): dict of cleaned data with features, split in train,
            validation and test
    """
    # Train model
    model_trainer.train(split_input_data.train_data, split_input_data.validation_data)


def predict_after_model_training(pj, model_trainer, split_input_data):
    """Predict data with new trained model.

    Args:
        pj (dict): Prediction job
        model_trainer (): Implementation of the AbstractModelTrainer class tailored to
            a specific algorithm.
        split_input_data (dict): dict of cleaned data with features, split in train,
            validation and test

    Returns:
        dict: dict of predicted data, split in train, validation and test
    """
    # Build predictor and create predictions
    predictor = PredictionModelCreator.create_prediction_model(
        pj,
        ForecastType.DEMAND,
        model_trainer.trained_model,
        model_trainer.confidence_interval,
    )
    train_predict = predictor.make_forecast(
        split_input_data.train.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )
    validation_predict = predictor.make_forecast(
        split_input_data.validation.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )
    test_predict = predictor.make_forecast(
        split_input_data.test.iloc[:, 1:].drop("Horizon", axis=1, errors="ignore")
    )

    return split_model_data(
        train=train_predict,
        validation=validation_predict,
        test=test_predict,
    )


def is_new_model_better(
    pj, context, model_trainer, split_input_data, compare_to_old=True
):
    """Evaluate quality of new model with old model and store it if it is better.

    Model is saved if better than the old model unless specified
    otherwise.

    Args:
        pj (dict): Prediction job
        context (openstf.task.utils.TaskContext): Task context for logging
        model_trainer (MLModelType): model trainer
        split_input_data (dict): dict of cleaned data with features, split in train,
            validation and test
        compare_to_old (bool, optional): Compare new trained model to old trained model
            if True. Defaults to True.
    """

    # Combine validation + train data
    combined = split_input_data.train.append(split_input_data.validation)

    # Evaluate model and store if better than old model
    # TODO Use predicted data series created above instead of predicting again
    # in better_than_old_model.
    if not compare_to_old or model_trainer.better_than_old_model(combined):
        context.logger.info("New model is better then old model", prediction_job=pj)
        model_trainer.store_model()
        new_model_is_better = True
    else:
        context.logger.warning("Old model is better then new model", prediction_job=pj)
        new_model_is_better = False
    return new_model_is_better


def create_evaluation_figures(model_trainer, split_input_data, split_predicted_data):
    """Create figures of feature importance and data series of input and predicted data.

    Args:
        model_trainer (MLModelType): model trainer
        split_input_data (dict): dict of cleaned data with features, split in train,
            validation and test
        split_predicted_data (dict): dict of predicted data, split in train, validation
            and test

    Returns:
        tuple: tuple containing:
            plotly.graph_objects.Figure: A treemap of the features.
            dict: dict of line plots of of split input and predicted
                data series per prediction horizon.
    """

    # Create figures
    figure_features = plot_feature_importance(model_trainer.feature_importance)
    horizons = split_input_data.train.Horizon.unique()
    input_data_list = [
        split_input_data.train,
        split_input_data.validation,
        split_input_data.test,
    ]
    predicted_data_list = [
        split_predicted_data.train,
        split_predicted_data.validation,
        split_predicted_data.test,
    ]
    figure_series = {
        f"Predictor{horizon}": plot_data_series(
            data=input_data_list,
            predict_data=predicted_data_list,
            horizon=horizon,
        )
        for horizon in horizons
    }
    return figure_features, figure_series


def write_results_new_model(
    pj,
    model_trainer,
    new_model_is_better,
    split_input_data,
    split_predicted_data,
    path_to_save,
):
    """Save results of new model to disk.

    Args:
        pj (dict): Prediction job
        model_trainer (MLModelType): model trainer
        new_model_is_better (boolean): True if new model is better than old model
        split_input_data (dict): dict of cleaned data with features, split in train,
            validation and test
        split_predicted_data (dict): dict of predicted data, split in train, validation
            and test
        path_to_save (pathlib.Path): path to save figures of results of new model
    """
    figure_features, figure_series = create_evaluation_figures(
        model_trainer, split_input_data, split_predicted_data
    )
    if new_model_is_better:
        # Save figures to disk
        save_loc = Path(path_to_save) / str(pj["id"])
    else:
        # Save figures to disk
        save_loc = Path(path_to_save) / str(pj["id"]) / "worse_model"

    os.makedirs(save_loc, exist_ok=True)
    figure_features.write_html(str(save_loc / "weight_plot.html"))
    for key, fig in figure_series.items():
        fig.write_html(str(save_loc / f"{key}.html"), auto_open=False)
        if not new_model_is_better:
            # write for a worse new model the series figures
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


def send_report_teams(pj, model_trainer, new_model_is_better):
    """Send Teams message about result of new model.

    Args:
        pj (dict): Prediction job
        model_trainer (MLModelType): model trainer
        new_model_is_better (boolean): True if new model is better than old model
    """
    if new_model_is_better:
        send_report_teams_better(pj, model_trainer.feature_importance)
    else:
        send_report_teams_worse(pj)


def train_specific_model(context, pid):
    """Train model for given prediction id.

    Tracy-compatible function to train a specific model based on the prediction id (pid)
    Should not be used outside of Tracy, preferred alternative:
        train_model_pipeline

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
    train_model_pipeline(pj, context, compare_to_old=False, retrain_young_models=True)
