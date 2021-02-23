# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import plotly
from ktpbase.database import DataBase
from ktpbase.log import logging

from stf.feature_engineering.capacity_prognoses_features import (
    apply_capacity_features
)
from stf.model.capacity_prognosis_model import CapacityPrognosisModel
from stf.model.capacity_prognosis_utils import (
    prepare_prediction_data, prepare_training_data, visualize_predictions
)
from stf.model.serializer.xgboost.xgboost import XGBModelSerializer


def predict_capacity_prognosis(
        pj, datetime_start, datetime_end, y_hor=None):
    """Predict capacity prognoses for specific prediction job

    Args:
        pj: (dict) prediction job
        datetime_start: (datetime.datetime) start date
        datetime_end: (datetime.datetime) end date
        y_hor: (list): prediction horizons
    """
    if y_hor is None:
        y_hor = list(range(13))
    db = DataBase()

    # initialize logging
    logger = logging.get_logger(__name__)

    # get input data
    logger.info("Get input data")
    load_data = db.get_load_pid(
        pid=pj["id"], datetime_start=datetime_start, datetime_end=datetime_end
    )
    sjv_data = db.get_sjv(datetime_start, datetime_end)
    input_data = pd.concat([load_data, sjv_data], axis=1)

    # apply features
    logger.info("Apply features")
    feature_data, _ = apply_capacity_features(
        input_data, y_col="load_max", y_hor=y_hor
    )

    # prepare prediction points
    logger.info("Prepare prediction data")
    x = prepare_prediction_data(feature_data, y_col="load_max", y_hor=y_hor)

    # create model
    logger.info("Create capacity prognosis model")
    model = CapacityPrognosisModel()

    # load model
    serializer = XGBModelSerializer()
    model_location = serializer.trained_models_folder
    directory = model_location / str(pj["id"]) / "capacity"
    logger.info("Loading model", model_directory=directory)
    model.load(directory=directory)

    # predict
    logger.info("Make prediction")
    y_pred, y_pred_prob = model.predict(x)

    # visualize predictions
    logger.info("Visualize prediction")
    fig = visualize_predictions(df=y_pred_prob, classes=model.classes)

    # save figure
    plotly.offline.plot(
        fig,
        filename=str(directory / "capacity_prognoses.html"),
        show_link=False,
        auto_open=False,
    )
    logger.info("Sucessfully made capacity prediction")


def train_capacity_prognosis(pj, datetime_start, datetime_end, y_hor=[0, 6, 13]):
    """ Train a capacity prognoses model for a specific prediction job.

    Args:
        pj: (dict) prediction job
    """
    db = DataBase()

    # initialize logging
    logger = logging.get_logger(__name__)

    # get input data
    logger.info("Get input data")
    load_data = db.get_load_pid(
        pid=pj["id"], datetime_start=datetime_start, datetime_end=datetime_end
    )
    sjv_data = db.get_sjv(datetime_start, datetime_end)
    input_data = pd.concat([load_data, sjv_data], axis=1)

    # apply features
    logger.info("Apply features")
    feature_data, classes = apply_capacity_features(
        input_data, y_col="load_max", y_hor=y_hor, outlier_removal=False
    )

    # prepare data
    logger.info("Prepare prediction data")
    train_x, train_y, train_h, val_x, val_y, val_h = prepare_training_data(
        feature_data, y_col="load_max"
    )

    # create model
    logger.info("Create capacity prognosis model")
    model = CapacityPrognosisModel(classes=classes)

    # train model
    logger.info("Train model")
    model.train(train_x, train_y, val_x, val_y)

    # evaluate model performance
    logger.info("Evaluate model")
    scores = {
        "train": model.evaluate(train_x, train_y),
        "validation": model.evaluate(val_x, val_y),
    }  # Gives code smell but is necessary for unit tests
    # TODO: make train report?
    logger.info("Training metrics", scores=scores)

    # save model
    logger.info("Saving model")
    serializer = XGBModelSerializer()
    model_location = str(serializer.trained_models_folder)
    directory = model_location + "/" + str(pj["id"]) + "/capacity/"
    model.save(directory=directory)
