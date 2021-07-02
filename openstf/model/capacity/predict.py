# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
import pandas as pd
import plotly
from openstf_dbc.database import DataBase
import structlog
from openstf_dbc.config.config import ConfigManager

from openstf.feature_engineering.capacity_prognoses_features import (
    apply_capacity_features,
)
from openstf.model.capacity.model import CapacityPredictionModel
from openstf.model.capacity.utils import prepare_prediction_data, visualize_predictions


def predict_capacity_prognosis(pj, datetime_start, datetime_end, y_hor=None):
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
    logger = structlog.get_logger(__name__)

    # get input data
    logger.info("Get input data")
    load_data = db.get_load_pid(
        pid=pj["id"], datetime_start=datetime_start, datetime_end=datetime_end
    )
    load_profiles_data = db.get_tdcv_load_profiles(datetime_start, datetime_end)
    load_profile_names = list(load_profiles_data.columns)
    input_data = pd.concat([load_data, load_profiles_data], axis=1)

    # apply features
    logger.info("Apply capacity features")
    feature_data, _ = apply_capacity_features(
        input_data, y_col="load_max", y_hor=y_hor, load_profile_names=load_profile_names
    )

    # prepare prediction points
    logger.info("Prepare prediction data")
    x = prepare_prediction_data(feature_data, y_col="load_max", y_hor=y_hor)

    # create model
    logger.info("Create capacity prognosis model")
    model = CapacityPredictionModel()

    # load model
    # FIXME use serialiazer for loading and saving
    trained_models_folder = ConfigManager.get_instance().paths.trained_models
    directory = Path(trained_models_folder) / str(pj["id"]) / "capacity"
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
