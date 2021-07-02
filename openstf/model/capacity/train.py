# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
import pandas as pd
from openstf_dbc.database import DataBase
from openstf_dbc.config.config import ConfigManager
import structlog

from openstf.feature_engineering.capacity_prognoses_features import (
    apply_capacity_features,
)
from openstf.model.capacity.model import CapacityPredictionModel
from openstf.model.capacity.utils import prepare_training_data


def train_capacity_prognosis(pj, datetime_start, datetime_end, y_hor=[0, 6, 13]):
    """Train a capacity prognoses model for a specific prediction job.

    Args:
        pj: (dict) prediction job
    """
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
    logger.info("Apply features")
    feature_data, classes = apply_capacity_features(
        input_data,
        y_col="load_max",
        y_hor=y_hor,
        outlier_removal=False,
        load_profile_names=load_profile_names,
    )

    # prepare data
    logger.info("Prepare prediction data")
    train_x, train_y, train_h, val_x, val_y, val_h = prepare_training_data(
        feature_data, y_col="load_max"
    )

    # create model
    logger.info("Create capacity prognosis model")
    model = CapacityPredictionModel(classes=classes)

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

    # FIXME use serialiazer for loading and saving
    trained_models_folder = ConfigManager.get_instance().paths.trained_models
    directory = Path(trained_models_folder) / f'{pj["id"]}' / "capacity"
    model.save(directory=directory)
