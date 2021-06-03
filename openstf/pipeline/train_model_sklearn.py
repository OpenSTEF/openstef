# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Union

import pandas as pd
from sklearn.base import RegressorMixin
import structlog
# from ktpbase.config.config import ConfigManager

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.confidence_interval_generator import ConfidenceIntervalGenerator
from openstf.model.model_creator import ModelCreator
from openstf.metrics.reporter import Reporter, Report

from openstf.model.serializer import PersistentStorageSerializer
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.validation import validation

DEFAULT_TRAIN_HORIZONS: List[float] = [0.25, 24.0]
MAXIMUM_MODEL_AGE: int = 7

DEFAULT_EARLY_STOPPING_ROUNDS: int = 10
PENALTY_FACTOR_OLD_MODEL: float = 1.2

SAVE_PATH = Path(".")


# def train_model_pipeline(
#     pj: dict,
#     input_data: pd.DataFrame,
#     check_old_model_age: bool,
#     trained_models_folder: Optional[Union[str, Path]],
#     save_figures_folder: Optional[Union[str, Path]]
# ):
#     config = ConfigManager.get_instance()

#     if trained_models_folder is None:
#         trained_models_folder = Path(config.paths.trained_models_folder)

#     if save_figures_folder is None:
#         save_figures_folder = Path(config.paths.webroot) / pj["id"]

#     logger = structlog.get_logger(__name__)
#     serializer = PersistentStorageSerializer(trained_models_folder)

#     # Get old model and age
#     try:
#         old_model = serializer.load_model(pid=pj["id"])
#         old_model_age = old_model.age
#     except FileNotFoundError:
#         old_model = None
#         old_model_age = float("inf")
#         logger.warning("No old model found, train new model")

#     # Check old model age and continue yes/no
#     if (old_model_age < MAXIMUM_MODEL_AGE) and check_old_model_age:
#         logger.warning(
#             "Old model is younger than {MAXIMUM_MODEL_AGE} days, skip training"
#         )
#         return

#     # train model
#     try:
#         model, report = train_model_pipeline_core(pj, input_data, old_model)
#     except RuntimeError as e:
#         return

#     # save model
#     serializer.save_model(model, pid=pj["id"])
#     # save figures
#     report.save_figures(save_path=save_figures_folder)


def train_model_pipeline_core(
    pj: dict,
    input_data: pd.DataFrame,
    old_model: RegressorMixin = None,
    horizons: List[float] = DEFAULT_TRAIN_HORIZONS,
) -> Tuple[RegressorMixin, Report]:
    """Run the train model pipeline.

        TODO once we have a data model for a prediction job this explantion is not
        required anymore.

        For training a model the following keys in the prediction job dictionairy are
        expected:
            "name"          Arbitray name only used for logging
            "model"         Model type, any of "xgb", "lgb",
            "hyper_params"  Hyper parameters dictionairy specific to the model_type
            "feature_names"      List of features to train model on or None to use all features

    Args:
        pj (dict): Prediction job
        input_data (pd.DataFrame): Input data
        old_model (RegressorMixin, optional): Old model to compare to. Defaults to None.
        horizons (List[float]): Horizons to train on in hours.

    Raises:
        ValueError: When input data is insufficient
        RuntimeError: When old model is better than new model

    Returns:
        Tuple[RegressorMixin, Report]: Trained model and report (with figures)
    """

    logger = structlog.get_logger(__name__)
    # Validate and clean data
    validated_data = validation.clean(validation.validate(input_data))

    # Check if sufficient data is left after cleaning
    if not validation.is_data_sufficient(validated_data):
        raise ValueError(
            f"Input data is insufficient for {pj['name']} "
            f"after validation and cleaning"
        )

    # Add features
    data_with_features = TrainFeatureApplicator(
        horizons=horizons, feature_names=pj["feature_names"]
    ).add_features(validated_data)

    # Split data
    train_data, validation_data, test_data = split_data_train_validation_test(
        data_with_features.sort_index(axis=1)
    )

    # Create relevant model
    model = ModelCreator.create_model(model_type=pj["model"])

    # split x and y data
    train_x, train_y = train_data.iloc[:, 1:], train_data.iloc[:, 0]
    validation_x, validation_y = validation_data.iloc[:, 1:], validation_data.iloc[:, 0]

    # Configure evals for early stopping
    eval_set = [(train_x, train_y), (validation_x, validation_y)]

    model.set_params(params=pj["hyper_params"])
    model.fit(
        train_x,
        train_y,
        eval_set=eval_set,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=0,
    )
    logging.info("Fitted a new model, not yet stored")

    # Check if new model is better than old model
    # NOTE it would be better to move this code out of the pipeline
    # training a model should probably not be responsible for this
    if old_model is not None:
        combined = train_data.append(validation_data)
        x_data, y_data = combined.iloc[:, 1:], combined.iloc[:, 0]

        # Score method always returns R^2
        score_new_model = model.score(x_data, y_data)
        score_old_model = old_model.score(x_data, y_data)

        # Check if R^2 is better for old model
        if score_old_model > score_new_model * PENALTY_FACTOR_OLD_MODEL:
            raise (RuntimeError(f"Old model is better than new model for {pj['name']}"))
        else:
            logging.info(
                "New model is better than old model, continuing with training procces"
            )

        logger.info(
            "New model is better than old model, continuing with training procces"
        )

    # Do confidence interval determination
    model = ConfidenceIntervalGenerator(
        pj, validation_data
    ).generate_confidence_interval_data(model)

    # Report about the training procces
    report = Reporter(pj, train_data, validation_data, test_data).generate_report(model)

    return model, report
