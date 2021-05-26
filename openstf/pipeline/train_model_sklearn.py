# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import pandas as pd

import joblib
import logging

from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator
from openstf.model.confidence_interval_generator import ConfidenceIntervalGenerator
from openstf.model.model_creator import ModelCreator
from openstf.metrics.reporter import Reporter
from openstf.model.serializer import PersistentStorageSerializer
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.validation.validation import validate, clean, is_data_sufficient

TRAIN_HORIZONS: list[float] = [0.25, 24.0]
MAXIMUM_MODEL_AGE: int = 7

EARLY_STOPPING_ROUNDS: int = 10
PENALTY_FACTOR_OLD_MODEL: float = 1.2

SAVE_PATH = Path(".")
OLD_MODEL_PATH = "."


def train_model_pipeline(
    pj: dict,
    input_data: pd.DataFrame,
    check_old_model_age: bool = True,
    compare_to_old: bool = True,
) -> None:

    # Get old model and age
    old_model_age = float("inf")  # Default in case old model could not be loaded
    try:
        old_model = PersistentStorageSerializer(pj).load_model()
        old_model_age = old_model.age
    except FileNotFoundError:
        logging.info("No old model found retraining anyway")
        compare_to_old = (
            False  # If we do not have an old model we cannot use is to compare
        )
    # Check old model age and continue yes/no
    if (old_model_age < MAXIMUM_MODEL_AGE) and check_old_model_age:
        logging.info("Model is newer than 7 days!")
        return

    # Get hyper parameters
    hyper_params = pj["hyper_params"]

    # Validate and clean data
    validated_data = clean(validate(input_data))

    # Check if sufficient data is left after cleaning
    if not is_data_sufficient(validated_data):
        raise (
            RuntimeError(
                f"Not enough data left after validation for {pj['name']}, check input data!"
            )
        )

    # Add features
    data_with_features = TrainFeatureApplicator(
        TRAIN_HORIZONS, features=pj["features_set"]
    ).add_features(validated_data)

    # Split data
    train_data, validation_data, test_data = split_data_train_validation_test(
        data_with_features.sort_index(axis=1)
    )

    # Create relevant model
    model = ModelCreator(pj).create_model()

    # Configure evals for early stopping
    eval_set = [
        (train_data.iloc[:, 1:], train_data.iloc[:, 0]),
        (validation_data.iloc[:, 1:], validation_data.iloc[:, 0]),
    ]

    model.set_params(params=hyper_params)
    model.fit(
        train_data.iloc[:, 1:],
        train_data.iloc[:, 0],
        eval_set=eval_set,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=0
    )
    logging.debug('Fitted a new model, not yet stored')

    # Check if new model is better than old model
    if compare_to_old:
        combined = train_data.append(validation_data)

        # Score method always returns R^2
        score_new_model = model.score(combined.iloc[:, 1:], combined.iloc[:, 0])
        score_old_model = old_model.score(combined.iloc[:, 1:], combined.iloc[:, 0])

        # Check if R^2 is better for old model
        if score_old_model > score_new_model * PENALTY_FACTOR_OLD_MODEL:
            raise (RuntimeError(f"Old model is better than new model for {pj['name']}"))
        else:
            logging.info(
                "New model is better than old model, continuing with training procces"
            )

    # Report about the training procces
    Reporter(
        pj, train_data, validation_data, test_data
    ).make_and_save_dashboard_figures(model)

    # Do confidence interval determination
    model = ConfidenceIntervalGenerator(
        pj, validation_data
    ).generate_confidence_interval_data(model)

    # Persist model
    PersistentStorageSerializer(pj).save_model(model)
    logging.info('New model stored')
