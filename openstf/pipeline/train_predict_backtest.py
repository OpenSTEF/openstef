# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Union, List
from pathlib import Path
import logging

import pandas as pd

from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstf.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)
from openstf.model.standard_deviation_generator import StandardDeviationGenerator
from openstf.model.model_creator import ModelCreator
from openstf.metrics.reporter import Reporter
from openstf.model.serializer import PersistentStorageSerializer
from openstf.model_selection.model_selection import split_data_train_validation_test
from openstf.validation import validation
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator

DEFAULT_TRAIN_HORIZONS: List[float] = [0.25, 24.0]
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10


def train_model_and_forecast_back_test(
    pj: dict,
    input_data: pd.DataFrame,
    trained_models_folder: Union[str, Path],
    save_figures_folder: Union[str, Path],
    training_horizons: List[float] = None,
) -> pd.DataFrame:
    """Pipeline for a back test.

        DO NOT USE THIS PIPELINE FOR OPERATIONAL FORECASTS

    Args:
        pj (dict): Prediction job.
        input_data (pd.DataFrame): Input data
        trained_models_folder:
        trained_models_folder (Path): Path where trained models are stored
        save_figures_folder (Path): path were reports (mostly figures) about the training procces are stored
        training_horizons (list): Horizons to train on in hours.
            These horizons are also used to make predictions (one for every horizon)

    Returns:
        forecast (pandas.DataFrame)

    """
    if training_horizons is None:
        training_horizons = DEFAULT_TRAIN_HORIZONS

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
        horizons=training_horizons, feature_names=pj["feature_names"]
    ).add_features(validated_data)

    # Split data
    train_data, validation_data, test_data = split_data_train_validation_test(
        data_with_features.sort_index(axis=0), test_fraction=0.15, back_test=True
    )

    # Create relevant model
    model = ModelCreator.create_model(pj)

    # split x and y data
    train_x, train_y = train_data.iloc[:, 1:-1], train_data.iloc[:, 0]
    validation_x, validation_y = (
        validation_data.iloc[:, 1:-1],
        validation_data.iloc[:, 0],
    )

    # Configure evals for early stopping
    eval_set = [(train_x, train_y), (validation_x, validation_y)]

    model.set_params(**pj["hyper_params"])
    model.fit(
        train_x,
        train_y,
        eval_set=eval_set,
        early_stopping_rounds=DEFAULT_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )
    logging.info("Fitted a new model, not yet stored")

    # Do confidence interval determination
    model = StandardDeviationGenerator(
        pj, validation_data
    ).generate_standard_deviation_data(model)

    # Report about the training procces
    report = Reporter(pj, train_data, validation_data, test_data).generate_report(model)

    # Save model
    serializer = PersistentStorageSerializer(trained_models_folder)
    serializer.save_model(model, pid=pj["id"])

    # Save reports/figures
    report.save_figures(save_path=save_figures_folder)

    # Load most recent model for the given pid
    model = PersistentStorageSerializer(
        trained_models_folder=trained_models_folder
    ).load_model(pid=pj["id"])

    # Predict
    model_forecast = model.predict(test_data.iloc[:, 1:-1])
    forecast = pd.DataFrame(index=test_data.index, data={"forecast": model_forecast})

    # Add confidence
    forecast = ConfidenceIntervalApplicator(
        model, test_data.iloc[:, 1:-1]
    ).add_confidence_interval(forecast, pj)

    # Prepare for output
    forecast = add_prediction_job_properties_to_forecast(
        pj,
        forecast,
    )

    # Add column with realised load and horizon information
    forecast["realised"] = test_data.iloc[:, 0]
    forecast["Horizon"] = test_data.iloc[:, -1]

    return forecast
