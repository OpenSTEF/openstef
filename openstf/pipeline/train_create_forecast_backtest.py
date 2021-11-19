# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Tuple

import random
from itertools import accumulate

from openstf.model.regressors.regressor import OpenstfRegressor

import pandas as pd
from openstf_dbc.services.prediction_job import PredictionJobDataClass

from openstf.data_classes.model_specifications import ModelSpecificationDataClass
from openstf.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstf.pipeline.train_model import train_pipeline_common
from openstf.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)
from openstf.feature_engineering.feature_applicator import TrainFeatureApplicator

DEFAULT_TRAIN_HORIZONS: List[float] = [0.25, 24.0]
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10


def train_model_and_forecast_back_test(
    pj: PredictionJobDataClass,
    modelspecs: ModelSpecificationDataClass,
    input_data: pd.DataFrame,
    training_horizons: List[float] = None,
    n_folds: int = 1,
) -> Tuple[
    pd.DataFrame,
    List[OpenstfRegressor],
    List[pd.DataFrame],
    List[pd.DataFrame],
    List[pd.DataFrame],
]:
    """Pipeline for a back test.

        When number of folds is larger than 1: apply pipeline for a back test when forecasting the entire input range.
        - Makes use of kfold cross validation in order to split data multiple times.
        - Results of all the testsets are added together to obtain the forecast for the whole input range.
        - Obtaining the days for each fold can be done either randomly or not

        DO NOT USE THIS PIPELINE FOR OPERATIONAL FORECASTS

    Args:
        pj (PredictionJobDataClass): Prediction job.
        modelspecs (ModelSpecificationDataClass): Dataclass containing model specifications
        input_data (pd.DataFrame): Input data
        training_horizons (list): horizons to train on in hours.
            These horizons are also used to make predictions (one for every horizon)
        n_folds (int): number of folds to apply (if 1, no cross validation will be applied)

    Returns:
        forecast (pandas.DataFrame)

    """
    if training_horizons is None:
        training_horizons = DEFAULT_TRAIN_HORIZONS

    apply_folds = True if n_folds > 1 else False

    models_entire = []
    valid_data_entire = []
    train_data_entire = []
    test_data_entire = []

    if apply_folds:
        # prepare data in order to apply nfolds
        input_data.index = pd.to_datetime(input_data.index)
        input_data["dates"] = input_data.index.date
        input_data["random_fold"] = None

        # divide each day in a fold
        input_data = group_kfold(input_data, n_folds)

        # iterate of the folds, train and forecast for each fold
        for i in range(n_folds):
            test_idx = input_data[
                input_data.index.isin(input_data[input_data.random_fold == i].index)
            ].index.sort_values()
            train_idx = input_data[
                input_data.index.isin(input_data[input_data.random_fold != i].index)
            ].index.sort_values()
            test_df = input_data.loc[test_idx, :]
            train_df = input_data.loc[train_idx, :]

            test_df = TrainFeatureApplicator(
                horizons=training_horizons, feature_names=modelspecs.feature_names
            ).add_features(
                test_df.iloc[:, :-2]
            )  # ignore the added columns (dates, random_fold)

            (
                forecast_i,
                model,
                train_data,
                validation_data,
                test_data,
            ) = train_model_and_forecast_test_core(
                pj,
                modelspecs,
                train_df.iloc[:, :-2],  # ignore the added columns (dates, random_fold)
                test_df,
                training_horizons,
                apply_folds,
                test_fraction=0.0,
            )

            models_entire.append(model)
            valid_data_entire.append(validation_data)
            train_data_entire.append(train_data)
            test_data_entire.append(test_data)

            if i > 0:
                forecast = forecast.append(forecast_i)
            else:
                forecast = forecast_i
    else:
        (
            forecast,
            model,
            train_data,
            validation_data,
            test_data,
        ) = train_model_and_forecast_test_core(
            pj,
            modelspecs,
            input_data,
            training_horizons=training_horizons,
            apply_folds=apply_folds,
        )

        models_entire.append(model)
        valid_data_entire.append(validation_data)
        train_data_entire.append(train_data)
        test_data_entire.append(test_data)

    return (
        forecast,
        models_entire,
        train_data_entire,
        valid_data_entire,
        test_data_entire,
    )


def train_model_and_forecast_test_core(
    pj: PredictionJobDataClass,
    modelspecs: ModelSpecificationDataClass,
    input_data: pd.DataFrame,
    test_data: pd.DataFrame = None,
    training_horizons: List[float] = None,
    apply_folds: bool = False,
    test_fraction: float = 0.15,
) -> Tuple[pd.DataFrame, OpenstfRegressor]:
    """Core part of the backtest pipeline, in order to create a model and forecast from input data

    Args:
        pj (PredictionJobDataClass): Prediction job.
        modelspecs (ModelSpecificationDataClass): Dataclass containing model specifications
        input_data (pd.DataFrame): Input data
        training_horizons (list): horizons to train on in hours.
            These horizons are also used to make predictions (one for every horizon)

    Returns:
        forecast (pandas.DataFrame)

    """
    model, report, train_data, validation_data, test_data_ = train_pipeline_common(
        pj,
        modelspecs,
        input_data,
        training_horizons,
        test_fraction=test_fraction,
        backtest=True,
    )

    if not apply_folds:
        test_data = test_data_

    # Predict
    model_forecast = model.predict(test_data.iloc[:, 1:-1])
    forecast = pd.DataFrame(index=test_data.index, data={"forecast": model_forecast})

    # Define tAhead to something meaningfull in the context of a backtest
    forecast["tAhead"] = test_data.iloc[:, -1]

    # Add confidence
    forecast = ConfidenceIntervalApplicator(
        model, test_data.iloc[:, 1:-1]
    ).add_confidence_interval(forecast, pj)

    # Prepare for output
    forecast = add_prediction_job_properties_to_forecast(
        pj, forecast, algorithm_type="backtest"
    )

    # Add column with realised load and horizon information
    forecast["realised"] = test_data.iloc[:, 0]
    forecast["horizon"] = test_data.iloc[:, -1]

    return forecast, model, train_data, validation_data, test_data


def group_kfold(
    input_data: pd.DataFrame, n_folds: int, random_split: bool = True
) -> pd.DataFrame:
    """Function to group data into groups, according to the date and the number of folds
        - each date gets assigned a number between 0 and n_folds

    Args:
        input_data (pd.DataFrame): Input data
        n_folds (int): Number of folds
        random_split (bool): Indicates if random split needs to be applied

    Returns:
        grouped data (pandas.DataFrame)

    """
    unique_dates = input_data["dates"].unique()
    # Group separators
    len_data = len(unique_dates)
    size = len_data // n_folds
    rem = len_data % n_folds
    separators = list(accumulate([0] + [size + 1] * rem + [size] * (n_folds - rem)))

    items = list(unique_dates)

    if random_split:
        random.shuffle(items)

    for i, s in enumerate(zip(separators, separators[1:])):
        group = items[slice(*s)]
        input_data.loc[
            input_data[input_data["dates"].isin(group)].index, "random_fold"
        ] = i
    return input_data
