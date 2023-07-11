# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model_selection.model_selection import backtest_split_default
from openstef.pipeline import train_model
from openstef.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)

DEFAULT_TRAIN_HORIZONS: list[float] = [0.25, 24.0]
DEFAULT_EARLY_STOPPING_ROUNDS: int = 10


def train_model_and_forecast_back_test(
    pj: PredictionJobDataClass,
    modelspecs: ModelSpecificationDataClass,
    input_data: pd.DataFrame,
    training_horizons: list[float] = None,
    n_folds: int = 1,
) -> tuple[
    pd.DataFrame,
    list[OpenstfRegressor],
    list[pd.DataFrame],
    list[pd.DataFrame],
    list[pd.DataFrame],
]:
    """Pipeline for a back test.

    When number of folds is larger than 1: apply pipeline for a back test when forecasting
    the entire input range.

        - Makes use of kfold cross validation in order to split data multiple times.
        - Results of all the testsets are added together to obtain the forecast for the whole input range.
        - Obtaining the days for each fold can be done either randomly or not
        **DO NOT USE THIS PIPELINE FOR OPERATIONAL FORECASTS**

    Args:
        pj: Prediction job.
        modelspecs: Dataclass containing model specifications
        input_data: Input data
        training_horizons: horizons to train on in hours.
            These horizons are also used to make predictions (one for every horizon)
        n_folds: number of folds to apply (if 1, no cross validation will be applied)

    Returns:
        - Forecast (pandas.DataFrame)
        - Fitted models (list[OpenStfRegressor])
        - Train data sets (list[pd.DataFrame])
        - Validation data sets (list[pd.DataFrame])
        - Test data sets (list[pd.DataFrame])

    """
    if pj.backtest_split_func is None:
        backtest_split_func = backtest_split_default
        backtest_split_args = {"stratification_min_max": pj["model"] != "proloaf"}
    else:
        backtest_split_func, backtest_split_args = pj.backtest_split_func.load(
            required_arguments=["data", "n_folds"]
        )

    data_with_features = train_model.train_pipeline_step_compute_features(
        input_data=input_data, pj=pj, model_specs=modelspecs, horizons=training_horizons
    )

    # The use of zip allows to take advantage of the lazy estimation mechanisms of Python, especially if the
    # backtest_split_func returns a generator. This can avoid unwanted multiple data copies.
    # 1. First we retrieve a generator (use of () comprehensive) on (model, forecast, train, val, test)
    # 2. Then we unzip the result into generators separated by result type (models, forecasts, trains, vals, tests)
    (
        models_folds,
        forecast_folds,
        train_data_folds,
        validation_data_folds,
        test_data_folds,
    ) = zip(
        *(
            train_model_and_forecast_test_core(
                pj,
                modelspecs,
                train_data,
                validation_data,
                test_data,
            )
            + (train_data, validation_data, test_data)
            for train_data, validation_data, test_data, _ in backtest_split_func(
                data_with_features, n_folds, **backtest_split_args
            )
        )
    )

    return (
        pd.concat(forecast_folds, axis=0).sort_index(),
        list(models_folds),
        list(train_data_folds),
        list(validation_data_folds),
        list(test_data_folds),
    )


def train_model_and_forecast_test_core(
    pj: PredictionJobDataClass,
    modelspecs: ModelSpecificationDataClass,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[OpenstfRegressor, pd.DataFrame]:
    """Trains the model and forecast on the test set.

    Args:
        pj: Prediction job.
        modelspecs: Dataclass containing model specifications
        train_data: Train data with computed features
        validation_data: Validation data with computed features
        test_data: Test data with computed features

    Returns:
        - The trained model
        - The forecast on the test set.

    """
    model = train_model.train_pipeline_step_train_model(
        pj, modelspecs, train_data, validation_data
    )

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

    return model, forecast
