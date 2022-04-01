# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Tuple

import pandas as pd
from openstef.data_classes.prediction_job import PredictionJobDataClass

from openstef.data_classes.model_specifications import ModelSpecificationDataClass
from openstef.model.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstef.model.regressors.regressor import OpenstfRegressor
from openstef.model_selection.model_selection import group_kfold
from openstef.pipeline import train_model
from openstef.postprocessing.postprocessing import (
    add_prediction_job_properties_to_forecast,
)

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

    if pj.backtest_split_func is None:
        backtest_split_func = default_backtest_split
        backtest_split_args = {}
    else:
        backtest_split_func, backtest_split_args = pj.backtest_split_func.load()

    data_with_features = train_model.train_pipeline_compute_features(
        input_data=input_data, pj=pj, modelspecs=modelspecs, horizons=training_horizons
    )

    (
        models_folds,
        forecast_folds,
        train_data_folds,
        validation_data_folds,
        test_data_folds,
    ) = zip(
        *(
            train_model_and_forecast_test_core(
                pj, modelspecs, train_data, validation_data, test_data
            )
            + (train_data, validation_data, test_data)
            for train_data, validation_data, test_data in backtest_split_func(
                data_with_features, n_folds, pj, **backtest_split_args
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
    pj, modelspecs, train_data, validation_data, test_data
):
    model = train_model.train_pipeline_train_model(
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


def default_backtest_split(input_data, n_folds, pj, test_fraction=0.15):
    if n_folds > 1:
        input_data.index = pd.to_datetime(input_data.index)
        input_data["dates"] = input_data.index
        input_data = group_kfold(input_data, n_folds)

        for ifold in range(n_folds):
            test_data = input_data[input_data["random_fold"] == ifold].sort_index()

            (
                train_data,
                validation_data,
                test_data,
            ) = train_model.train_data_split_default(
                input_data.iloc[:, :-2],
                pj,
                test_fraction=0,
                backtest=True,
                test_data_predefined=test_data,
            )

            yield train_data, validation_data, test_data
    else:
        yield train_model.train_data_split_default(
            input_data, pj, backtest=True, test_fraction=test_fraction
        )
