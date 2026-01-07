# ruff: noqa
# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
from openstef_models.models.forecasting.forecaster import ForecastDataset
from openstef_models.utils.data_split import DataSplitter
import pandas as pd
import pytest
from openstef_models.models.forecasting.median_forecaster import MedianForecaster, MedianForecasterConfig
from openstef_core.datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Q
from openstef_core.testing import create_timeseries_dataset
from openstef_models.presets import ForecastingWorkflowConfig, create_forecasting_workflow


def test_median_returns_median():
    # Arrange     
    index = pd.date_range("2020-01-01T00:00", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0,4.0,7.0],
        load_lag_PT1H=[1.0,np.nan, np.nan],
        load_lag_PT2H=[4.0,1.0, np.nan],
        load_lag_PT3H=[7.0,4.0,1.0],
        available_at=index,
    )   

    training_input_data = ForecastInputDataset.from_timeseries(
            dataset=training_data,
            target_column="load",
            forecast_start=index[0],
        )

    expected_result = ForecastDataset(
            data=pd.DataFrame(
                {
                    "quantile_P50": [4.0, 4.0, 4.0],
                },
                index=index,
            ),
            sample_interval=training_input_data.sample_interval,
        )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT36H")])
    model = MedianForecaster(config=config)


    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)
    
    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)


def test_median_handles_some_missing_data():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1H=[1.0, np.nan, np.nan],
        load_lag_PT2H=[np.nan, 1.0, np.nan],
        load_lag_PT3H=[3.0, np.nan, 1.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [2.0, 1.5, 1.5],
            },
            index=index,
        ),
        sample_interval=training_input_data.sample_interval,
    )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_handles_missing_data_for_some_horizons():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[5.0, 5.0, 5.0],
        load_lag_PT1H=[5.0, np.nan, np.nan],
        load_lag_PT2H=[np.nan, 5.0, np.nan],
        load_lag_PT3H=[np.nan, np.nan, 5.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [5.0, 5.0, 5.0],
            },
            index=index,
        ),
        sample_interval=training_input_data.sample_interval,
    )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_handles_all_missing_data():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[np.nan, np.nan, np.nan],
        load_lag_PT1H=[np.nan, np.nan, np.nan],
        load_lag_PT2H=[np.nan, np.nan, np.nan],
        load_lag_PT3H=[np.nan, np.nan, np.nan],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [],
            },
            index=pd.DatetimeIndex([], freq="h"),
        ),
        sample_interval=training_input_data.sample_interval,
        forecast_start=training_input_data.forecast_start,
    )


    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_uses_lag_features_if_available():
    # Arrange
    index = pd.date_range("2023-01-01T00:00", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[4.0, 5.0, 6.0],
        load_lag_PT1H=[1.0, 2.0, 3.0],
        load_lag_PT2H=[4.0, 5.0, 6.0],
        load_lag_PT3H=[7.0, 8.0, 9.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [4.0, 5.0, 6.0],
            },
            index=index,
        ),
        sample_interval=training_input_data.sample_interval,
    )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_handles_small_gap():
    # Arrange
    index = pd.date_range("2023-01-01T00:00", periods=5, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[5.0, 4.0,3.0,2.0,1.0,],
        load_lag_PT1H=[1.0,np.nan, np.nan, np.nan, np.nan],
        load_lag_PT2H=[2.0,1.0, np.nan, np.nan, np.nan],
        load_lag_PT3H=[3.0,2.0,1.0, np.nan, np.nan],
        load_lag_PT4H=[4.0,3.0,2.0,1.0, np.nan],
        load_lag_PT5H=[5.0,4.0,3.0,2.0,1.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    # Remove the second row to create a small gap
    training_input_data.data = training_input_data.data[training_input_data.data.index != "2023-01-01T01:00"]


    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [3.0,  3.0, 3.0, 3.0],
            },
            index=pd.date_range("2023-01-01T00:00", periods=5, freq="h").delete(1),
        ),
        sample_interval=training_input_data.sample_interval,
    )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_handles_large_gap():
    # Arrange
    index_1 = pd.date_range("2023-01-01T00:00", periods=3, freq="h")
    training_data_1 = create_timeseries_dataset(
        index=index_1,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1H=[4.0, 5.0, 6.0],
        load_lag_PT2H=[7.0, 8.0, 9.0],
        available_at=index_1,
    )
    
    index_2 = pd.date_range("2023-01-02T01:00", periods=3, freq="h")
    training_data_2 = create_timeseries_dataset(
        index=index_2,
        load=[10.0, 11.0, 12.0],
        load_lag_PT1H=[13.0, 14.0, 15.0],
        load_lag_PT2H=[16.0, 17.0, 18.0],
        available_at=index_2,
    )
    
    training_data = training_data_1

    training_data.data = pd.concat([training_data_1.data, training_data_2.data])


    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index_1[0],
    )

    expected_result = ForecastDataset(
        data=pd.DataFrame(
            {
                "quantile_P50": [5.5, 6.5, 7.5, 14.5, 15.5, 16.5],
            },
            index=pd.DatetimeIndex(pd.concat([index_1.to_series(), index_2.to_series()])),
        ),
        sample_interval=training_input_data.sample_interval,
    )
    expected_result.index.freq = None

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act
    model.fit(training_input_data)
    result = model.predict(training_input_data)

    # Assert
    assert result.sample_interval == expected_result.sample_interval
    pd.testing.assert_frame_equal(result.data, expected_result.data)

def test_median_fit_with_missing_features_raises():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1H=[1.0, 2.0, 3.0],
        load_lag_PT2H=[4.0, 5.0, 6.0],
        load_lag_PT3H=[7.0, 8.0, 9.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3H")])
    model = MedianForecaster(config=config)
    model.fit(training_input_data)

    # Create prediction data missing one lag feature
    prediction_data = training_data.data.copy()
    prediction_data = prediction_data.drop(columns=["load_lag_PT3H"])
    prediction_input_data = ForecastInputDataset(
        data=prediction_data,
        target_column="load",
        forecast_start=index[0],
        sample_interval=training_input_data.sample_interval,
    )

    # Act & Assert
    with pytest.raises(ValueError, match="The input data is missing the following lag features"):
        model.predict(prediction_input_data)

def test_median_fit_with_no_lag_features_raises():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        unrelated_feature=[1.0, 2.0, 3.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3H")])
    model = MedianForecaster(config=config)

    # Act & Assert
    with pytest.raises(ValueError, match="No lag features found in the input data."):
        model.fit(training_input_data)

def test_median_fit_with_inconsistent_lag_features_raises():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="h")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1H=[1.0, 2.0, 3.0],
        load_lag_PT5H=[4.0, 5.0, 6.0],
        load_lag_PT60H=[7.0, 8.0, 9.0],
        load_lag_P4D=[10.0, 11.0, 12.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3H")])
    model = MedianForecaster(config=config)

    # Act & Assert
    with pytest.raises(ValueError, match="Lag features are not evenly spaced"):
        model.fit(training_input_data)

def test_median_fit_with_inconsistent_frequency_raises():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="min")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1H=[1.0, 2.0, 3.0],
        load_lag_PT2H=[4.0, 5.0, 6.0],
        load_lag_PT3H=[7.0, 8.0, 9.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3H")])
    model = MedianForecaster(config=config)

    # Act & Assert
    with pytest.raises(ValueError, match="does not match the model frequency."):
        model.fit(training_input_data)

def test_predicting_without_fitting_raises():
    # Arrange
    index = pd.date_range("2023-01-01", periods=3, freq="min")
    training_data = create_timeseries_dataset(
        index=index,
        load=[1.0, 2.0, 3.0],
        load_lag_PT1M=[1.0, 2.0, 3.0],
        load_lag_PT2M=[4.0, 5.0, 6.0],
        load_lag_PT3M=[7.0, 8.0, 9.0],
        available_at=index,
    )

    training_input_data = ForecastInputDataset.from_timeseries(
        dataset=training_data,
        target_column="load",
        forecast_start=index[0],
    )

    config = MedianForecasterConfig(quantiles=[Q(0.5)], horizons=[LeadTime.from_string("PT3M")])
    model = MedianForecaster(config=config)

    # Act & Assert
    with pytest.raises(AttributeError, match="This MedianForecaster instance is not fitted yet"):
        model.predict(training_input_data)
