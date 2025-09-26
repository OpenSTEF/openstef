# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Example demonstrating how to use the MedianForecaster.

This example shows how to set up and use the MedianForecaster, an autoregressive
forecasting model that computes the median of lag features to predict future values.
The model is particularly useful for signals with slow dynamics or noise.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import LeadTime, Quantile
from openstef_models.models.forecasting.median_forecaster import MedianForecaster, MedianForecasterConfig


def create_sample_data_with_lags() -> ForecastInputDataset:
    """Create sample energy load data with lag features.
    
    This simulates a typical scenario where you have historical load data
    with engineered lag features representing previous time steps.
    
    Returns:
        ForecastInputDataset with load values and lag features.
    """
    # Create a time series with some realistic-looking energy load data
    timestamps = pd.date_range(
        start="2025-01-01T00:00:00",
        periods=48,  # 2 days of hourly data
        freq="1h",
    )
    
    # Simulate load data with daily patterns and some noise
    base_load = 1000  # MW base load
    daily_pattern = 200 * np.sin(2 * np.pi * np.arange(48) / 24)  # Daily cycle
    noise = np.random.normal(0, 50, 48)  # Random noise
    load_values = base_load + daily_pattern + noise
    
    # Create lag features representing historical load values
    # T-60min means the load from 60 minutes ago, T-120min from 120 minutes ago, etc.
    data = pd.DataFrame({
        "load": load_values,
        "T-60min": np.concatenate([np.full(1, np.nan), load_values[:-1]]),  # 1 hour lag
        "T-120min": np.concatenate([np.full(2, np.nan), load_values[:-2]]),  # 2 hour lag
        "T-180min": np.concatenate([np.full(3, np.nan), load_values[:-3]]),  # 3 hour lag
        "T-240min": np.concatenate([np.full(4, np.nan), load_values[:-4]]),  # 4 hour lag
        "temperature": 15 + 10 * np.sin(2 * np.pi * np.arange(48) / 24),  # Temperature feature
    }, index=timestamps)
    
    return ForecastInputDataset(
        data=data,
        sample_interval=timedelta(hours=1),
        target_column="load",
        forecast_start=datetime.fromisoformat("2025-01-01T12:00:00"),  # Start forecasting from noon
    )


def main():
    """Demonstrate MedianForecaster usage."""
    print("MedianForecaster Example")
    print("=" * 50)
    
    # Step 1: Create configuration
    print("\n1. Creating MedianForecaster configuration...")
    config = MedianForecasterConfig(
        quantiles=[Quantile(0.5)],  # Only median predictions supported
        horizons=[LeadTime(timedelta(hours=6))],  # 6-hour ahead forecasts
    )
    print(f"   Quantiles: {[float(q) for q in config.quantiles]}")
    print(f"   Horizons: {[str(h) for h in config.horizons]}")
    
    # Step 2: Create forecaster instance
    print("\n2. Creating MedianForecaster instance...")
    forecaster = MedianForecaster(config)
    print(f"   Forecaster type: {type(forecaster).__name__}")
    print(f"   Is fitted: {forecaster.is_fitted}")
    
    # Step 3: Prepare data
    print("\n3. Preparing sample data with lag features...")
    data = create_sample_data_with_lags()
    print(f"   Data shape: {data.data.shape}")
    print(f"   Time range: {data.index[0]} to {data.index[-1]}")
    print(f"   Forecast start: {data.forecast_start}")
    print(f"   Available features: {list(data.data.columns)}")
    
    # Show some sample data
    print("\n   Sample of data with lag features:")
    lag_columns = [col for col in data.data.columns if col.startswith("T-")]
    sample_data = data.data[["load"] + lag_columns].head(8)
    print(sample_data.to_string())
    
    # Step 4: Fit the model
    print("\n4. Fitting the MedianForecaster...")
    forecaster.fit(data)
    print(f"   Is fitted: {forecaster.is_fitted}")
    print(f"   Detected lag features: {forecaster._feature_names}")
    print(f"   Data frequency: {forecaster._frequency_minutes} minutes")
    
    # Step 5: Make predictions
    print("\n5. Making predictions...")
    predictions = forecaster.predict(data)
    print(f"   Predictions shape: {predictions.data.shape}")
    print(f"   Prediction columns: {list(predictions.data.columns)}")
    print(f"   Sample interval: {predictions.sample_interval}")
    
    # Show prediction results
    print("\n   Prediction results:")
    print(predictions.data.head(10).to_string())
    
    # Step 6: Analyze predictions
    print("\n6. Analyzing predictions...")
    median_predictions = predictions.data["quantile_P50"]
    print(f"   Number of predictions: {len(median_predictions)}")
    print(f"   Mean predicted load: {median_predictions.mean():.2f} MW")
    print(f"   Min predicted load: {median_predictions.min():.2f} MW")
    print(f"   Max predicted load: {median_predictions.max():.2f} MW")
    print(f"   Std predicted load: {median_predictions.std():.2f} MW")
    
    # Step 7: Demonstrate state serialization
    print("\n7. Demonstrating state serialization...")
    state = forecaster.to_state()
    print(f"   State keys: {list(state.keys())}")
    
    # Create new forecaster from state
    new_forecaster = MedianForecaster()
    restored_forecaster = new_forecaster.from_state(state)
    print(f"   Restored forecaster is fitted: {restored_forecaster.is_fitted}")
    
    # Verify identical predictions
    restored_predictions = restored_forecaster.predict(data)
    predictions_match = np.allclose(
        predictions.data["quantile_P50"].values,
        restored_predictions.data["quantile_P50"].values,
        equal_nan=True
    )
    print(f"   Predictions match after restoration: {predictions_match}")
    
    print("\n" + "=" * 50)
    print("MedianForecaster example completed successfully!")
    
    # Additional tips
    print("\nTips for using MedianForecaster:")
    print("• Use evenly spaced lag features (T-60min, T-120min, T-180min, etc.)")
    print("• Suitable for signals with slow dynamics or stable states")
    print("• Handles missing data (NaN) gracefully")
    print("• Autoregressive: predictions influence future predictions")
    print("• No hyperparameters to tune - just fit and predict")
    print("• Works best with small training datasets")


if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)
    main()