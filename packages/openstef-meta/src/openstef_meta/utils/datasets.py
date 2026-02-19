# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Dataset utility functions for ensemble forecasting."""

import pandas as pd

from openstef_core.datasets import ForecastInputDataset


def combine_forecast_input_datasets(
    input_data: ForecastInputDataset, additional_features: ForecastInputDataset | None, join: str = "inner"
) -> ForecastInputDataset:
    """Combine base forecaster predictions with optional additional features.

    Args:
        input_data: ForecastInputDataset containing base forecaster predictions.
        additional_features: Optional ForecastInputDataset containing additional features to combine.
        join: Type of join to perform on the datasets. Defaults to "inner".

    Returns:
        Combined ForecastInputDataset containing both input data and additional features.
    """
    if not isinstance(additional_features, ForecastInputDataset):
        return input_data
    if join != "inner":
        raise NotImplementedError("Only 'inner' join is currently supported.")
    df_additional = additional_features.data
    if input_data.target_column in df_additional.columns:
        df_additional = df_additional.drop(columns=[input_data.target_column])

    df_input = input_data.data
    df = pd.concat(
        [df_input, df_additional],
        axis=1,
        join="inner",
    )

    return ForecastInputDataset(
        data=df,
        sample_interval=input_data.sample_interval,
        target_column=input_data.target_column,
        sample_weight_column=input_data.sample_weight_column,
        forecast_start=input_data.forecast_start,
    )


__all__ = ["combine_forecast_input_datasets"]
