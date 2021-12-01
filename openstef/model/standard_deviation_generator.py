# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin


class StandardDeviationGenerator:
    def __init__(self, validation_data: pd.DataFrame) -> None:

        self.validation_data = validation_data

    def generate_standard_deviation_data(self, model: RegressorMixin) -> RegressorMixin:

        # Define some variables
        predicted = None
        self.standard_deviation = pd.DataFrame()

        # Loop over horizons and ask prediction for each specific horizon
        for horizon in self.validation_data.horizon.unique():
            # Make subset for this specific horizon
            sub_val = self.validation_data[self.validation_data.horizon == horizon]
            try:
                predicted = model.predict(sub_val.iloc[:, 1:-1])
            except Exception as e:
                print("Could not get prediction from new model!", e)

            # Calculate confidence interval for this horizon
            confidence_interval_horizon = self._calculate_standard_deviation(
                sub_val.iloc[:, 0], predicted
            )
            confidence_interval_horizon[
                "horizon"
            ] = horizon  # Label with respective horizon
            self.standard_deviation = self.standard_deviation.append(
                confidence_interval_horizon
            )

        model.standard_deviation = self.standard_deviation

        return model

    @staticmethod
    def _calculate_standard_deviation(realised, predicted):
        """Protected static method to calculate the corrections for a model

        Args:
            realised: pd.series with realised load
            predicted: pd.series with load predicted by new model

        Returns:
            pd.DataFrame: with model corrections
        """
        result = pd.DataFrame(index=range(24), columns=["stdev", "hour"])
        # Calculate the error for each predicted point
        error = realised - predicted
        error.index = error.index.hour  # Hour only, remove the rest
        # For the time starts with 00, 01, 02, etc. TODO (MAKE MORE ELEGANT SOLUTION THAN A LOOP)
        for hour in range(24):
            hour_error = error[error.index == hour]
            result["stdev"].iloc[hour] = np.std(
                hour_error[1:]
            )  # Exclude first item as this is the hour itself!
            result["hour"].iloc[hour] = hour

        result = result.astype("float")

        return result
