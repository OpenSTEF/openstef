from typing import cast, override
import numpy as np
from openstef_core.base_model import BaseConfig
from openstef_core.datasets import ForecastDataset, ForecastInputDataset, TimeSeriesDataset
from openstef_core.exceptions import NotFittedError
from openstef_core.mixins import Predictor
from openstef_core.types import Quantile
from openstef_models.models.calibration.calibrator import Calibrator
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression


class IsotonicQuantileCalibrator(Calibrator[ForecastInputDataset, ForecastDataset]):
    def __init__(self, model: xgb.XGBRegressor, quantiles: list[Quantile]) -> None:
        self._model = model
        self._is_fitted = False
        self._calibrators: dict[Quantile, IsotonicRegression] = {}
        self.quantiles = quantiles

    @override
    def is_fitted(self) -> bool:
        return self._is_fitted

    @override
    def fit(self, data: ForecastInputDataset) -> None:
        x_cal, y_cal = data.input_data(), data.target_series()

        uncalibrated_predictions = self._model.predict(x_cal)

        for i, quantile in enumerate(self.quantiles):
            predicted_quantile = uncalibrated_predictions[:, i]

            sort_indices = np.argsort(predicted_quantile)

            sorted_predictions = predicted_quantile[sort_indices]
            sorted_gt = y_cal.iloc[sort_indices]

            window_size = max(5, len(sorted_predictions) // 10)
            n_samples = len(sorted_predictions)

            calibrated_values = np.empty(n_samples)

            for j in range(n_samples):
                start_idx = max(0, j - window_size // 2)
                end_idx = min(n_samples, j + window_size // 2)
                window_gt = sorted_gt.iloc[start_idx:end_idx]
                calibrated_values[j] = np.quantile(window_gt, float(quantile))

            calibrator = IsotonicRegression(increasing=True, out_of_bounds="clip")
            calibrator.fit(sorted_predictions, calibrated_values)
            self._calibrators[quantile] = calibrator

        self._is_fitted = True

    @override
    def calibrate(self, predictions: ForecastDataset) -> ForecastDataset:
        if not self._is_fitted:
            raise NotFittedError("Model is not fitted yet.")

        uncalibrated_preds = predictions.data

        # Apply calibration to all quantiles (vectorized)
        calibrated_preds = np.column_stack([
            self._calibrators[q].predict(uncalibrated_preds[q.format()].to_numpy())
            for q in self.quantiles
        ])

        calibration_preds_df = pd.DataFrame(
            data=calibrated_preds,
            index=predictions.index,
            columns=[q.format() for q in self.quantiles]
        )

        # Ensure quantile ordering using vectorized sort
        return ForecastDataset(
            data=calibration_preds_df,
            sample_interval=predictions.sample_interval,
            forecast_start=predictions.forecast_start,
        )
