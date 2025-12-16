# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""Mixins for adding explainability features to forecasting models.

Provides base classes that enable models to expose feature importance scores
and generate visualization plots.
"""

from abc import ABC, abstractmethod

import pandas as pd
import plotly.graph_objects as go

from openstef_core.datasets.validated_datasets import ForecastInputDataset
from openstef_core.types import Q, Quantile
from openstef_models.explainability.plotters.feature_importance_plotter import FeatureImportancePlotter


class ExplainableForecaster(ABC):
    """Mixin for forecasters that can explain feature importance.

    Provides a standardized interface for accessing and visualizing feature
    importance scores across different forecasting models.
    """

    @property
    @abstractmethod
    def feature_importances(self) -> pd.DataFrame:
        """Get feature importance scores for this model.

        Returns DataFrame with feature names as index and quantiles as columns.
        Each quantile represents the importance distribution across multiple
        model training runs or folds.

        Returns:
            DataFrame with feature names as index and quantile columns.
            Values represent normalized importance scores summing to 1.0.

        Note:
            The returned DataFrame must have feature names as index and quantile
            columns in format 'quantile_PXX' (e.g., 'quantile_P50', 'quantile_P95').
            All quantile values must be between 0 and 1.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_contributions(self, data: ForecastInputDataset, *, scale: bool) -> pd.DataFrame:
        """Get feature contributions for each prediction.

        Args:
            data: Input dataset for which to compute feature contributions.
            scale: Whether to scale contributions to sum to the prediction value.

        Returns:
            DataFrame with contributions per feature.
        """
        raise NotImplementedError

    def plot_feature_importances(self, quantile: Quantile = Q(0.5)) -> go.Figure:
        """Create interactive treemap visualization of feature importances.

        Args:
            quantile: Which quantile of importance scores to display.
                Defaults to median (0.5).

        Returns:
            Plotly Figure containing treemap with feature importance scores.
            Color intensity indicates relative importance of each feature.
        """
        return FeatureImportancePlotter().plot(scores=self.feature_importances, quantile=quantile)
