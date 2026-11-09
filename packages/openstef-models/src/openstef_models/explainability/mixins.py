# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Mixins for adding explainability features to forecasting models.

Provides base classes that enable models to expose feature importance scores
and generate visualization plots.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import pandas as pd
import plotly.graph_objects as go

from openstef_core.datasets import ForecastInputDataset, TimeSeriesDataset
from openstef_core.types import Q, Quantile
from openstef_models.explainability.plotters.contributions_plotter import ContributionsPlotter
from openstef_models.explainability.plotters.feature_importance_plotter import FeatureImportancePlotter
from openstef_models.explainability.plotters.learning_curve_plotter import LearningCurvePlotter


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


class ContributionsMixin(ABC):
    """Mixin for forecasters that can explain per-sample feature contributions.

    Unlike ``ExplainableForecaster`` which provides aggregate feature importance,
    this mixin provides per-sample decomposition of predictions — i.e., how
    much each feature contributed to the prediction for each individual sample.

    For tree-based models (XGBoost), this corresponds to SHAP TreeExplainer values.
    For linear models (GBLinear), this is the coefficient x feature value decomposition.
    For ensembles, this shows each base model's contribution weight.
    """

    @abstractmethod
    def predict_contributions(self, data: ForecastInputDataset) -> TimeSeriesDataset:
        """Compute per-sample feature contributions for the given input data.

        Returns a TimeSeriesDataset where columns are feature names (or model
        names for ensemble contributions) and rows correspond to the same time
        index as the input.  Values represent the additive contribution of each
        feature to the prediction at that timestep.

        Args:
            data: Preprocessed input data (same format as ``predict()`` takes).

        Returns:
            TimeSeriesDataset with feature contributions.  Columns are features,
            rows are timesteps.  A ``bias`` column may be included for the
            model intercept/base value.
        """

    def plot_contributions(
        self,
        data: ForecastInputDataset,
        kind: Literal["heatmap", "waterfall", "bar"] = "heatmap",
        **kwargs: Any,
    ) -> go.Figure:
        """Plot per-sample feature contributions.

        Calls ``predict_contributions()`` and visualizes the result using the
        requested chart type.

        Args:
            data: Preprocessed input data.
            kind: Chart type — ``"heatmap"``, ``"waterfall"``, or ``"bar"``.
            **kwargs: Forwarded to the corresponding plotter method
                (e.g. ``top_n``, ``timestep``).

        Returns:
            Plotly Figure.

        Raises:
            ValueError: If *kind* is not one of the supported chart types.
        """
        contributions = self.predict_contributions(data)
        plotters = {
            "heatmap": ContributionsPlotter.plot_heatmap,
            "waterfall": ContributionsPlotter.plot_waterfall,
            "bar": ContributionsPlotter.plot_bar,
        }
        if kind not in plotters:
            msg = f"Unknown plot kind {kind!r}. Choose from {list(plotters)}"
            raise ValueError(msg)
        return plotters[kind](contributions=contributions, **kwargs)


def build_learning_curves(evals_result: Mapping[str, Mapping[str, Sequence[float]]]) -> pd.DataFrame:
    """Convert a boosting library's evaluation history into a learning curve table.

    XGBoost and LightGBM store one entry per evaluation set, in the order the
    sets were given to ``fit()``. OpenSTEF passes the training set first and,
    when available, the validation set second, so the entries are mapped to
    ``"train"`` and ``"validation"`` by position, not by name. If an entry holds
    several metrics, only the first one is used.

    Args:
        evals_result: Evaluation history, for example ``model.evals_result()``.
            Maps each evaluation set to a mapping of metric name to a list with
            one value per boosting round.

    Returns:
        DataFrame indexed by boosting round (starting at 1) with a ``train``
        column and, when validation data was given, a ``validation`` column.
    """
    set_names = ["train", "validation"]
    curves = {
        set_name: [float(value) for value in next(iter(metrics.values()))]
        for set_name, metrics in zip(set_names, evals_result.values(), strict=False)
    }
    curves_df = pd.DataFrame(curves)
    curves_df.index = pd.RangeIndex(start=1, stop=len(curves_df) + 1, name="round")
    return curves_df


class LearningCurveMixin(ABC):
    """Mixin for forecasters that record their training progress.

    Exposes the error after each boosting round on the training data and, when
    validation data was given, on the validation data. Comparing both curves
    makes overfitting visible: the training error keeps falling while the
    validation error stops improving or rises.
    """

    @property
    @abstractmethod
    def learning_curves(self) -> pd.DataFrame:
        """Per round error history recorded while fitting.

        Returns:
            DataFrame indexed by boosting round (starting at 1) with a ``train``
            column and, when validation data was given, a ``validation`` column.
            Values are the model's evaluation metric.
        """
        raise NotImplementedError

    def plot_learning_curves(self) -> go.Figure:
        """Plot the train and validation error per boosting round.

        Returns:
            Plotly Figure with one line per learning curve. When validation data
            was given, the round with the lowest validation error is marked.
        """
        return LearningCurvePlotter().plot(curves=self.learning_curves)
