"""ExplainableMetaForecaster Mixin."""

from abc import ABC, abstractmethod

import pandas as pd

from openstef_core.datasets import ForecastInputDataset


class ContributionsMixin(ABC):
    """Mixin class for models that support contribution analysis."""

    @abstractmethod
    def predict_contributions(self, X: ForecastInputDataset) -> pd.DataFrame:
        """Get feature contributions for the given input data X."""
        raise NotImplementedError("This method should be implemented by subclasses.")
