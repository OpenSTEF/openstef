# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

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
