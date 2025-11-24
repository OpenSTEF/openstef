# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""Core meta model interfaces and configurations.

Provides the fundamental building blocks for implementing meta models in OpenSTEF.
These mixins establish contracts that ensure consistent behavior across different meta model types
while ensuring full compatability with regular Forecasters.
"""

from openstef_models.models.forecasting.gblinear_forecaster import (
    GBLinearForecaster,
    GBLinearHyperParams,
)
from openstef_models.models.forecasting.lgbm_forecaster import LGBMForecaster, LGBMHyperParams
from openstef_models.models.forecasting.lgbmlinear_forecaster import (
    LGBMLinearForecaster,
    LGBMLinearHyperParams,
)
from openstef_models.models.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    XGBoostHyperParams,
)

BaseLearner = LGBMForecaster | LGBMLinearForecaster | XGBoostForecaster | GBLinearForecaster
BaseLearnerHyperParams = LGBMHyperParams | LGBMLinearHyperParams | XGBoostHyperParams | GBLinearHyperParams
