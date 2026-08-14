# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""Forecast postprocessing transformations.

Contains transforms that are applied to forecast results to improve accuracy,
apply business constraints, or enhance prediction quality. These transforms
operate on ForecastDataset objects after the core prediction step.
"""

from typing import TYPE_CHECKING

from openstef_models.transforms.postprocessing.confidence_interval_applicator import ConfidenceIntervalApplicator
from openstef_models.transforms.postprocessing.conformalized_quantile_calibrator import ConformalizedQuantileCalibrator
from openstef_models.transforms.postprocessing.isotonic_quantile_calibrator import IsotonicQuantileCalibrator
from openstef_models.transforms.postprocessing.quantile_sorter import QuantileSorter

if TYPE_CHECKING:
    from openstef_models.transforms.postprocessing.mapie_quantile_calibrator import MapieQuantileCalibrator


def __getattr__(name: str) -> object:
    """Load optional postprocessing transforms only when requested."""
    if name == "MapieQuantileCalibrator":
        from openstef_models.transforms.postprocessing.mapie_quantile_calibrator import (  # noqa: PLC0415
            MapieQuantileCalibrator,
        )

        globals()[name] = MapieQuantileCalibrator
        return MapieQuantileCalibrator
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


__all__ = [
    "ConfidenceIntervalApplicator",
    "ConformalizedQuantileCalibrator",
    "IsotonicQuantileCalibrator",
    "MapieQuantileCalibrator",
    "QuantileSorter",
]
