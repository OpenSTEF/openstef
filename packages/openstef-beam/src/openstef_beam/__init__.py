# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""OpenSTEF BEAM: Backtesting, Evaluation, Analysis and Metrics framework.

BEAM provides all the tools to test energy forecasting models under realistic conditions.
Instead of simple validation that can mislead, BEAM simulates real-world scenarios using
versioned data - ensuring models only use information available at prediction time.

Key features:

**Real-world simulation**: Uses versioned data to prevent data leakage. Models are
retrained periodically and can only access historical information, just like in production.

**Flexible integration**: Plug in your own forecasting models, create custom metrics,
design specific visualizations, and select relevant lead times for your use case.

**Complete workflow**: Backtesting → Evaluation → Analysis → Benchmarking. From testing
individual models to comparing multiple approaches across different energy targets.

**Lead time analysis**: Evaluate how forecast quality changes from 1-hour to 48-hour
predictions, critical for operational planning in energy systems.

Use BEAM to:
    - Test models under realistic operational constraints
    - Compare forecasting approaches with versioned data integrity
    - Generate flexible reports tailored to your metrics and visualizations
    - Ensure evaluation results match real-world deployment performance

BEAM's versioned data approach and flexible architecture make it the reliable choice
for energy forecasting model validation that translates to production success.
"""

import logging

# Set up logging configuration
root_logger = logging.getLogger(name=__name__)
if not root_logger.handlers:
    root_logger.addHandler(logging.NullHandler())

__all__ = []
