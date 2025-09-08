# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""OpenSTEF 3.0 compatibility layer for OpenSTEF 4.0 interfaces."""

import logging

# Set up logging configuration
root_logger = logging.getLogger(name=__name__)
if not root_logger.handlers:
    root_logger.addHandler(logging.NullHandler())

__all__ = []
