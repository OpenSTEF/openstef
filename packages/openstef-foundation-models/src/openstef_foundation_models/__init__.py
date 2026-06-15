# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <openstef@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""Foundation model support for OpenSTEF."""

import logging

root_logger = logging.getLogger(name=__name__)
if not root_logger.handlers:
    root_logger.addHandler(logging.NullHandler())

__all__: list[str] = []
