# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class LoggerType(StrEnum):
    STANDARD = "logging"
    STRUCTLOG = "structlog"
