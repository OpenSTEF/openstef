# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import structlog

from openstef.logging.base_logger import BaseLogger
from openstef.settings import Settings


class StructlogLogger(BaseLogger):
    def __init__(self, name: str = __file__):
        self.logger = structlog.get_logger(name)
        structlog.stdlib.set_log_level(Settings.log_level)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs):
        self.logger.exception(message, **kwargs)
