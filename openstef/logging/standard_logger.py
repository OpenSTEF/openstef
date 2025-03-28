# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import logging

from openstef.logging.base_logger import BaseLogger
from openstef.settings import Settings


class StandardLogger(BaseLogger):
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        logging.basicConfig(level=Settings.log_level)

    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)

    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra=kwargs)

    def bind(self, **kwargs):
        """Not implemented for StandardLogger"""
        return self
