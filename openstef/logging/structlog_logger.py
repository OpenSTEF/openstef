# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import logging

import structlog

from openstef.logging.base_logger import BaseLogger
from openstef.settings import Settings


class StructlogLogger(BaseLogger):
    def __init__(self, name: str):
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(
                logging.getLevelName(Settings.log_level)
            )
        )
        self.logger = structlog.get_logger(name)

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

    def bind(self, **kwargs):
        return self.logger.bind(**kwargs)
