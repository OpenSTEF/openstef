# SPDX-FileCopyrightText: 2017-2025 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Any


class BaseLogger(ABC):
    """Abstract Base Logger Interface"""

    @abstractmethod
    def debug(self, message: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def info(self, message: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def error(self, message: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def exception(self, message: str, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def bind(self, **kwargs: Any) -> "BaseLogger":
        pass
