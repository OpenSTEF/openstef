# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <openstef@lfenergy.org> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from functools import lru_cache

from openstef.app_settings import AppSettings


@lru_cache
def _get_app_settings() -> AppSettings:
    return AppSettings()


Settings = _get_app_settings()
