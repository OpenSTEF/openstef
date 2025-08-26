# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import holidays

print(holidays.country_holidays("NL", years=[2026], categories=["optional"]))
print(holidays.country_holidays("NL", years=[2026], categories=["public"]))
