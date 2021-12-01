# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime, timedelta
from typing import Tuple

import holidays
import numpy as np
import pandas as pd

from openstef import PROJECT_ROOT

HOLIDAY_CSV_PATH: str = (
    PROJECT_ROOT / "openstef" / "data" / "dutch_holidays_2020-2022.csv"
)


def generate_holiday_feature_functions(
    country: str = "NL",
    years: list = None,
    path_to_school_holidays_csv: str = HOLIDAY_CSV_PATH,
) -> dict:
    """
    Generates functions for creating holiday feature.
    This improves forecast accuracy. Examples of features that are added are:
        2020-01-01 is 'Nieuwjaarsdag'
        2022-12-24 - 2023-01-08 is the 'Kerstvakantie'
        2022-10-15 - 2022-10-23 is the 'HerfstvakantieNoord'


    The holidays are based on a manually generated csv file.
    The information is collected using:
    https://www.schoolvakanties-nederland.nl/ and the python holiday function
    The official following official ducth holidays are included untill 2023:
        - Kerstvakantie
        - Meivakantie
        - Herstvakantie
        - Bouwvak
        - Zomervakantie
        - Voorjaarsvakantie

        - Nieuwjaarsdag
        - Pasen
        - Koningsdag
        - Hemelvaart
        - Pinksteren
        - Kerst

    The 'Brugdagen' are updated untill dec 2020. (Generated using agenda)

    Returns:
        (dict): Dictionary with functions that check if a given date is a holiday, keys
                consist of "Is" + the_name_of_the_holiday_to_be_checked
    """

    if years is None:
        now = datetime.now()
        years = [now.year - 1, now.year]

    country_holidays = holidays.CountryHoliday(country, years=years)

    # Make holiday function dict
    holiday_functions = {}
    # Add check function that includes all holidays of the provided csv
    holiday_functions.update(
        {
            "is_national_holiday": lambda x: np.isin(
                x.index.date, np.array(list(country_holidays))
            )
        }
    )
    # Define empty list to keep track of bridgedays
    bridge_days = []
    # Loop over list of holidays names
    for date, holiday_name in sorted(country_holidays.items()):
        # Define function explicitely to mitigate 'late binding' problem
        def make_holiday_func(requested_date):
            return lambda x: np.isin(x.index.date, np.array([requested_date]))

        # Create lag function for each holiday
        holiday_functions.update(
            {"is_" + holiday_name.replace(" ", "_").lower(): make_holiday_func(date)}
        )

        # Check for bridge day
        holiday_functions, bridge_days = check_for_bridge_day(
            date, holiday_name, country, years, holiday_functions, bridge_days
        )

    # Add feature function that includes all bridgedays
    holiday_functions.update(
        {"is_bridgeday": lambda x: np.isin(x.index.date, np.array(list(bridge_days)))}
    )

    # Manully generated csv including all dutch schoolholidays for different regions
    df_holidays = pd.read_csv(path_to_school_holidays_csv, index_col=None)
    df_holidays["datum"] = pd.to_datetime(df_holidays.datum).apply(lambda x: x.date())

    # Add check function that includes all holidays of the provided csv
    holiday_functions.update(
        {"is_schoolholiday": lambda x: np.isin(x.index.date, df_holidays.datum.values)}
    )

    # Loop over list of holidays names
    for holiday_name in list(set(df_holidays.name)):
        # Define function explicitely to mitigate 'late binding' problem
        def make_holiday_func(holidayname=holiday_name):
            return lambda x: np.isin(
                x.index.date, df_holidays.datum[df_holidays.name == holidayname].values
            )

        # Create lag function for each holiday
        holiday_functions.update(
            {
                "is_"
                + holiday_name.replace(" ", "_").lower(): make_holiday_func(
                    holidayname=holiday_name
                )
            }
        )

    return holiday_functions


# Check for bridgedays
def check_for_bridge_day(
    date: datetime,
    holiday_name: str,
    country: str,
    years: list,
    holiday_functions: dict,
    bridge_days: list,
) -> Tuple[dict, list]:
    """Checks for bridgedays associated to a specific holiday with date (date).

    Any found bridgedays are appende dto the bridgedays list. Also a specific feature
    function for the bridgeday is added to the general holidayfuncitons dictionary.

    Args:
        date: datetime.datetime, date of holiday to check for associated bridgedays
        holiday_name: name of the holiday
        country: country for which to detect the bridgedays
        years: list of years for which to detect bridgedays
        holiday_functions: dictionary to which the featurefunction has to be appended to in case of a bridgeday
        bridge_days: list of bridgedays to which any found bridgedays have to be appended

    Returns:
        holiday_functions: dict with holiday feature functions
        bridge_days: list of bridgedays
    """
    country_holidays = holidays.CountryHoliday(country, years=years)

    # Define function explicitely to mitigate 'late binding' problem
    def make_holiday_func(requested_date):
        return lambda x: np.isin(x.index.date, np.array([requested_date]))

    # Looking forward: If day after tomorow is a national holiday or
    # a saturday check if tomorow is not a national holiday

    is_saturday_in_two_days = (date + timedelta(days=2)).weekday() == 5
    is_holiday_in_two_days = (date + timedelta(days=2)) in country_holidays

    is_holiday_tommorow = (date + timedelta(days=1)) in country_holidays
    is_weekend_tommorrow = (date + timedelta(days=1)).weekday() in [5, 6]

    if (is_holiday_in_two_days or is_saturday_in_two_days) and (
        not is_holiday_tommorow and not is_weekend_tommorrow
    ):

        # Create feature function for each holiday
        holiday_functions.update(
            {
                "is_bridgeday"
                + holiday_name.replace(" ", "_").lower(): make_holiday_func(
                    (date + timedelta(days=1))
                )
            }
        )
        bridge_days.append((date + timedelta(days=1)))

    # Looking backward: If day before yesterday is a national holiday
    # or a sunday check if yesterday is a national holiday
    is_saturday_two_days_ago = (date - timedelta(days=2)).weekday() == 6
    is_holiday_two_days_ago = (date - timedelta(days=2)) in country_holidays
    is_holiday_yesterday = (date - timedelta(days=1)) in country_holidays
    is_weekend_yesterday = (date - timedelta(days=1)).weekday() in [5, 6]

    if (is_saturday_two_days_ago or is_holiday_two_days_ago) and (
        not is_holiday_yesterday and not is_weekend_yesterday
    ):

        # Create featurefunction for the bridge function
        holiday_functions.update(
            {
                "is_bridgeday"
                + holiday_name.replace(" ", "_").lower(): make_holiday_func(
                    (date - timedelta(days=1))
                )
            }
        )
        bridge_days.append((date - timedelta(days=1)))

    return holiday_functions, bridge_days
