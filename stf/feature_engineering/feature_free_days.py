# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os
from datetime import timedelta, datetime

import numpy as np
import pandas as pd

import holidays

HOLIDAY_CSV_PATH = os.path.dirname(__file__) + "/dutch_holidays_2020-2022.csv"


def create_holiday_functions(country="NL", years=None, path_to_school_holidays_csv=HOLIDAY_CSV_PATH):
    """
    This function provides functions for creating holiday feature.
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
        {"is_national_holiday": lambda x: np.isin(x.index.py_datetime().date, country_holidays)}
    )

    # Loop over list of holidays names
    for date, holiday_name in sorted(country_holidays.items()):
        # Define function explicitely to mitigate 'late binding' problem
        def make_holiday_func(date):
            return lambda x: x.index.py_datetime().date is date

        # Create lag function for each holiday
        holiday_functions.update(
            {"is_" + holiday_name: make_holiday_func(date)}
        )

        # Check for bridgedays
        # Looking forward: If day after tomorow is a national holiday or
        # a saturday check if tomorow is not a national holiday
        if (date+timedelta(days=2)) in country_holidays or\
                (date+timedelta(days=2)).weekday() == 5:

            # If tomorow is not a national holiday or a weekend day make it a bridgeday
            if not (date+timedelta(days=1)) in country_holidays \
                    and not (date+timedelta(days=1)).weekday() in [5, 6]:

                # Create feature function for each holiday
                holiday_functions.update(
                    {"is_bridgeday" + holiday_name:
                         make_holiday_func((date+timedelta(days=1)))}
                )
        # Looking backward: If day before yesterday is a national holiday
        # or a sunday check if yesterday is a national holiday
        if (date-timedelta(days=2)) in country_holidays or\
                (date-timedelta(days=2)).weekday() == 6:

            # If yesterday is a not a national holiday or a weekend day ymake it a bridgeday
            if not (date-timedelta(days=1)) in country_holidays \
                    and not (date-timedelta(days=1)).weekday() in [5, 6]:

                # Create featurefunction for the bridge function
                holiday_functions.update(
                    {"is_bridgeday" + holiday_name:
                         make_holiday_func((date-timedelta(days=1)))}
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
                x.index.date,
                df_holidays.datum[df_holidays.name == holidayname].values
            )
        # Create lag function for each holiday
        holiday_functions.update(
            {"is_" + holiday_name: make_holiday_func(holidayname=holiday_name)}
        )

    return holiday_functions


if __name__ == "__main__":
    holiday_functions = create_holiday_functions(country="NL")
    print(holiday_functions)
