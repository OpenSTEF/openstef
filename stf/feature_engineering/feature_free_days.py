# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os

import numpy as np

import holidays

HOLIDAY_CSV_PATH = os.path.dirname(__file__) + "/dutch_holidays_2020-2022.csv"


def create_holiday_functions(country="FR", years=None):
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
        years = [2020, 2021]

    holidays_object = holidays.CountryHoliday(country, years=years)

    # Make holiday function dict
    holiday_functions = {}
    # Add check function that includes all holidays of the provided csv
    holiday_functions.update(
        {"IsHoliday": lambda x: np.isin(x.index.py_datetime().date, holidays)}
    )

    # Loop over list of holidays names
    for date, holidayname in sorted(holidays_object.items()):
        # Define function explicitely to mitigate 'late binding' problem
        def make_holiday_func(holidayname=holidayname):
            return lambda x: x.index.py_datetime().date is date

        # Create lag function for each holiday
        holiday_functions.update(
            {"Is" + holidayname: make_holiday_func(holidayname=holidayname)}
        )

        # Extend with school holidays from workalendar here

        # Make schoolholiday "IsHoliday" functions and combine with original one

    return holiday_functions


if __name__ == "__main__":
    holiday_functions = create_holiday_functions(country="NL")
    print(holiday_functions)
