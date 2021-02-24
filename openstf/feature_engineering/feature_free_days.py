# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
import pandas as pd

from openstf import PROJECT_ROOT

HOLIDAY_CSV_PATH = PROJECT_ROOT / "openstf" / "data" / "dutch_holidays_2020-2022.csv"


def create_holiday_functions():
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

    # Manully generated csv including all dutch holidays for different regions
    df_holidays = pd.read_csv(HOLIDAY_CSV_PATH, index_col=None)
    df_holidays["datum"] = pd.to_datetime(df_holidays.datum).apply(lambda x: x.date())

    # Make holiday function dict
    holiday_functions = {}
    # Add check function that includes all holidays of the provided csv
    holiday_functions.update(
        {"IsFeestdag": lambda x: np.isin(x.index.date, df_holidays.datum.values)}
    )

    # Loop over list of holidays names
    for holidayname in list(set(df_holidays.name)):
        # Define function explicitely to mitigate 'late binding' problem
        def make_holiday_func(holidayname=holidayname):
            return lambda x: np.isin(
                x.index.date, df_holidays.datum[df_holidays.name == holidayname].values
            )

        # Create lag function for each holiday
        holiday_functions.update(
            {"Is" + holidayname: make_holiday_func(holidayname=holidayname)}
        )
    return holiday_functions
