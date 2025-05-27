<!--
SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
SPDX-License-Identifier: MPL-2.0
-->

# Holiday Feature Bug Fix Documentation

## Issue Description

A bug was discovered in OpenSTEF's holiday feature generation that caused some holidays (specifically New Year's Day and potentially others) to not be correctly marked in the feature engineering process. The issue affected the `holiday_features.py` module.

### Technical Root Cause

The root cause was a classic Python "late binding closure" issue in the lambda functions used to create holiday detection features. The original implementation created lambda functions in a loop, which captured the loop variable by reference rather than by value. As a result, when the lambda functions were later called, they all referenced the last value assigned to the loop variable, rather than the value at the time the lambda was defined.

This meant that only the last date for a given holiday name (e.g., the most recent New Year's Day) would be correctly detected, while earlier occurrences of the same holiday would be missed.

## Solution

The fix involved several changes to the `holiday_features.py` module:

1. Added `collections` module to help group holidays by name
2. Restructured the holiday feature generation to first group all holiday dates by name
3. Created one function per holiday name that checks all dates for that holiday type
4. Used default arguments in lambda functions to capture the current value of variables at definition time
5. Applied the same fix to school holiday functions and bridge day functions

The key technique used in the fix is using default arguments in lambda functions to capture the current value of variables at the time the lambda is defined. This is a well-known solution to Python's late binding closure issue.

### Example of the Fix

Original code (with the bug):
```python
# Loop over list of holidays names
for date, holiday_name in sorted(country_holidays.items()):
    # Define function explicitly to mitigate 'late binding' problem
    def make_holiday_func(requested_date):
        return lambda x: np.isin(x.index.date, np.array([requested_date]))

    # Create lag function for each holiday
    holiday_functions.update(
        {"is_" + holiday_name.replace(" ", "_").lower(): make_holiday_func(date)}
    )
```

Fixed code:
```python
# Group holiday dates by name
holiday_dates_by_name = collections.defaultdict(list)
for date, holiday_name in sorted(country_holidays.items()):
    holiday_dates_by_name[holiday_name].append(date)

# Create one function per holiday name that checks all dates for that holiday
for holiday_name, dates in holiday_dates_by_name.items():
    # Use a default argument to capture the dates at definition time
    holiday_functions.update(
        {
            "is_" + holiday_name.replace(" ", "_").lower(): 
                lambda x, dates=dates: np.isin(x.index.date, np.array(dates))
        }
    )
```

## Testing

The fix was thoroughly tested in several ways:

1. Unit tests for the original feature engineering functionality were updated to check for specific holidays
2. A new test file `test_holiday_detection.py` was created to specifically verify that New Year's Day and other holidays are correctly detected across multiple years
3. A test that checks a full year of data was added to verify correct holiday counts
4. A direct test of the holiday function generator was created to confirm correct behavior when the functions are called directly

All tests now pass, confirming that the fix correctly resolves the issue.

## Additional Notes

The fix makes the holiday feature detection more accurate, which should improve forecast accuracy in the OpenSTEF system. The changes are backward compatible and shouldn't require any changes to code that uses the holiday features.

This fix addresses the general Python late binding closure issue that can affect any code that creates function closures in loops. By using default arguments to capture values at definition time, we ensure that the closures work correctly regardless of when they are called.
