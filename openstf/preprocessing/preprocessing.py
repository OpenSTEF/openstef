
import numpy as np

from openstf.validation import validation
from openstf.feature_engineering.apply_features import apply_multiple_horizon_features
from openstf.feature_engineering.general import remove_features_not_in_set


def pre_process_data(data, featureset=None, horizons=None):
    """Function that automates the pre processing of the data.

    Args:
        data (pd.DataFrame): Data with (unvalidated) input data and without features.

    Returns:

        pd.DataFrame: Cleaned data with features.

    """
    if horizons is None:
        horizons = [0.25, 47]

    # Validate input data
    validated_data = validation.validate(data)

    # Apply features
    # TODO it would be nicer to only generate the required features
    validated_data_data_with_features = apply_multiple_horizon_features(
        validated_data, h_aheads=horizons
    )

    # remove features not in requested set if required
    if featureset is not None:
        validated_data_data_with_features = remove_features_not_in_set(
            validated_data_data_with_features, featureset
        )

    # Clean up data
    clean_data_with_features = validation.clean(validated_data_data_with_features)

    return clean_data_with_features


def replace_repeated_values_with_nan(df, max_length, column_name):
    """Replace repeated values with NaN.

        Replace repeated values (sequentially repeating values), which repeat longer
        than a set max_length (in data points) with NaNs.

    Args:
        df (pandas.DataFrame): Data from which you would like to set repeating values to nan
        max_length (int): If a value repeats more often, sequentially, than this value, all those points are set to NaN
        column_name (string): the pandas dataframe column name of the column you want to process

    Rrturns:
        pandas.DataFrame: data, similar to df, with the desired values set to NaN.
    """
    data = df.copy(deep=True)
    indices = []
    old_value = -1000000000000
    value = 0
    for index, r in data.iterrows():
        value = r[column_name]
        if value == old_value:
            indices.append(index)
        elif (value != old_value) & (len(indices) > max_length):
            indices = indices[max_length:]
            data.at[indices, column_name] = np.nan
            indices = []
            indices.append(index)
        elif (value != old_value) & (len(indices) <= max_length):
            indices = []
            indices.append(index)
        old_value = value
    if len(indices) > max_length:
        data.at[indices, column_name] = np.nan
    return data


def replace_invalid_data(df, suspicious_moments):
    """Function that detects invalid data using the nonzero_flatliner function and converts the output to NaN values.

    Input:
    - df: pd.dataFrame(index=DatetimeIndex, columns = [load1, ..., loadN]). Load_corrections should be indicated by 'LC_'
    - suspicious_moments (pd.dataframe): output of function nonzero_flatliner in new variable


    return:
    - pd.DataFrame without invalid data (converted to NaN values)"""
    if suspicious_moments is not None:
        for index, row in suspicious_moments.iterrows():
            df[(row[0]) : (row[1])] = np.nan
        return df
    return df