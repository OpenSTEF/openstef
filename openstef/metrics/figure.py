# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module contains all functions for generating figures."""
from typing import List, Union
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_feature_importance(feature_importance: pd.DataFrame) -> go.Figure:
    """Created a treemap plot based on feature importance and weights.

    Args:
        feature_importance: A DataFrame describing the
            feature importances and weights of the trained model.

    Returns:
        A treemap of the features.

    """
    feature_importance["parent"] = "Feature importance"

    return go.Figure(
        go.Treemap(
            labels=feature_importance.index,
            parents=feature_importance["parent"],
            values=feature_importance["gain"],
            customdata=feature_importance["weight"],
            marker=dict(colors=feature_importance["weight"], colorscale="greens"),
            hovertemplate=(
                "<b>%{label}</b><br>importance: %{value:.1%}"
                "<br>weight: %{customdata:.1%}<extra></extra>"
            ),
        ),
        layout={
            "margin": {
                "t": 0,
                "r": 0,
                "b": 0,
                "l": 0,
            }
        },
    )


def plot_data_series(
    data: Union[List[pd.DataFrame], List[pd.Series]],
    predict_data: Union[List[pd.DataFrame], List[pd.Series]] = None,
    horizon: int = 47,
    names: List[str] = None,
) -> go.Figure:
    """Plots passed data and optionally prediction data for specified horizon.

    Args:
        data: There are two options to use this function. Either
            pass a list of pandas.DataFrame where each dataframe contains a load
            column and a horizon column. Or pass a list of pandas.Series with
            unique indexing.
        predict_data: Similar to data, but
            for prediction data instead. When passing a list of pandas.DataFrame
            the column forecast should exist. Can be set to None.
        horizon: This function will select only data matching
            this horizon. Defaults to 47.
        names: The names that will be used in the
            legend of the plot. If None is passed, this will be build
            automatically based on the number of series passed.

    Returns:
        A line plot of each passed data series.

    """
    series_names = {
        1: ("series",),
        2: ("train", "validation"),
        3: ("train", "validation", "test"),
    }

    num_series = len(data)

    if names is None and num_series > 3:
        raise ValueError(
            "Cannot pass names=None when passing data with more than 3 series."
        )

    if names is None:
        names = series_names[num_series]

    if predict_data is None:
        # Check if horizon columns exists in the data
        if "horizon" in data[0]:
            # Filter data on given horizon
            filtered = []
            for series in data:
                mask = series["horizon"] == horizon
                filtered.append(series[mask]["load"])
        else:
            filtered = data

        return _plot_data(names, filtered)

    # Check if horizon columns exists in the data
    if "horizon" in data[0]:
        # Filter data on given horizon
        actuals = []
        predictions = []

        for series, predict_series in zip(data, predict_data):
            mask = series["horizon"] == horizon
            actuals.append(series[mask]["load"])
            predictions.append(predict_series[mask]["forecast"])
    else:
        actuals = data
        predictions = predict_data

    fig = _plot_data_and_predictions(names, actuals, predictions)
    fig.update_layout(
        title=f"Predictor in action for horizon: {horizon}",
    )

    return fig


def _plot_data(names: List[str], series: List[pd.Series]) -> go.Figure:
    """Create plot of data consisting of different splits.

    Note:
        Do not use this function directly, use plot_data_series instead.

    Args:
        names: Name of each seperate split.
        series: Each split as a seperate series.

    Returns:
        A line plot of each passed series.

    """
    # Build a combined DataFrame with all data.
    # This step is important to create forced NaNs to create gaps in the plot.
    combined = []
    for name, sequence in zip(names, series):
        combined.extend(
            [
                sequence.rename(name),
            ]
        )
    df_plot = pd.concat(combined, axis=1)

    fig = go.Figure()

    # Add a trace for every data series
    for i, name in enumerate(names):
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[name],
                name=name,
                line=dict(color=px.colors.qualitative.Set2[i]),
            )
        )

    fig.update_layout(yaxis_title="Load (MW)")

    return fig


def _plot_data_and_predictions(
    names: List[str], actuals: List[pd.Series], predictions: List[pd.Series]
) -> go.Figure:
    """Create plot of different data and prediction splits.

    Note:
        Do not use this function directly, use plot_data_series instead.

    Args:
        names: Name of each seperate split. The passed names will be suffixed with _actual and _predict for data and predictions respectively.
        actuals: Each data split as a seperate series.
        predictions: Each prediction split as a seperate series.

    Returns:
        A line plot of each passed series.

    """
    # Build a combined DataFrame with all data.
    # This step is important to create forced NaNs to create gaps in the plot.
    combined = []
    for name, actual, prediction in zip(names, actuals, predictions):
        combined.extend(
            [
                actual.rename(f"{name}_actual"),
                prediction.rename(f"{name}_predict"),
            ]
        )
    df_plot = pd.concat(combined, axis=1)

    fig = go.Figure()

    # Add a trace for every data series
    for i, name in enumerate(names):
        actual, predict = f"{name}_actual", f"{name}_predict"

        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[actual],
                name=actual,
                line=dict(color=px.colors.qualitative.Set2[i]),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[predict],
                name=predict,
                line=dict(dash="dot", color=px.colors.qualitative.Dark2[i]),
            )
        )

    fig.update_layout(yaxis_title="Load (MW)")

    return fig


def convert_to_base64_data_uri(path_in: str, path_out: str, content_type: str) -> None:
    """Read file, convert it to a data_uri, then writes the data_uri to file.

    Args:
        path_in: Path of the file that will be converted
        path_out: Path of the file containing the data uri
        content_type: Content type of the data uri according to
            (https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Type).

    """
    with open(path_in, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    data_uri = "data:{0};base64,{1}".format(content_type, encoded)

    with open(path_out, "wt") as f:
        f.write(data_uri)
