# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime

import cufflinks as cf
import numpy as np
import pandas as pd

cf.go_offline()


def balance_classes(x, y):
    # find classes
    classes = y.unique()

    # find minimal number of samples per class
    n = min([len(np.unique(y[y == c].index.values)) for c in classes])

    # if min class is very small, raise amount of samples per class
    replace = False
    if n < 4:
        n = 4
        replace = True

    # sampled indices
    inds = []
    for c in classes:
        class_inds = np.unique(y[y == c].index.values)
        if len(class_inds) > 0:
            inds += list(np.random.choice(class_inds, size=n, replace=replace))

    mask = np.isin(y.index.values, inds)
    return x[mask], y[mask]


def prepare_training_data(df, y_col, val_n=4, val_width="14D"):
    # drop rows were load is not known
    df = df.dropna(subset=[y_col])

    # split in x (given) and y (to predict)
    x = df[[x for x in df.columns if x != y_col and x != "horizon"]]
    y = df[y_col]
    h = df["horizon"]

    # datetimes of data
    datetime_start = pd.to_datetime(str(df.index[0]))
    datetime_end = pd.to_datetime(str(df.index[-1]))

    # define validation set ranges
    val_ranges = []
    for i in range(val_n):
        val_start = (
            datetime_start
            + (i + 1) * ((datetime_end - datetime_start) / val_n)
            - pd.Timedelta(val_width)
        )
        val_end = val_start + pd.Timedelta(val_width)
        val_ranges.append((str(val_start.date()), str(val_end.date())))

    # validation set
    val_x = pd.concat([x[val_start:val_end] for val_start, val_end in val_ranges])
    val_y = pd.concat([y[val_start:val_end] for val_start, val_end in val_ranges])
    val_h = pd.concat([h[val_start:val_end] for val_start, val_end in val_ranges])

    # train set
    train_x = x.copy().drop(val_x.index)
    train_y = y.copy().drop(val_y.index)
    train_h = h.copy().drop(val_h.index)

    # balance classes
    train_x, train_y = balance_classes(train_x, train_y)
    val_x, val_y = balance_classes(val_x, val_y)

    return train_x, train_y, train_h, val_x, val_y, val_h


def prepare_prediction_data(df, y_col, y_hor):
    # pair prediction horizons to actual days
    horizon_dates = {}
    for h in y_hor:
        horizon_dates[str(h)] = str(
            (datetime.utcnow() + pd.Timedelta(str(h + 1) + "D")).date()
        )

    # for each horizon select corresponding input data
    x = []
    for h in horizon_dates:
        x.append(df.loc[horizon_dates[h]][df.loc[horizon_dates[h]]["horizon"] == h])
    x = pd.concat(x)

    # drop value to predict
    x = x.drop([y_col, "horizon"], axis="columns")

    return x


def visualize_predictions(df, classes):
    # define columns
    columns = []
    i = 0
    for c in classes:
        if i == 0:
            columns.append("<" + str(round(classes[c][1], 1)))
        elif i == len(classes) - 1:
            columns.append(str(round(classes[c][0], 1)) + str(">"))
        else:
            columns.append(
                str(round(classes[c][0], 1)) + "-" + str(round(classes[c][1], 1))
            )
        i += 1
    df.columns = columns

    # make plot
    fig = df.iplot(
        kind="heatmap",
        colorscale="blues",
        title="Capacity Prognosis (14D)",
        yTitle="Load Peak (MW)",
        asFigure=True,
        zmin=0,
        zmax=1,
    )

    return fig
