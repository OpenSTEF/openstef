# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, MetaEstimatorMixin, clone
from sklearn.utils.validation import check_is_fitted


class GroupedRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Meta-model that trains an instance of the base estimator for each key of a groupby operation applied on the data.
    The base estimator is a sklearn regressor, the groupby is performed on the columns specified in parameters.
    Moreover fit and predict methods can be performed in parallel for each group key thanks to joblib.
    Example:

    data = | index | group | x0 | x1 | x3 | y |
           |   0   |   1   | .. | .. | .. | . |
           |   1   |   2   | .. | .. | .. | . |
           |   2   |   1   | .. | .. | .. | . |
           |   3   |   2   | .. | .. | .. | . |

           [              X              ][ Y ]

    The GroupedRegressor on the data with the group_columns='group' fits 2 models :
        the model 1 with the row 0 and 2, columns x0, x1 and x3 as the features and column y as the target.
        the model 2 with the row 1 and 3, columns x0, x1 and x3 as the features and column y as the target.

    Parameters
    ----------
    base_estimator: RegressorMixin
        Regressor .

    group_columns : str, int, list
        Name(s) of the column(s) used as the key for groupby operation.

    n_jobs : int default=1
        The maximum number of concurrently running jobs,
         such as the number of Python worker processes when backend=”multiprocessing”
         or the size of the thread-pool when backend=”threading

     Attributes
    ----------
        feature_names_: list(str)
            All input feature (without group_columns).

        estimators_: dict(str, RegressorMixin)
            Dictionnary that stocks fitted estimators for each group.
            The keys are the keys of grouping and the values are the regressors fitted on the grouped data.
    """

    def __init__(self, base_estimator, group_columns, n_jobs=1):
        self.base_estimator = base_estimator
        if type(group_columns) in [int, str]:
            self.group_columns = [group_columns]
        else:
            self.group_columns = group_columns
        self.n_jobs = 1

    def _get_tags(self):
        return self.base_estimator._get_tags()

    def _check_group_columns(self, df):
        if type(self.group_columns) is not list:
            raise ValueError(
                "The group columns parameter should be a list, it gets a {}".format(
                    type(self.group_columns)
                )
            )
        for c in self.group_columns:
            if c not in list(df.columns):
                raise ValueError("The group column {} is missing!".format(c))

    def _partial_fit(self, group, df_group):
        estimator = clone(self.base_estimator)
        X = df_group.loc[:, self.feature_names_]
        y = df_group.loc[:, "__target__"]
        return (group, estimator.fit(X, y))

    def _partial_predict(self, group, df_group):
        return self.estimators_[group].predict(df_group)

    @classmethod
    def grouped_compute(cls, df, group_columns, func, n_jobs=1):
        index_name = df.index.name or "index"
        df_reset = df.reset_index()

        df_res = pd.DataFrame(index=df_reset.index)

        gb = df_reset.groupby(group_columns)

        if n_jobs > 1:
            group_res = joblib.Parallel(n_jobs=n_jobs)(
                joblib.delayed(func)(
                    group, df_group.set_index(index_name).drop(group_columns, axis=1)
                )
                for group, df_group in gb
            )
        else:
            group_res = (
                func(group, df_group.set_index(index_name).drop(group_columns, axis=1))
                for group, df_group in gb
            )
        return group_res, gb, df_res

    def _grouped_predict(self, df, n_jobs=1):
        group_res, gb, df_res = self.grouped_compute(
            df,
            self.group_columns,
            lambda group, df_group: self._partial_predict(group, df_group),
            n_jobs,
        )

        for (group, group_index), result in zip(gb.groups.items(), group_res):
            df_res.loc[group_index, "__result__"] = result

        return df_res["__result__"].to_numpy()

    def _grouped_fit(self, df, n_jobs=1):
        group_res, _, _ = self.grouped_compute(
            df,
            self.group_columns,
            lambda group, df_group: self._partial_fit(group, df_group),
            n_jobs,
        )
        return dict(group_res)

    def fit(self, x, y):
        df = pd.DataFrame(x).copy(deep=True)
        self._check_group_columns(df)
        self.feature_names_ = [
            c for c in list(df.columns) if c not in self.group_columns
        ]
        df.loc[:, "__target__"] = y
        self.estimators_ = self._grouped_fit(df, self.n_jobs)
        return self

    def predict(self, x):
        check_is_fitted(self)
        df = pd.DataFrame(x)
        self._check_group_columns(df)
        return self._grouped_predict(df, self.n_jobs)
