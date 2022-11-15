# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module defines the grouped regressor."""
from typing import Any, Callable, Union

import joblib
import numpy as np
import pandas as pd
from pandas.core.groupby.generic import DataFrameGroupBy
from sklearn.base import BaseEstimator, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


class GroupedRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Meta-model that trains an instance of the base estimator for each key of a groupby operation applied on the data.

    The base estimator is a sklearn regressor, the groupby is performed on the columns specified in parameters.
    Moreover fit and predict methods can be performed in parallel for each group key thanks to joblib.

    Example:

    .. code-block:: md

        data =  | index | group | x0 | x1 | x3 | y |
                |   0   |   1   | .. | .. | .. | . |
                |   1   |   2   | .. | .. | .. | . |
                |   2   |   1   | .. | .. | .. | . |
                |   3   |   2   | .. | .. | .. | . |

                [              X              ][ Y ]


    The GroupedRegressor on the data with the group_columns='group' fits 2 models:
        - The model 1 with the row 0 and 2, columns x0, x1 and x3 as the features and column y as the target.
        - The model 2 with the row 1 and 3, columns x0, x1 and x3 as the features and column y as the target.

    Args:
        base_estimator: Regressor .

        group_columns: Name(s) of the column(s) used as the key for groupby operation.

        n_jobs: default=1
            The maximum number of concurrently running jobs,
            such as the number of Python worker processes when backend=”multiprocessing”
            or the size of the thread-pool when backend=”threading

    Attributes:
        feature_names_:  All input feature (without group_columns).

        estimators_:
            Dictionnary that stocks fitted estimators for each group.
            The keys are the keys of grouping and the values are the regressors fitted on the grouped data.

    """

    def __init__(
        self,
        base_estimator: RegressorMixin,
        group_columns: Union[str, int, list[str], list[int]],
        n_jobs: int = 1,
    ):
        """Initialize meta model."""
        self.base_estimator = base_estimator
        if type(group_columns) in [int, str]:
            self.group_columns = [group_columns]
        else:
            self.group_columns = group_columns
        self.n_jobs = n_jobs

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

    def _partial_fit(
        self, group: Any, df_group: pd.DataFrame, eval_set=None, **kwargs
    ) -> tuple[Any, BaseEstimator]:
        estimator = clone(self.base_estimator)
        X = df_group.loc[:, self.feature_names_]
        y = df_group.loc[:, "__target__"]

        if eval_set is None:
            estimator_fitted = estimator.fit(X, y, **kwargs)
        else:
            estimator_fitted = estimator.fit(
                X,
                y,
                eval_set=[
                    (
                        df.loc[
                            (df[self.group_columns] == group).to_numpy().flatten(),
                            self.feature_names_,
                        ],
                        df.loc[
                            (df[self.group_columns] == group).to_numpy().flatten(),
                            "__target__",
                        ],
                    )
                    for df in eval_set
                ],
                **kwargs
            )

        return (group, estimator_fitted)

    def _partial_predict(self, group, df_group, **kwargs):
        return self.estimators_[group].predict(df_group, **kwargs)

    @classmethod
    def grouped_compute(
        cls,
        df: pd.DataFrame,
        group_columns: Union[list[str], list[int]],
        func: Callable[[tuple, pd.DataFrame], np.array],
        n_jobs: int = 1,
        eval_set=None,
    ) -> tuple[tuple[np.array, ...], DataFrameGroupBy, pd.DataFrame]:
        """Computes the specified function on each group defined by the grouping columns.

        It is an utility function used to perform fit and predict on each group.
        The df_res is the final dataframe that aggregate the results for each
        group. The group_res is a tuple where each field is corresponding to a results for a group. The gb is the
        grouping object.

        Args:
            df: DataFrame containing the input data necessary for the computation .
            group_columns: List of the columns used for the groupby operation
            func: Function that take the group key and the conrresponding data of this group
                and perform the computation on this group.
            n_jobs: The maximum number of concurrently running jobs,

        Returns:
            The tuple of the results of each group, the grouping dataframe and the global dataframe of results.

        """
        index_name = df.index.name or "index"
        df_reset = df.reset_index()

        df_res = pd.DataFrame(index=df_reset.index)

        gb = df_reset.groupby(group_columns)

        if n_jobs > 1:
            # Preferred scaling is at cluster level (e.g. k8s/serverless) instead of process level
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

    def _grouped_predict(self, df: pd.DataFrame, n_jobs: int = 1, **kwargs) -> np.array:
        group_res, gb, df_res = self.grouped_compute(
            df,
            self.group_columns,
            lambda group, df_group: self._partial_predict(group, df_group, **kwargs),
            n_jobs,
        )

        for (group, group_index), result in zip(gb.groups.items(), group_res):
            df_res.loc[group_index, "__result__"] = np.array(result)

        return df_res["__result__"].to_numpy()

    def _grouped_fit(
        self, df: pd.DataFrame, n_jobs: int = 1, eval_set=None, **kwargs
    ) -> dict[Any, BaseEstimator]:
        group_res, _, _ = self.grouped_compute(
            df,
            self.group_columns,
            lambda group, df_group: self._partial_fit(
                group, df_group, eval_set=eval_set, **kwargs
            ),
            n_jobs,
        )
        return dict(group_res)

    def fit(self, x: np.ndarray, y: np.ndarray, eval_set=None, **kwargs):
        """Fit the model."""
        df = pd.DataFrame(x).copy(deep=True)
        self._check_group_columns(df)

        eval_df = None
        if eval_set is not None:
            eval_df = []
            for x_set, y_set in eval_set:
                self._check_group_columns(x_set)
                df_set = pd.DataFrame(x_set).copy(deep=True)
                df_set["__target__"] = y_set
                eval_df.append(df_set)

        self.feature_names_ = [
            c for c in list(df.columns) if c not in self.group_columns
        ]
        df.loc[:, "__target__"] = y
        self.estimators_ = self._grouped_fit(
            df, self.n_jobs, eval_set=eval_df, **kwargs
        )
        return self

    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Make a predicion."""
        check_is_fitted(self)
        df = pd.DataFrame(x)
        self._check_group_columns(df)
        return self._grouped_predict(df, self.n_jobs, **kwargs)
