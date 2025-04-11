# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData

import sklearn
from lightgbm import LGBMRegressor

from openstef.model.metamodels.grouped_regressor import GroupedRegressor
from openstef.model.regressors.linear import LinearRegressor


class TestGroupedRegressor(BaseTestCase):
    def setUp(self):
        self.train_input = TestData.load("reference_sets/307-train-data.csv")

        self.train_with_time = self.train_input.copy(deep=True)
        self.train_with_time["time"] = self.train_with_time.index.day

        self.train_x = self.train_with_time.iloc[:, 1:]
        self.train_y = self.train_with_time.iloc[:, 0]
        self.val_x = self.train_input.iloc[:, 1:]
        self.val_y = self.train_input.iloc[:, 0]

    def test_missing_group_columns(self):
        model = GroupedRegressor(LinearRegressor(), group_columns=["time"])
        model_without_group = GroupedRegressor(LinearRegressor(), group_columns=None)

        # test handling of group columns
        with self.assertRaises(ValueError):
            model_without_group.fit(self.train_x, self.train_y)

        with self.assertRaises(ValueError):
            model.fit(self.val_x, self.val_y)

    def test_fit(self):
        model = GroupedRegressor(LinearRegressor(), group_columns=["time"])
        # test fitting metamodel
        model.fit(self.train_x, self.train_y)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model))

        model_parallel = GroupedRegressor(
            LinearRegressor(), group_columns="time", n_jobs=4
        )
        # test parallel fitting
        model_parallel.fit(self.train_x, self.train_y)
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model_parallel))

        model = GroupedRegressor(LinearRegressor(), group_columns=["time"])
        model.fit(self.train_x, self.train_y)
        # test prediction
        res = model.predict(self.train_x)
        group = self.train_with_time.iloc[:, -1]
        for k, estimator in model.estimators_.items():
            self.assertTrue(
                (
                    res[group == k]
                    == estimator.predict(
                        self.train_with_time.loc[group == k].iloc[:, 1:-1]
                    )
                ).all()
            )

    def test_kwargs_handling(self):
        # test kwargs in fit and predict methods
        model_lgb = GroupedRegressor(LGBMRegressor(), group_columns="time")
        group = self.train_with_time.iloc[:, -1]
        with self.assertRaises(ValueError):
            model_lgb.fit(
                self.train_x, self.train_y, eval_set=[(self.val_x, self.val_y)]
            )

        model_lgb.fit(
            self.train_x,
            self.train_y,
            eval_set=[(self.train_x, self.train_y)],
        )
        self.assertIsNone(sklearn.utils.validation.check_is_fitted(model_lgb))
        res = model_lgb.predict(self.train_x, raw_score=True)
        for k, estimator in model_lgb.estimators_.items():
            self.assertTrue(
                (
                    res[group == k]
                    == estimator.predict(
                        self.train_with_time.loc[group == k].iloc[:, 1:-1],
                        raw_score=True,
                    )
                ).all()
            )
