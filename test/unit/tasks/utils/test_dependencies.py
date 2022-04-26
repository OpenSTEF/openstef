# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest

from openstef.data_classes.prediction_job import PredictionJobDataClass
from openstef.tasks.utils import dependencies as deps


class TestDependencies(unittest.TestCase):
    def _build_prediction_job(self, pj_id, depends_on=None):
        return PredictionJobDataClass(
            id=pj_id,
            depends_on=depends_on,
            model="",
            forecast_type="",
            train_components=False,
            name="",
            lat=0,
            lon=0,
            resolution_minutes=0,
            horizon_minutes=0,
        )

    def setUp(self) -> None:
        self.pjs_with_deps = [
            self._build_prediction_job(1),
            self._build_prediction_job(2),
            self._build_prediction_job(3),
            self._build_prediction_job(4, depends_on=[1, 2]),
            self._build_prediction_job(5, depends_on=[1, 3]),
            self._build_prediction_job(6, depends_on=[4]),
            self._build_prediction_job(7),
        ]

        self.pjs_without_deps = [
            self._build_prediction_job(1),
            self._build_prediction_job(2),
            self._build_prediction_job(3),
            self._build_prediction_job(4),
            self._build_prediction_job(5),
            self._build_prediction_job(6),
            self._build_prediction_job(7),
        ]

        self.pjs_with_deps_and_str_ids = [
            self._build_prediction_job("one"),
            self._build_prediction_job("two"),
            self._build_prediction_job("three"),
            self._build_prediction_job(4, depends_on=["one", "two"]),
            self._build_prediction_job("five", depends_on=["one", "three"]),
            self._build_prediction_job("six", depends_on=[4]),
            self._build_prediction_job(7),
        ]

    def test_has_dependencies(self):
        assert deps.has_dependencies(self.pjs_with_deps)
        assert not deps.has_dependencies(self.pjs_without_deps)

    def test_find_groups_with_dependencies(self):
        graph, groups = deps.find_groups(self.pjs_with_deps, randomize_groups=True)
        assert len(groups) == 3

        expected_groups = [{1, 2, 3, 7}, {4, 5}, {6}]
        for actual_group, expected_group in zip(groups, expected_groups):
            actual_group = {pj.id for pj in actual_group}
            assert actual_group == expected_group

    def test_find_groups_without_dependencies(self):
        graph, groups = deps.find_groups(self.pjs_without_deps)
        assert len(groups) == 1
        actual_group = {pj.id for pj in groups[0]}
        assert actual_group == set(range(1, 8))

    def test_find_groups_with_dependencies_and_str_ids(self):
        graph, groups = deps.find_groups(
            self.pjs_with_deps_and_str_ids, randomize_groups=False
        )
        assert len(groups) == 3

        expected_groups = [{"one", "two", "three", 7}, {4, "five"}, {"six"}]
        for actual_group, expected_group in zip(groups, expected_groups):
            actual_group = {pj.id for pj in actual_group}
            assert actual_group == expected_group
