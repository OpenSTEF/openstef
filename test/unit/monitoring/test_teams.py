# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from test.unit.utils.base import BaseTestCase
from test.unit.utils.data import TestData
from unittest.mock import patch

import pandas as pd

from openstef.monitoring import teams


@patch("openstef.monitoring.teams.pymsteams")
class TestTeams(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)

    def test_post_teams(self, teamsmock):

        msg = "test"

        teams.post_teams(msg, url="MOCK_URL")
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)

    def test_post_teams_invalid_keys(self, teamsmock):

        msg = "test"
        invalid_coefs = pd.DataFrame(
            {
                "coef_name": ["a"],
                "coef_value_last": [0],
                "coef_value_new": [1],
            },
        )
        coefsdf = pd.DataFrame()

        teams.post_teams(
            msg,
            url="MOCK_URL",
            invalid_coefficients=invalid_coefs,
            coefficients_df=coefsdf,
        )
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)

    def test_build_sql_query_string(self, teamsmock):
        query_df = pd.DataFrame(data=[["a", 1], ["b", 2]], columns=["key", "value"])
        table = "table"

        query_expected = (
            "```  \nINSERT INTO table (key, value) VALUES  \n('a', 1),  \n('b', 2) "
            " \n```"
        )

        query_result = teams.build_sql_query_string(query_df, table)
        self.assertEqual(query_result, query_expected)
