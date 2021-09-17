# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd

from openstf.monitoring import teams

from test.utils import BaseTestCase, TestData


@patch("openstf.monitoring.teams.pymsteams")
class TestTeams(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.pj = TestData.get_prediction_job(pid=307)

    def test_post_teams(self, teamsmock):

        msg = "test"

        teams.post_teams(msg)
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)

    def test_post_teams_alert(self, teamsmock):

        msg = "test"

        teams.post_teams_alert(msg)
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

        teams.post_teams(msg, invalid_coefs=invalid_coefs, coefsdf=coefsdf)
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)

    def test_build_sql_query_string(self, teamsmock):
        query_df = pd.DataFrame(data=[["a", 1], ["b", 2]], columns=["key", "value"])
        table = "table"

        query_expected = "```  \nINSERT INTO table (key, value) VALUES  \n('a', 1),  \n('b', 2)  \n```"

        query_result = teams.build_sql_query_string(query_df, table)
        self.assertEqual(query_result, query_expected)

    def test_post_teams_better(self, teamsmock):

        test_feature_weights = pd.DataFrame(data={"gain": [1, 2]})

        teams.send_report_teams_better(self.pj, test_feature_weights)
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)

    @patch("openstf.monitoring.teams.open", MagicMock())
    def test_post_teams_worse(self, teamsmock):

        teams.send_report_teams_worse(self.pj)
        card_mock = teamsmock.connectorcard.return_value
        self.assertTrue(card_mock.send.called)
