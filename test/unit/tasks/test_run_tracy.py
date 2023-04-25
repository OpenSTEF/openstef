# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from test.unit.utils.data import TestData
from unittest import TestCase
from unittest.mock import MagicMock, patch

from openstef.tasks.run_tracy import run_tracy


def build_context_mock():
    context = MagicMock()
    context.database.ktp_api.get_all_tracy_jobs.return_value = TestData.load(
        "tracy_jobs.json"
    )
    return context


@patch("openstef.tasks.run_tracy.teams")
@patch("openstef.tasks.run_tracy.train_model_task")
@patch("openstef.tasks.run_tracy.optimize_hyperparameters_task")
class TestRunTracyTask(TestCase):
    def setUp(self) -> None:
        pass

    def test_run_tracy_happy_flow(self, *args):
        context = build_context_mock()
        jobs = context.database.ktp_api.get_all_tracy_jobs.return_value
        run_tracy(context)

        # if all jobs are successfull they should all have been deleted
        self.assertEqual(
            context.database.ktp_api.delete_tracy_job.call_count, len(jobs)
        )

    def test_run_tracy_no_jobs(self, *args):
        context = build_context_mock()
        context.database.ktp_api.get_all_tracy_jobs.return_value = []

        run_tracy(context)

        # nothing should be done
        # no info messages should have been logged only one warning
        self.assertEqual(context.logger.warning.call_count, 1)
        self.assertEqual(context.logger.info.call_count, 0)

    def test_run_tracy_unknown_job(self, *args):
        context = build_context_mock()
        jobs = context.database.ktp_api.get_all_tracy_jobs.return_value
        jobs[0]["function"] = "UNKNOWN FUNCTION"

        run_tracy(context)
        # If one job did fail it should not have been deleted
        self.assertEqual(
            context.database.ktp_api.delete_tracy_job.call_count, len(jobs) - 1
        )

    def test_run_tracy_failed_job(self, *args):
        hyperparams_task = args[0]
        teams_mock = args[2]
        hyperparams_task.side_effect = ValueError("Mock raising an exception")

        context = build_context_mock()
        jobs = context.database.ktp_api.get_all_tracy_jobs.return_value
        num_hyperparams_task = sum(
            [1 for j in jobs if "hyperparameters" in j["function"]]
        )

        run_tracy(context)

        # All hyperparameter optimization task should have failed
        # (teams is used when a job fails)
        self.assertEqual(teams_mock.post_teams.call_count, num_hyperparams_task)
