# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import traceback

from ktpbase.config.config import ConfigManager
from ktpbase.database import DataBase
from ktpbase.log import logging

from openstf.monitoring.teams import post_teams
from openstf.monitoring.performance_meter import PerformanceMeter
from openstf.tasks.utils.predictionjobloop import PredictionJobException
from openstf.tasks.utils import utils


class TaskContext:
    def __init__(
        self,
        task_file,
        suppress_exceptions=False,
        post_teams_on_exception=True,
        on_exception=None,
        on_successful=None,
        on_end=None,
    ):
        """A context manager that can be used to run tasks with.

        Should be used as:
        with TaskContext("__file__") as context:
            pass

        Args:
            task_file (string): Path of the task module provided by __file__
            suppress_exceptions (bool, optional): If set to False the context
                manager will pass any raised exception on. Defaults to False.
            post_teams_on_exception (bool, optional): If set to True the context
                manager will automatically post a message to teams when an
                exception is encountered. Defaults to True.
            oon_exception (callable, optional): Callback, will be called
                when an exception is raised. Callable gets exc_type, exc_info,
                stack_info as arguments.
            on_successful (callable, optional): Callback, will be called
                everytime if the task is successful (no exception is raised).
            on_end (callable, optional): Callback, will be called if the task is
                completed. Callable gets a bool indicating success as argument.
        """
        self.name = utils.get_module_name(task_file)
        self.suppress_exceptions = suppress_exceptions
        self.post_teams_on_exception = post_teams_on_exception
        self.on_exception = on_exception
        self.on_successful = on_successful
        self.on_end = on_end

    def __enter__(self):
        self.config = ConfigManager.get_instance()

        logging.configure_logging(
            loglevel=self.config.loglevel,
            runtime_env=self.config.env,
        )
        self.logger = logging.get_logger(__name__).bind(task=self.name)

        self.database = DataBase()

        self.perf_meter = PerformanceMeter(self.logger)

        self.perf_meter.start_level("task", self.name)

        return self

    def __exit__(self, exc_type, exc_info, stack_info):
        self.successful = exc_type is None

        if self.successful:
            if self.on_successful is not None:
                self.on_successful()
        else:
            self.logger.error(
                "An error has occured",
                exc_info=exc_info,
                stack_info=stack_info,
            )
            if self.on_exception is not None:
                self.on_exception(exc_type, exc_info, stack_info)

            if self.post_teams_on_exception:
                self._send_teams_message(exc_type, exc_info, stack_info)

        if self.on_end is not None:
            self.on_end(self.successful)

        self.perf_meter.complete_level(self.successful)

        # Returning true stops the propagation of the exception
        return self.suppress_exceptions

    def _send_teams_message(self, exc_type, exc_info, stack_info):
        stack_text = "".join(traceback.format_exception(exc_type, exc_info, stack_info))
        # The teams webhook behaves weird. Using the correct order \r\n does NOT work
        # The 4 spaces forces teams to format a monospaced message for some reason
        stack_text = stack_text.replace("\n", "\n\r    ")
        msg = {
            "title": f"Task '{self.name}' raised an exception",
            "color": "#a54f4f",
            "sections": [],
        }
        if exc_type is PredictionJobException:
            metrics = exc_info.metrics
            msg["sections"].append(
                {
                    "facts": [
                        ("Number of pjs in this task", metrics["num_jobs"]),
                        ("Number of pjs started", metrics["jobs_started"]),
                        ("Number of pjs successful", metrics["jobs_successful"]),
                        ("Number of pjs unsuccessful", metrics["jobs_unsuccessful"]),
                    ],
                }
            )
            string_pids_unsuccessful = ", ".join(
                [f"{pid:d}" for pid in metrics["pids_unsuccessful"]]
            )
            msg["sections"].append(
                {
                    "facts": [
                        ("Unsuccessful pids", string_pids_unsuccessful),
                    ]
                }
            )

        msg["sections"].append(
            {
                "text": f"    {stack_text}",
            }
        )

        post_teams(msg)
