# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import traceback
from typing import Callable

import structlog

from openstef.exceptions import PredictionJobException
from openstef.monitoring.performance_meter import PerformanceMeter
from openstef.monitoring.teams import post_teams


class TaskContext:
    """A context manager that can be used to run tasks with.

    Should be used as:

    .. code-block:: py

        with TaskContext("my_task_name") as context:
            pass

    Args:
        name: Name of the task
        config: Configuration object, can be found in openSTEF-dbc
        database: Database object, can be found in openSTEF-dbc
        suppress_exceptions: If set to False the context
            manager will pass any raised exception on. Defaults to False.
        post_teams_on_exception: If set to True the context
            manager will automatically post a message to teams when an
            exception is encountered. Defaults to True.
        on_exception : Callback, will be called
            when an exception is raised. Callable gets exc_type, exc_info,
            stack_info as arguments.
        on_successful: Callback, will be called
            everytime if the task is successful (no exception is raised).
        on_end: Callback, will be called if the task is
            completed. Callable gets a bool indicating success as argument.

    """

    def __init__(
        self,
        name: str,
        config: object,
        database: object,
        suppress_exceptions: bool = False,
        post_teams_on_exception: bool = True,
        on_exception: Callable = None,
        on_successful: Callable = None,
        on_end: Callable = None,
    ):
        """Initialize context manager."""
        self.name = name
        self.suppress_exceptions = suppress_exceptions
        self.post_teams_on_exception = post_teams_on_exception
        self.on_exception = on_exception
        self.on_successful = on_successful
        self.on_end = on_end
        self.config = config
        self.database = database

    def __enter__(self):
        self.logger = structlog.get_logger(__name__).bind(task=self.name)

        self.perf_meter = PerformanceMeter(self.logger)

        self.perf_meter.start_level("task", self.name)

        return self

    def __exit__(self, exc_type: type, exc_info: str, stack_info: str) -> bool:
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

    def _send_teams_message(
        self, exc_type: type, exc_info: str, stack_info: str
    ) -> None:
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
                        ("Unsuccessful pid(s)", string_pids_unsuccessful),
                    ]
                }
            )
            # Add details of exceptions and pids
            string_pids_exceptions = "\n".join(
                [f"{key}:{value}\n" for key, value in metrics["exceptions"].items()]
            )
            msg["sections"].append(
                {
                    "facts": [
                        ("Exceptions: pid(s)", string_pids_exceptions),
                    ]
                }
            )

        msg["sections"].append(
            {
                "text": f"    {stack_text}",
            }
        )

        post_teams(
            msg, url=self.config.teams_monitoring_url, proxies=self.config.proxies
        )
