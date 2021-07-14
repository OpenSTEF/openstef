# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import random
import sys


class PredictionJobLoop:
    def __init__(
        self,
        context,
        stop_on_exception=False,
        random_order=True,
        on_exception_callback=None,
        on_successful_callback=None,
        on_end_callback=None,
        prediction_jobs=None,
        **pj_kwargs
    ):
        """Convenience objects that maps a function over prediction jobs.

        Default behaviour is to automatically get prediction jobs from the
        database. Any keyword argument passed will be directed to the getting
        function. If another set of prediction jobs is desired, manually pass
        them using the prediction_jobs argument.

        Args:
            context (openstf.tasks.util.taskcontext.TaskContext): The
                context to run this loop in.
            stop_on_exception (bool, optional): Whether to break out of the
                loop when an exception is raised. Defaults to False.
            random_order (bool, optional): Whether to randomize the order of the
                prediction jobs. Defaults to True. Does not apply to manually
                passed prediction jobs.
            on_exception_callback (callable, optional): Callback, will be called
                everytime an exception is raised. Callable gets the pj and
                exception raised as arguments
            on_successful_callback (callable, optional): Callback, will be called
                everytime an iteration is successful (no exception is raised).
                Callable gets the pj as argument.
            on_end_callback (callable, optional): Callback, will be called everytime an
                iteration is completed. Callable gets the pj and and bool
                indicating success as argument.
            prediction_jobs (list of dicts, optional): Manually pass a list of
                prediction jobs that will be looped over. A prediction job is a dict
                with at least a value set for the key `id`. If set to None, will fetch
                prediction jobs from database.
            **pj_kwargs: Any other kwargs willed will be directed to the
                prediction job getting function.

        """
        self.context = context
        self.stop_on_exception = stop_on_exception
        self.random_order = random_order
        self.on_exception_callback = on_exception_callback
        self.on_successful_callback = on_successful_callback
        self.on_end_callback = on_end_callback
        self.pj_kwargs = pj_kwargs

        if prediction_jobs is None:
            self.prediction_jobs = self._get_prediction_jobs()
        else:
            self.prediction_jobs = self._add_missing_info_prediction_jobs(
                prediction_jobs
            )

        if self.random_order:
            random.shuffle(self.prediction_jobs)

    def _get_prediction_jobs(self):
        """Fetches prediction jobs from database."""
        self.context.logger.info(
            "Querying prediction jobs from database", **self.pj_kwargs
        )
        prediction_jobs = self.context.database.get_prediction_jobs(**self.pj_kwargs)

        return prediction_jobs

    def _add_missing_info_prediction_jobs(self, prediction_jobs_input):
        prediction_jobs = []

        for pj in prediction_jobs_input:
            if len(pj.keys()) > 1:
                prediction_jobs.append(pj)
                continue

            prediction_jobs.append(self.context.database.get_prediction_job(pj["id"]))

        return prediction_jobs

    def map(self, function, *args, **kwargs):
        """Maps the passed function over all prediction jobs.

        Args:
            function (callable): The function that will be applied to each prediction
                job separately.
            *args: Any other arguments or passed to the function.
            **kwargs: All keyword arguments are passed to the function. This method is
                highly prefered over using args, since kwargs will be automatically
                logged.

        Raises:
            PredictionJobException: This exception will be raised if one or more
            prediction jobs raised an exception during the loop.
        """
        pids_successful = []
        pids_unsuccessful = []
        last_job_exception = None

        num_jobs = len(self.prediction_jobs)
        self.context.perf_meter.checkpoint("pre-loop")

        # loop over prefiction jobs's
        for i, prediction_job in enumerate(self.prediction_jobs):
            successful = False

            self.context.logger = self.context.logger.bind(
                prediction_id=prediction_job.get("id"),
                prediction_name=prediction_job.get("name", ""),
            )

            self.context.perf_meter.start_level(
                "iteration",
                i,
                num_jobs=num_jobs,
                pid=prediction_job.get("id"),
                **kwargs
            )

            try:
                function(prediction_job, *args, **kwargs)

                pids_successful.append(prediction_job["id"])
                successful = True

                self._handle_successful_iteration(prediction_job)

            except Exception as exception:
                pids_unsuccessful.append(prediction_job["id"])
                last_job_exception = exception

                self._handle_exception_during_iteration(prediction_job, exception)

                if self.stop_on_exception:
                    break
            finally:
                self._handle_finished_last_iteration(prediction_job, successful)

            self.context.logger = self.context.logger.unbind("prediction_id")

        jobs_successful = len(pids_successful)
        jobs_unsuccessful = len(pids_unsuccessful)
        jobs_started = jobs_successful + jobs_unsuccessful

        # This log is for human readable logging
        self.context.perf_meter.checkpoint(
            "loop",
            num_jobs=num_jobs,
            jobs_started=jobs_started,
            jobs_successful=jobs_successful,
            jobs_unsuccessful=jobs_unsuccessful,
            successful=int(jobs_unsuccessful > 0),
        )

        if jobs_unsuccessful > 0:
            metrics = {
                "num_jobs": num_jobs,
                "pids_successful": pids_successful,
                "pids_unsuccessful": pids_unsuccessful,
                "jobs_successful": jobs_successful,
                "jobs_unsuccessful": jobs_unsuccessful,
                "jobs_started": jobs_started,
            }

            raise PredictionJobException(metrics) from last_job_exception

    def _handle_successful_iteration(self, prediction_job):
        if self.on_successful_callback is not None:
            try:
                self.on_successful_callback(prediction_job)
            except Exception:
                _, exc_info, stack_info = sys.exc_info()
                self.context.logger.error(
                    "An exception occured when executing the on_successful_callback\
                        callback function for this iteration",
                    exc_info=exc_info,
                    stack_info=stack_info,
                )

    def _handle_exception_during_iteration(self, prediction_job, e):
        _, exc_info, stack_info = sys.exc_info()
        self.context.logger.error(
            "An exception occured during this iteration",
            exc_info=exc_info,
            stack_info=stack_info,
        )

        if self.on_exception_callback is None:
            return

        try:
            self.on_exception_callback(prediction_job, e)
        except Exception:
            _, exc_info, stack_info = sys.exc_info()
            self.context.logger.error(
                "An exception occured when executing the on_exception_callback "
                "callback function for this iteration",
                exc_info=exc_info,
                stack_info=stack_info,
            )

    def _handle_finished_last_iteration(self, prediction_job, successful):

        self.context.perf_meter.complete_level(successful)

        if self.on_end_callback is not None:
            try:
                self.on_end_callback(prediction_job, successful)
            except Exception:
                _, exc_info, stack_info = sys.exc_info()
                self.context.logger.error(
                    "An exception occured when executing the on_end_callback callback\
                    function for this iteration",
                    exc_info=exc_info,
                    stack_info=stack_info,
                )


class PredictionJobException(Exception):
    def __init__(self, metrics=None):
        super().__init__("One or more prediction jobs raised an exception.")
        if metrics is None:
            metrics = {}
        self.metrics = metrics
