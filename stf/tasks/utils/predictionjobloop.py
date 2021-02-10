# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import random
import sys


class PredictionJobLoop():
    def __init__(
            self,
            context,
            stop_on_exception=False,
            random_order=True,
            on_exception=None,
            on_successful=None,
            on_end=None,
            prediction_jobs=None,
            **pj_kwargs):
        """Convenience objects that maps a function over prediction jobs.

        Default behaviour is to automatically get prediction jobs from the
        database. Any keyword argument passed will be directed to the getting
        function. If another set of prediction jobs is desired, manually pass
        them using the prediction_jobs argument.

        Args:
            context (stf.tasks.util.taskcontext.TaskContext): The
                context to run this loop in.
            stop_on_exception (bool, optional): Whether to break out of the
                loop when an exception is raised. Defaults to False.
            random_order (bool, optional): Whether to randomize the order of the
                prediction jobs. Defaults to True. Does not apply to manually
                passed prediction jobs.
            on_exception (callable, optional): Callback, will be called
                everytime an exception is raised. Callable gets the pj and
                exception raised as arguments
            on_successful (callable, optional): Callback, will be called
                everytime an iteration is successful (no exception is raised).
                Callable gets the pj as argument.
            on_end (callable, optional): Callback, will be called everytime an
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
        self.on_exception = on_exception
        self.on_successful = on_successful
        self.on_end = on_end
        self.pj_kwargs = pj_kwargs

        if prediction_jobs is None:
            self._get_prediction_jobs()
        else:
            # retrieve pj info if not provided
            prediction_jobs = [pj if len(pj.keys()) > 1
                               else self.context.database.get_prediction_job(pj['id'])
                               for pj in prediction_jobs]
            self.prediction_jobs = prediction_jobs

    def _get_prediction_jobs(self):
        """Fetches prediction jobs from database.
        """
        self.context.logger.info(
            "Querying prediction jobs from database", **self.pj_kwargs
        )
        self.prediction_jobs = self.context.database.get_prediction_jobs(
            **self.pj_kwargs
        )

        if self.random_order:
            random.shuffle(self.prediction_jobs)

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

        for i, prediction_job in enumerate(self.prediction_jobs):
            successful = False

            self.context.logger = self.context.logger.bind(
                prediction_id=prediction_job.get("id"),
                prediction_name=prediction_job.get("name", ""),
            )

            self.context.perf_meter.start_level(
                "iteration", i, num_jobs=num_jobs, **kwargs
            )

            try:
                function(prediction_job, *args, **kwargs)

                pids_successful.append(prediction_job["id"])
                successful = True

                if self.on_successful is not None:
                    try:
                        self.on_successful(prediction_job)
                    except Exception:
                        _, exc_info, stack_info = sys.exc_info()
                        self.context.logger.error(
                            "An exception occured when executing the on_successful\
                             callback function for this iteration",
                            exc_info=exc_info,
                            stack_info=stack_info,
                        )

            except Exception as exception:
                last_job_exception = exception
                _, exc_info, stack_info = sys.exc_info()
                self.context.logger.error(
                    "An exception occured during this iteration",
                    exc_info=exc_info,
                    stack_info=stack_info,
                )
                pids_unsuccessful.append(prediction_job["id"])

                if self.on_exception is not None:
                    try:
                        self.on_exception(prediction_job, exception)
                    except Exception:
                        _, exc_info, stack_info = sys.exc_info()
                        self.context.logger.error(
                            "An exception occured when executing the on_exception\
                             callback function for this iteration",
                            exc_info=exc_info,
                            stack_info=stack_info,
                        )

                if self.stop_on_exception:
                    break
            finally:
                self.context.perf_meter.complete_level(successful)

                if self.on_end is not None:
                    try:
                        self.on_end(prediction_job, successful)
                    except Exception:
                        _, exc_info, stack_info = sys.exc_info()
                        self.context.logger.error(
                            "An exception occured when executing the on_end callback\
                            function for this iteration",
                            exc_info=exc_info,
                            stack_info=stack_info,
                        )

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


class PredictionJobException(Exception):
    def __init__(self, metrics={}):
        super(PredictionJobException, self).__init__(
            "One or more prediction jobs raised an exception."
        )
        self.metrics = metrics
