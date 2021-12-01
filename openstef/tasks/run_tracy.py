# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

"""run_tracy.py
Tracy checks the mysql todolist and tries her best to execute the
functions with desired inputs
This scripts works as follows:
  1. Checks the mysql table 'todolist' for jobs (which are not already in progress and
    which are not already failed)
  2. Set all newly aquired jobs to 'in progress'
For each job;
  3. Convert input arguments to a dict with 'args' and 'kwargs'
  4. Interpret the given function and arguments
  5. Execute the job
  6. Post result to Slack
  7. Remove job from mysql table
If job fails, set in progress to 2
All functions that tracy is able to execute need to be imported and defined in the
available_functions.

Example:
    This module is meant to be called directly from a CRON job.

    Alternatively this code can be run directly by running::
        $ python run_tracy.py
Attributes:
"""

# sql to create the Tracy jobs table (todolist)

# CREATE TABLE IF NOT EXISTS `tst_icarus`.`todolist` (
# `id` INT NOT NULL AUTO_INCREMENT ,
# `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ,
# `function` VARCHAR(200) NOT NULL ,
# `args` VARCHAR(200) NOT NULL ,
# `inprogress` BOOLEAN NULL DEFAULT NULL ,
# PRIMARY KEY (`id`), UNIQUE `id` (`id`))
# ENGINE = InnoDB;

from pathlib import Path

from openstef.enums import TracyJobResult
from openstef.monitoring import teams
from openstef.tasks.optimize_hyperparameters import optimize_hyperparameters_task
from openstef.tasks.train_model import train_model_task
from openstef.tasks.utils.taskcontext import TaskContext


def run_tracy(context):
    # Get all Tracy jobs
    tracy_jobs = context.database.ktp_api.get_all_tracy_jobs(inprogress=0)
    num_jobs = len(tracy_jobs)

    if num_jobs == 0:
        context.logger.warning(f"Number of tracy jobs is {num_jobs}, exit task")
        return

    context.logger.info("Start processing Tracy jobs", num_jobs=num_jobs)

    for i, job in enumerate(tracy_jobs):

        # get a new logger with bound job
        logger = context.logger.bind(job=job)
        logger.info("Process job", job_counter=i, total_jobs=num_jobs)

        # Set all retrieved items of the todolist to inprogress
        job["inprogress"] = 1
        context.database.ktp_api.update_tracy_job(job)

        pid = int(job["args"])
        pj = context.database.get_prediction_job(pid)
        result, exc = run_tracy_job(job, pj, context)
        # job processing was succefull
        if result is TracyJobResult.SUCCESS:
            logger.info("Succesfully processed Tracy job")
            # Delete job when succesfull
            context.database.ktp_api.delete_tracy_job(job)
            logger.info("Delete Tracy job")

        # job was unknown
        elif result is TracyJobResult.UNKNOWN:
            logger.error(f"Unkown Tracy job {job['function']}")

        # job processing failed / raised an exception
        elif result is TracyJobResult.FAILED:
            job["inprogress"] = 2
            context.database.ktp_api.update_tracy_job(job)
            msg = "Exception occured while processing Tracy job"
            logger.error(msg, exc_info=exc)
            teams.post_teams(teams.format_message(title=msg, params=job))

    context.logger.info("Finished processing all Tracy jobs - Tracy out!")


def run_tracy_job(job, pj, context):
    # Try to process Tracy job
    try:
        # If train model job (TODO remove old name when jobs are done)
        if job["function"] in ["train_model", "train_specific_model"]:
            train_model_task(pj, context, check_old_model_age=False)

        # If optimize hyperparameters job (TODO remove old name when jobs are done)
        elif job["function"] in [
            "optimize_hyperparameters",
            "optimize_hyperparameters_for_specific_pid",
        ]:
            optimize_hyperparameters_task(pj, context)

        # Else unknown job
        else:
            return TracyJobResult.UNKNOWN, None

        # job processing was succesfull
        return TracyJobResult.SUCCESS, None

    # Processing of Tracy job failed
    except Exception as e:
        return TracyJobResult.FAILED, e


def main():
    taskname = Path(__file__).name.replace(".py", "")

    with TaskContext(taskname) as context:
        run_tracy(context)


if __name__ == "__main__":
    main()
