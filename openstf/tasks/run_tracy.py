# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
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

# sql om de todolist tabel te maken:

# CREATE TABLE IF NOT EXISTS `tst_icarus`.`todolist` (
# `id` INT NOT NULL AUTO_INCREMENT ,
# `created` DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ,
# `function` VARCHAR(200) NOT NULL ,
# `args` VARCHAR(200) NOT NULL ,
# `inprogress` BOOLEAN NULL DEFAULT NULL ,
# PRIMARY KEY (`id`), UNIQUE `id` (`id`))
# ENGINE = InnoDB;

from functools import partial

from ktpbase.database import DataBase

from openstf.pipeline.optimize_hyperparameters import optimize_hyperparameters
from openstf.pipeline.train_model import train_model_pipeline
from openstf.monitoring.teams import post_teams
from openstf.tasks.utils.utils import (
    convert_string_args_to_dict_args,
    interpret_string_as_functions,
)
from openstf.tasks.utils.taskcontext import TaskContext


def get_and_evaluate_todos(context):
    # Get the to do list.
    todo_dict = context.database.ktp_api.get_all_tracy_jobs(inprogress=0)
    num_jobs = len(todo_dict)

    if num_jobs == 0:
        context.logger.warning("No TODO jobs to perform, exit", num_jobs=num_jobs)
        return

    context.logger.info("Start performing TODO jobs", num_jobs=num_jobs)

    for i, job in enumerate(todo_dict):
        # Set all retrieved items of the todolist to inprogress
        job["inprogress"] = 1
        context.database.ktp_api.update_tracy_job(job)

        context.logger = context.logger.bind(job_id=job["id"])
        context.logger.info("Process job", job_counter=i, total_jobs=num_jobs)

        # First try to parse the input directly
        try:
            interpreted_func = job["function"] + "(" + job["args"] + ")"
        except Exception as e:
            context.logger.error(
                "An exception occured while interpreting function", exc_info=e, job=job
            )
            context.logger.info("Try alternative parsing of function and arguments")

            function_arguments = convert_string_args_to_dict_args(job["args"])
            interpreted_func = interpret_string_as_functions(
                job["function"],
                function_arguments["args"],
                function_arguments["kwargs"],
            )

        context.logger.info("Interpreted the job", interpreted_func=interpreted_func)

        # Make eval safe by exluding all global functions, except the functions from func
        # Do this by setting al usually available global functions to None,
        # And adding func explicitely

        available_functions = {
            "train_specific_model": partial(train_specific_model, context),
            "optimize_hyperparameters_for_specific_pid": optimize_hyperparameters,
        }

        forbid_globals = {key: None for key in globals().keys()}
        forbid_globals.update(available_functions)

        # Try to evaluate the function
        try:
            # TODO eval is considered bad practice in (almost) every case
            res = eval(interpreted_func, forbid_globals)
            context.logger.debug(
                "Evaluated interpreted function",
                return_value=res,
                interpreted_func=interpreted_func,
            )

            teams_message = (
                f"Tracy executed a job. Job: `{interpreted_func}`   \n"
                f"Result: {str(res)}"
            )
            # TODO is this not redundant? for example optimize_hyperparameters
            # already sends a teams message with the results?
            post_teams(teams_message)

            # Remove job from mysql todolist
            context.database.ktp_api.delete_tracy_job(job)
            context.logger.info("Job removed from mysql table")
        except Exception as e:
            job["inprogress"] = 2
            context.database.ktp_api.update_tracy_job(job)
            context.logger.error(
                "An exception occured during evaluation of the interpreted function",
                exc_info=e,
            )
            teams_message = (
                f"Tracy could not execute job (`{interpreted_func}`), "
                f"set inprogress to 2"
            )
            post_teams(teams_message)

    context.logger = context.logger.unbind("job_id")
    context.logger.info("Finished all Tracy jobs - Tracy out!")


def train_specific_model(context, pid):
    """Train model for given prediction id.

    Tracy-compatible function to train a specific model based on the prediction id (pid)
    Should not be used outside of Tracy, preferred alternative:
        train_model_pipeline

    Args:
        pid (int): Prediction id of the corresponding prediction job.

    Returns:
        Trained model (FIXME can be various datatypes at present)
    """
    # Get DataBase instance:
    db = DataBase()

    # Get prediction job based on the given prediction ID (pid)
    pj = db.get_prediction_job(pid)

    # Train model for pj
    train_model_pipeline(pj, context, compare_to_old=False, retrain_young_models=True)


def main():
    with TaskContext("run_tracy") as context:
        get_and_evaluate_todos(context)


if __name__ == "__main__":
    main()
