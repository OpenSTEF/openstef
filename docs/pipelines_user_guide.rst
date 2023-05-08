.. comment:
    SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

.. _pipeline_user_guide:

Pipelines user guide
====================

As mentioned in the :ref:`concepts <concepts>` section, tasks are an extension of pipelines, that include getting data from a database,
raising task exceptions, and writing data to a database. In an operational setting, both tasks and pipelines can be used.
The main difference is that an operational application that leverages OpenSTEF's tasks fuctionality is easier to implement,
whereas the pipeline functionality offers more flexibility in terms of design and implementation in addition to offering more scalability.

To illustrate the task as well as the pipeline :ref:`concept <concepts>`, code snippets for both implementations are presented below.
These code snippets show two different ways in which OpenSTEF's pipeline functionality can be integrated within an application that runs in an operational setting.

Task implementation
-------------------

Let's first have a look at the task implementation, which is also the way it is done in the `GitHub repository containing the reference implementation <https://github.com/OpenSTEF/openstef-reference>`_.
In the case that model training, hyperparameter tuning, or forecasting is supposed to be ran according to a certain schedule, using CronJobs for example,
the task implementation is easy to set up.
However this implementation's scalability is limited. Additionally, this implementation relies on the `the OpenSTEF database connector <https://pypi.org/project/openstef-dbc/>`_, ``openstef-dbc``,
meaning that the databases have to be set up according to the `reference implementation <https://github.com/OpenSTEF/openstef-reference>`_.
Below, code snippets are shown for different types of tasks that demonstrate the use of OpenSTEF's task functionality.

Note that, apart from the imports, the implementation is the same for each type of task. The `config` object is a `pydantic.BaseSettings` object holding all relevanyt configuration such as usernames, secrets and hosts etc.

Train model task implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import sys
    from pathlib import Path

    from openstef.tasks import train_model as task
    from openstef_dbc.database import DataBase
    from openstef_dbc.log import logging

    def main():
        # Initialize logging
        logging.configure_logging(loglevel=config.loglevel, runtime_env=config.env)
        # Initialize database connection
        database = DataBase(config)
        task.main(config=config, database=database)


    if __name__ == "__main__":
        main()


Create forecast task implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    import sys
    from pathlib import Path

    from openstef.tasks import create_forecast as task
    from openstef_dbc.database import DataBase
    from openstef_dbc.log import logging

    def main():
        # Initialize logging
        logging.configure_logging(loglevel=config.loglevel, runtime_env=config.env)
        # Initialize database connection
        database = DataBase(config)
        task.main(config=config, database=database)


    if __name__ == "__main__":
        main()


Optimize hyperparameters task implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    from pathlib import Path

    from openstef.tasks import optimize_hyperparameters as task
    from openstef_dbc.database import DataBase
    from openstef_dbc.log import logging

    def main():
        # Initialize logging
        logging.configure_logging(loglevel=config.loglevel, runtime_env=config.env)
        # Initialize database connection
        database = DataBase(config)
        task.main(config=config, database=database)


    if __name__ == "__main__":
        main()


Create components forecast task implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    from pathlib import Path

    from openstef.tasks import create_components_forecast as task
    from openstef_dbc.database import DataBase
    from openstef_dbc.log import logging

    def main():
        # Initialize logging
        logging.configure_logging(loglevel=config.loglevel, runtime_env=config.env)
        # Initialize database connection
        database = DataBase(config)
        task.main(config=config, database=database)


    if __name__ == "__main__":
        main()


Create base case forecast task implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
::

    from pathlib import Path

    from openstef.tasks import create_basecase_forecast as task
    from openstef_dbc.database import DataBase
    from openstef_dbc.log import logging

    def main():
        # Initialize logging
        logging.configure_logging(loglevel=config.loglevel, runtime_env=config.env)
        # Initialize database connection
        database = DataBase(config)
        task.main(config=config, database=database)


    if __name__ == "__main__":
        main()


Pipeline implementation
-----------------------

The pipeline implementation does not rely on `the OpenSTEF database connector <https://pypi.org/project/openstef-dbc/>`_, ``openstef-dbc``.
Therefore, pipelines can be used together with any kind of database setup, unlike tasks,
which require databases to be implemented according to the `reference implementation <https://github.com/OpenSTEF/openstef-reference>`_.

A more scalable and arguably more neat set up than the `reference implementation <https://github.com/OpenSTEF/openstef-reference>`_,
is to expose the OpenSTEF pipeline functionality through an API,
for instance by using the `FastAPI framework <https://fastapi.tiangolo.com/>`_.
The code snippet below shows how OpenSTEF pipelines can be integrated into an API using the
`repository pattern <https://mpuig.github.io/Notes/fastapi_basics/02.repository_pattern/>`_::

    from typing import Any, List, Tuple

    import pandas as pd
    from openstef.data_classes.model_specifications import ModelSpecificationDataClass
    from openstef.data_classes.prediction_job import PredictionJobDataClass
    from openstef.metrics.reporter import Report
    from openstef.model.regressors.regressor import OpenstfRegressor
    from openstef.pipeline.create_basecase_forecast import create_basecase_forecast_pipeline
    from openstef.pipeline.create_forecast import create_forecast_pipeline_core
    from openstef.pipeline.optimize_hyperparameters import (
        optimize_hyperparameters_pipeline_core,
    )
    from openstef.pipeline.train_model import train_model_pipeline_core


    class OpenstefRepository:
        """Repository that exposes function to interact with OpenSTEF pipelines."""

        def forecast_pipeline(
            self,
            prediction_job: PredictionJobDataClass,
            input_data: pd.DataFrame,
            model: OpenstfRegressor,
            modelspecs: ModelSpecificationDataClass,
        ) -> pd.DataFrame:
            """Wrapper around the forecast pipeline of OpenSTEF.
            The input_data should contain a `load` column.
            """
            return create_forecast_pipeline_core(
                prediction_job, input_data, model, modelspecs
            )

        def basecase_forecast_pipeline(
            self,
            prediction_job: PredictionJobDataClass,
            input_data: pd.DataFrame,
        ) -> pd.DataFrame:
            """Wrapper around the basecase forecast pipeline of OpenSTEF.
            The input_data should contain a `load` column.
            """
            return create_basecase_forecast_pipeline(prediction_job, input_data)

        def train_pipeline(
            self,
            prediction_job: PredictionJobDataClass,
            modelspecs: ModelSpecificationDataClass,
            input_data: pd.DataFrame,
            horizons: List[float] = None,
            old_model: OpenstfRegressor = None,
        ) -> Tuple[
            OpenstfRegressor,
            Report,
            ModelSpecificationDataClass,
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        ]:
            """Wrapper around the train model pipeline of OpenSTEF.
            The input_data should contain a `load` column.
            """
            return train_model_pipeline_core(
                prediction_job,
                modelspecs,
                input_data,
                old_model,
                horizons=horizons,
            )

        def optimize_hyperparameters_pipeline(
            self,
            prediction_job: PredictionJobDataClass,
            input_data: pd.DataFrame,
            n_trials: int,
            horizons: List[float] = None,
        ) -> Tuple[
            OpenstfRegressor, ModelSpecificationDataClass, Report, dict, int, dict[str, Any]
        ]:
            """Wrapper around the optimize hyperparameters pipeline of OpenSTEF.
            The input_data should contain a `load` column.
            """
            return optimize_hyperparameters_pipeline_core(
                prediction_job, input_data, horizons, n_trials
            )
