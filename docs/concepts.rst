.. comment:
    SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

.. _concepts:

Concepts
========

Looking at the architecture of OpenSTEF helps to understand OpenSTEF concepts.

Software architecture
---------------------

.. image:: https://user-images.githubusercontent.com/25053215/184536369-ed608e86-1ea8-4c16-8c6a-5eaeb145eedf.png
  :width: 800

OpenSTEF is set up as a package that performs machine learning to forecast energy loads on the energy grid.
It contains:

* **Prediction job**: input configuration for a task and/or pipeline 
  (e.g. train an XGB model for a certain location).
* **Tasks**: can be called to perform training, forecasting, or evaluation. 
  All tasks use corresponding pipelines. Tasks include getting data from a database, 
  raising task exceptions, and writing data to a database. 
* **Pipelines**: can be called to perform training, forecasting or evaluation by 
  giving input data to the pipeline. Users can choose to use tasks 
  (which fetch/write data for you), or use pipelines directly 
  (which requires fetching/writing data yourself).
* **Data validation**: is called by pipelines to validate data (e.g. checking for flatliners).
* **Feature engineering**: is called by pipelines to select required features for training/forecasting based on the configuration from the prediction job (e.g. create new features for energy load of yesterday, last week).
* **Machine learning**: is called by pipelines to perform training, forecasting, or evaluation based on the configuration from the prediction job (e.g. train an XGB quantile model).
* **Model storage**: is called by pipelines to store or fetch trained machine learning model with MLFlow (e.g. store model locally in disk/database/s3_bucket on AWS).
* **Post processing**: is called by pipelines to post process forecasting (e.g. combine forecast dataframe with extra configuration information from prediction job or split load forecast into solar, wind, and energy usage forecast).


If tasks are used, the openstef-dbc package is required as an interface to the database for reading/writing.
The current interface in openstef-dbc is for a MySQL database for configuration data (e.g. information for prediction jobs) and Influx database for feature data (e.g. weather, load, energy price data) and energy forecast data.

.. _application-architecture:

Application architecture
------------------------

OpenSTEF is just a software package by itself and needs more parts to run as an application.

.. image:: https://user-images.githubusercontent.com/25053215/184536367-c7914697-7a2a-45b8-b447-36aec1a6c1af.png
  :width: 800

It requires:

* Github repository:

  * (create yourself) **Data fetcher**: software package to fetch data and write it to a database (e.g. a scheduled cron job to fetch weather data in Kubernetes).
  * (create yourself) **Data API**: API to provide data from a database or other source to applications and users (e.g. a REST API).
  * (create yourself) **Forecaster**: software package to fetch config/data and run openstef tasks/pipelines (e.g. a scheduled cron job to train/forecast in Kubernetes).
  * (open source) **OpenSTEF**: software package that performs machine learning to forecast energy loads on the energy grid.
  * (open source) **OpenSTEF-dbc**: software package that is interface to read/write data from/to a database for openstef tasks.
* CI/CD

  * (create yourself) **Energy forecasting Application CI/CD**: CICD pipeline to build, test, and deploy forecasting application (e.g. to Kubernetes via Jenkins/Tekton).
  * (open source) **OpenSTEF package CI/CD**: CICD pipeline to build, test, and release OpenSTEF package to PyPI (via github actions).

* **Compute**: software applications can be run on Kubernetes on AWS.
* **Database**: SQL, influx, or other database can be used to store fetched data and forecasts.
* **Dashboard**: dashboard to visualize historic and forecasted energy loads.

.. include:: dashboard.rst

|

.. include:: dazls.rst
