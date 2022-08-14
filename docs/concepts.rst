.. _concepts:

Concepts
========

Looking at the architecture of OpenSTEF package and example application helps to understand OpenSTEF and how to set it up.


Software architecture
---------------------

.. image:: https://user-images.githubusercontent.com/25053215/184536369-ed608e86-1ea8-4c16-8c6a-5eaeb145eedf.png
  :width: 800

OpenSTEF is setup as a package that performs machine learning to forecast energy loads on the energy grid.
It contains:

* **Prediction job**: input configuration for task and/or pipeline (e.g. train an XGB model for certain location).
* **Tasks**: can be called to perform training, forecasting or evaluation. All tasks use corresponding pipelines. Tasks include getting data from database, raising task exceptions and writing data to database.
* **Pipelines**: can be called to perform training, forecasting or evaluation by giving input data to pipeline. Users can choose to use tasks (which fetches/writes data) or use pipelines directly (which requires fetching/writing data yourself).
* **Data validation**: is called by pipelines to validate data (e.g. checking for flatliners).
* **Feature engineering**: is called by pipelines to select required features for training/forecasting based on configuration from prediction job (e.g. create new features for energy load of yesterday, last week).
* **Machine learning**: is called by pipelines to perform training, forecasting or evaluation based on configuration from prediction job (e.g. train an XGB quantile model).
* **Post processing**: is called by pipelines to postprocess forecasting (e.g. combine forecast dataframe with extra configuration information from prediction job or split load forecast into solar, wind and energy usage forecast).



Application architecture
------------------------

OpenSTEF application architecture

.. image:: https://user-images.githubusercontent.com/25053215/184536367-c7914697-7a2a-45b8-b447-36aec1a6c1af.png
  :width: 800
