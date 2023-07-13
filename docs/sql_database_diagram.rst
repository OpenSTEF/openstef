.. comment:
    SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

Overview of relational database
============================================================

OpenSTEF uses a relational database to store information about prediction jobs and measurements. An ER diagram of this database is shown bellow.

.. image:: _static/mysql_er_diagram.png
   :width: 6.3in
   :height: 4.24236in

The nescesarry tables are described in more detail bellow:

customers
-------------
+----------------+----------+-----------------------+-----------------+
| **Name**       | **Type** | **Comment**           | **Example**     |
+================+==========+=======================+=================+
| id             | int      | customer id           | 307             |
+----------------+----------+-----------------------+-----------------+
| name           | chr      | customer name         | Location_A      |
+----------------+----------+-----------------------+-----------------+
| vip            | bool     | extra important       | 1               |
|                |          | forecast (deprecated) |                 |
+----------------+----------+-----------------------+-----------------+
| active         | bool     | activity status       | 1               |
+----------------+----------+-----------------------+-----------------+

**Customer** : A customer is a collection of prediction jobs. This can be a collection of predictions belonging to a customer but also a collection of prediction belonging to a specific location or substation.

customersApiKeys
----------------
+----------------+----------------+-----------------+-----------------+
| **Name**       | **Type**       | **Comment**     | **Example**     |
+================+================+=================+=================+
| id             | int            | API key id      | 94              |
+----------------+----------------+-----------------+-----------------+
| cid            | int            | customer id     | 307             |
+----------------+----------------+-----------------+-----------------+
| name           | chr            | customer name   | Location_A      |
+----------------+----------------+-----------------+-----------------+
| apiKey         | chr            | API key value   | uuid-Location_A |
+----------------+----------------+-----------------+-----------------+

For users to post measurements or retrieve forecasts related to a
specific customer (used internally by Alliander).

**customers_predictions**

Correspondence table between customer ids and prediction jobs ids.

+--------------------+-----------+-----------------------+-------------+
| **Name**           | **Type**  | **Comment**           | **Example** |
+====================+===========+=======================+=============+
| customer_id        | int       | customer id           | 307         |
+--------------------+-----------+-----------------------+-------------+
| prediction_id      | int       | prediction job id     | 313         |
+--------------------+-----------+-----------------------+-------------+

energy_split_coefs
------------------
+----------------+----------------+-----------------+-----------------+
| **Name**       | **Type**       | **Comment**     | **Example**     |
+================+================+=================+=================+
| id             | int            | coefficient id  | 1135293         |
+----------------+----------------+-----------------+-----------------+
| pid            | int            | prediction job  | 317             |
|                |                | id              |                 |
+----------------+----------------+-----------------+-----------------+
| date_start     | date           | start date      | 2020-12-08      |
+----------------+----------------+-----------------+-----------------+
| date_end       | date           | end date        | 2021-03-08      |
+----------------+----------------+-----------------+-----------------+
| created        | datetime       | creation        | 2021-03-08      |
|                |                | datetime        | 12:14:19        |
+----------------+----------------+-----------------+-----------------+
| coef_name      | chr            | coefficient     | wind_ref        |
|                |                | name            |                 |
+----------------+----------------+-----------------+-----------------+
| coef_value     | float          | coefficient     | 52.6867         |
|                |                | value           |                 |
+----------------+----------------+-----------------+-----------------+

Deprecated: OpenSTEF now uses DAZLS (`Concepts — OpenSTEF
documentation <https://openstef.github.io/openstef/concepts.html#domain-adaptation-for-zero-shot-learning-in-sequence-dazls>`__)
to split net load between wind, solar and the rest.

genericpowercurves
----------------------
Contains the generic load curves of wind turbines. These curves are
two-parameters sigmoids (center and slope).

+---------------+------------+-------------------------+----------------+
| **Name**      | **Type**   | **Comment**             | **Example**    |
+===============+============+=========================+================+
| name          | chr        | turbine name            | Vestas V112    |
+---------------+------------+-------------------------+----------------+
| cut_in        | float      | min wind speed to       | 3              |
|               |            | produce (m/s)           |                |
+---------------+------------+-------------------------+----------------+
| cut_off       | float      | max wind speed to       | 25             |
|               |            | produce (m/s)           |                |
+---------------+------------+-------------------------+----------------+
| kind          | chr        | onshore / offshore      | onshore        |
+---------------+------------+-------------------------+----------------+
| manufacturer  | chr        |                         | Enercon        |
+---------------+------------+-------------------------+----------------+
| peak_capacity | float      | max power (W)           | 3040270        |
+---------------+------------+-------------------------+----------------+
| rated_power   | float      | rated power (W)         | 3000000        |
+---------------+------------+-------------------------+----------------+
| slope_center  | float      | Wind speed              | 7.91           |
|               |            | corresponding to 50% of |                |
|               |            | rated power (m/s)       |                |
+---------------+------------+-------------------------+----------------+
| steepness     | float      | See formula             | 0.76           |
+---------------+------------+-------------------------+----------------+

In openstef/feature_engineering/weather_features.py, the power delivered
by a wind turbine is computed as

.. math:: P(v) = \frac{P_{rated}}{1 + e^{- k(v - c)}},

where :math:`v` is the windspeed at hub height, :math:`P_{rated}` =
rated_power, :math:`k` = steepness and :math:`c` = slope_center.

**hyper_params**

Contains the names of the hyper-parameters to be tuned for each type of
model (xgb…).

+----------------+----------+-----------------------+-----------------+
| **Name**       | **Type** | **Comment**           | **Example**     |
+================+==========+=======================+=================+
| id             | int      | hyper-parameter id    | 1               |
+----------------+----------+-----------------------+-----------------+
| name           | chr      | hyper-parameter name  | subsample       |
+----------------+----------+-----------------------+-----------------+
| model          | chr      | name of the model     | xgb             |
+----------------+----------+-----------------------+-----------------+

Deprecated: Hyper-parameters are managed by MLflow.

hyper_param_values
------------------
Contains the values of the hyper-parameters to be tuned for each
prediction job.

+-----------------+----------+--------------------------+--------------+
| **Name**        | **Type** | **Comment**              | **Example**  |
+=================+==========+==========================+==============+
| id              | int      | Hyper-parameter id       | 159          |
+-----------------+----------+--------------------------+--------------+
| prediction_id   | int      | prediction job id        | 313          |
+-----------------+----------+--------------------------+--------------+
| hyper_params_id | int      | hyper-parameter id in    | 1            |
|                 |          | hyper_params             |              |
+-----------------+----------+--------------------------+--------------+
| value           | char     | hyper-parameter value    | 0.91         |
+-----------------+----------+--------------------------+--------------+
| created         | datetime | datetime when the value  | 2021-04-30   |
|                 |          | of the hyper-parameter   | 13:04:00     |
|                 |          | has been added to the    |              |
|                 |          | table                    |              |
+-----------------+----------+--------------------------+--------------+

Deprecated: Hyper-parameters are managed by MLflow.

NameToLatLon
------------
+----------------+-----------+-----------------+----------------------+
| **Name**       | **Type**  | **Comment**     | **Example**          |
+================+===========+=================+======================+
| regionInput    | chr       | region name     | Leeuwarden           |
+----------------+-----------+-----------------+----------------------+
| lon            | decimal   | longitude       | 5.800                |
+----------------+-----------+-----------------+----------------------+
| lat            | decimal   | latitude        | 53.201               |
+----------------+-----------+-----------------+----------------------+

predictions
-----------
Contains prediction jobs.

+--------------------+----------+------------------------+-----------------+
| **Name**           | **Type** | **Comment**            | **Example**     |
+====================+==========+========================+=================+
| id                 | int      | prediction job id      | 313             |
+--------------------+----------+------------------------+-----------------+
| name               | chr      | customer name          | Location_A      |
+--------------------+----------+------------------------+-----------------+
| typ                | chr      | type of forecast       | demand          |
+--------------------+----------+------------------------+-----------------+
| model              | chr      | type of model          | xgb             |
+--------------------+----------+------------------------+-----------------+
| created            | datetime | creation datetime of   | 2019-05-16      |
|                    |          | the prediction job     | 14:53:38        |
+--------------------+----------+------------------------+-----------------+
| active             | int      | 0 = off; 1 = on; 2 =   | 1               |
|                    |          | automatic              |                 |
+--------------------+----------+------------------------+-----------------+
| horizon_minutes    | int      | max forecast horizon   | 2880            |
|                    |          | (minutes)              |                 |
+--------------------+----------+------------------------+-----------------+
| resolution_minutes | int      | time resolution of     | 15              |
|                    |          | forecasts (minutes)    |                 |
+--------------------+----------+------------------------+-----------------+
| train_components   | bool     | use splitting          | 1               |
|                    |          | components ? (now in a |                 |
|                    |          | separate pipeline:     |                 |
|                    |          | component forecast)    |                 |
+--------------------+----------+------------------------+-----------------+
| ean                | chr      | id of the connection   | 000             |
|                    |          | point for Tennet       | 000000000000003 |
+--------------------+----------+------------------------+-----------------+

**Prediction**: A prediction is the core concept in openSTEF and largley translate to the prediction_job in the openSTEF code. To make a prediction a prediction is usualy coupled to one or more systems. These systems provide the measurement data for which a forecast is made.

predictions_quantiles_sets
--------------------------
Correspondence table between prediction jobs and the set of quantiles to
forecast.

+-----------------+---------+--------------------------+-----------------+
| **Name**        | **Type**| **Comment**              | **Example**     |
+=================+=========+==========================+=================+
| id              | int     |                          | 22              |
+-----------------+---------+--------------------------+-----------------+
| prediction_id   | int     | prediction job id        | 313             |
+-----------------+---------+--------------------------+-----------------+
| quantile_set_id | int     | id of the quantile sets  | 1               |
+-----------------+---------+--------------------------+-----------------+

predictions_systems
-------------------
Correspondence table between prediction jobs and systems.

+------------------+----------+---------------------+---------------------+
| **Name**         | **Type** | **Comment**         | **Example**         |
+==================+==========+=====================+=====================+
| prediction_id    | int      | prediction job id   | 317                 |
+------------------+----------+---------------------+---------------------+
| system_id        | chr      | system id           | Location_A_System_1 |
+------------------+----------+---------------------+---------------------+

-  A **prediction job** can correspond to multiple **systems** (e.g. pj
   #459)

-  A **system** can be linked to multiple **prediction jobs** (e.g.
   Location_A_System_1 for pj #317 and #459)

**System** : Represents a physical measurement system. All metadata is saved in this SQL table, the actual timeseries can be retrieved from influx by the corresponding system id.  

quantiles_sets
---------------
Contains the specifications of the quantile sets.

+----------------+----------+------------------+-------------------------+
| **Name**       | **Type** | **Comment**      | **Example**             |
|                |          |                  |                         |
+================+==========+==================+=========================+
| id             | int      | quantile set id  |                         |
+----------------+----------+------------------+-------------------------+
| quantiles      | json     | list of          | [0.05, 0.1, 0.3, 0.5,   |
|                |          | quantiles        | 0.7, 0.9, 0.95]         |
+----------------+----------+------------------+-------------------------+
| description    | chr      |                  | Default quantile set    |
+----------------+----------+------------------+-------------------------+

solarspecs
----------
Configuration for PV forecasts for each prediction job

+------------+----------+------------------------------------+-------------+
| **Name**   | **Type** | **Comment**                        | **Example** |
+============+==========+====================================+=============+
| pid        | int      | prediction job id                  |             |
+------------+----------+------------------------------------+-------------+
| lat        | double   | latitude                           |             |
+------------+----------+------------------------------------+-------------+
| lon        | double   | longitude                          |             |
+------------+----------+------------------------------------+-------------+
| radius     | int      | radius in km                       |             |
+------------+----------+------------------------------------+-------------+
| peak_power | int      | max power                          |             |
+------------+----------+------------------------------------+-------------+

2 cases:

-  Radius = ‘None’ : when the forecast is for a specific system

-  Radius > 0 when the forecast is for a region

This table is empty in openstef-reference.

Cf openstef/tasks/create_solar_forecast.py and get_solar_input in
openstef_dbc/services/model_input.py

systems
---------
Contains informations about **systems**.

+----------------------------------+----------+---------------+----------------+
| **Name**                         | **Type** | **Comment**   | **Example**    |
+==================================+==========+===============+================+
| sid                              | chr      | system id     | Locat          |
|                                  |          |               | ion_A_System_1 |
+----------------------------------+----------+---------------+----------------+
| origin                           | chr      | origin of the | ems (energy    |
|                                  |          | system data   | management     |
|                                  |          |               | system =       |
|                                  |          |               | SCADA)         |
+----------------------------------+----------+---------------+----------------+
| lat                              | double   | latitude      | 5.837          |
+----------------------------------+----------+---------------+----------------+
| lon                              | double   | longitude     | 51.813         |
+----------------------------------+----------+---------------+----------------+
| region                           | chr      |               |                |
+----------------------------------+----------+---------------+----------------+
| timezone                         | chr      |               |                |
+----------------------------------+----------+---------------+----------------+
| brand                            | chr      | additional    |                |
|                                  |          | information   |                |
|                                  |          | on            |                |
|                                  |          | measurements  |                |
+----------------------------------+----------+---------------+----------------+
| freq                             | int      | additional    |                |
|                                  |          | information   |                |
|                                  |          | on            |                |
|                                  |          | measurements  |                |
+----------------------------------+----------+---------------+----------------+
| qual                             | float    | additional    |                |
|                                  |          | information   |                |
|                                  |          | on            |                |
|                                  |          | measurements  |                |
+----------------------------------+----------+---------------+----------------+
| lag                              | float    | additional    |                |
|                                  |          | information   |                |
|                                  |          | on            |                |
|                                  |          | measurements  |                |
+----------------------------------+----------+---------------+----------------+
| created                          | datetime | system        |                |
|                                  |          | creation      |                |
|                                  |          | date          |                |
+----------------------------------+----------+---------------+----------------+
| autoupdate                       | tinyint  | ?             |                |
+----------------------------------+----------+---------------+----------------+
| polarity                         | int      | sign          | -1/1           |
|                                  |          | convention    |                |
|                                  |          | for           |                |
|                                  |          | production    |                |
|                                  |          | and load      |                |
+----------------------------------+----------+---------------+----------------+
| measurements_customer_api_key_id | int      | API to post   | 199            |
|                                  |          | measurements  |                |
+----------------------------------+----------+---------------+----------------+

systemsApiKeys
------------------
API key to retrieve systems measurements.

+----------------+----------------+-----------------+-------------------+
| **Name**       | **Type**       | **Comment**     | **Example**       |
+================+================+=================+===================+
| id             | int            | API key id      | 199               |
+----------------+----------------+-----------------+-------------------+
| name           | chr            |                 | Measurements      |
+----------------+----------------+-----------------+-------------------+
| apiKey         | chr            | API key value   | uuid-Measurements |
+----------------+----------------+-----------------+-------------------+

todolist
---------
This table is empty in openstef-reference.

+----------------------+-------------+----------------+----------------+
| **Name**             | **Type**    | **Comment**    | **Example**    |
+======================+=============+================+================+
| id                   | int         | id of the job  |                |
+----------------------+-------------+----------------+----------------+
| created              | datetime    |                |                |
+----------------------+-------------+----------------+----------------+
| function             | chr         | functions to   |                |
|                      |             | execute        |                |
+----------------------+-------------+----------------+----------------+
| args                 | chr         | arguments of   |                |
|                      |             | the functions  |                |
+----------------------+-------------+----------------+----------------+
| inprogress           | int         |                |                |
+----------------------+-------------+----------------+----------------+

Manually add a task besides those scheduled by Kubernetes. The list is
automatically checked by Kubernetes.

weatherforecastlocations
------------------------
Contains the locations of the weather stations. Not used in OpenSTEF.

+----------------+----------------+-----------------+-----------------+
| **Name**       | **Type**       | **Comment**     | **Example**     |
+================+================+=================+=================+
| created        | datetime       |                 | 2023-06-08      |
|                |                |                 | 18:26:44        |
+----------------+----------------+-----------------+-----------------+
| input_city     | chr            |                 | Deelen          |
+----------------+----------------+-----------------+-----------------+
| lat            | double         |                 | 52.067          |
+----------------+----------------+-----------------+-----------------+
| lon            | double         |                 | 5.8             |
+----------------+----------------+-----------------+-----------------+
| country        | chr            |                 | NL              |
+----------------+----------------+-----------------+-----------------+
| active         | int            |                 | 1               |
+----------------+----------------+-----------------+-----------------+

windspecs
---------
This table is empty in openstef-reference. Contains the information for
the wind power forecast related to a prediction job.

+--------------+--------------+-------------------------+-------------+
| **Name**     | **Type**     | **Comment**             | **Example** |
+==============+==============+=========================+=============+
| pid          | int          | prediction job id       |             |
+--------------+--------------+-------------------------+-------------+
| lat          | double       |                         |             |
+--------------+--------------+-------------------------+-------------+
| lon          | double       |                         |             |
+--------------+--------------+-------------------------+-------------+
| turbine_type | chr          | corresponds to the      |             |
|              |              | field ‘name’ in         |             |
|              |              | genericpowercurves      |             |
+--------------+--------------+-------------------------+-------------+
| n_turbines   | int          | number of wind turbines |             |
+--------------+--------------+-------------------------+-------------+
| hub_height   | int          | height of the turbines  |             |
|              |              | (m)                     |             |
+--------------+--------------+-------------------------+-------------+

The hub height is used to extrapolate the wind speed forecast at the
correct height.

Cf calculate_windspeed_at_hubheight in
openstef/feature_engineering/weather_features.py.
