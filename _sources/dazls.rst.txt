.. comment:
    SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com>
    SPDX-License-Identifier: MPL-2.0

Domain Adaptation for Zero Shot Learning in Sequence (DAZLS)
============================================================

DAZLS is an energy splitting function in OpenSTEF. Is a technique which
transfers knowledge from complete-information substations to
incomplete-information substations for solar and wind power prediction.
It is being used in :mod:`openstef.pipeline.create_component_forecast`
to issue the prediction.

This function trains a splitting model on data from multiple substations
with known components and uses this model to carry out a prediction for
target substations with unknown components. The training data from the
known substations include weather, location, and total load information
of each substation and predicts the solar and the wind power of the
target substations.

The model is developed as a zero-shot learning method because it has to
carry out the prediction of target substations with unknown components
by using training data from other substations with known components. For
this purpose, the method is formulated as a 2-step approach by combining
two models deployed in sequence, the Domain and the Adaptation model.

The schema bellow depicts the structure of the DAZLS model. The input of
the model is data from the complete-information substations. For every
known substation we have input data, source metadata and output data. At
first, we feed the input data to train the Domain model. Domain model
gives a predicted output. This predicted output data, linked together
with the source metadata of each substation, is being used as the input
to train the Adaptation model. Then, the Adaptation model provides the
final output prediction of solar and wind power for the target
substations.

.. figure:: https://user-images.githubusercontent.com/66070103/189650328-377ebb79-e8a7-40c6-acf3-64a5bb6197a4.jpg
   :alt: Presentation3

   Domain Adaptation Model

For more information about DAZLS model, see:

Teng, S.Y., van Nooten, C.C., van Doorn, J.M., Ottenbros, A., Huijbregts, M., Jansen, J.J.
Improving Near Real-Time Predictions of Renewable Electricity Production
at Substation Level (Submitted)

HOW TO USE: The code which loads and stores the DAZLS model is in the
notebook file
`05. Split net load into Components.ipynb <https://github.com/OpenSTEF/openstef-offline-example/tree/master/examples/05.%20Split%20net%20load%20into%20Components>`__.
When running this notebook, a dazls_stored.sav file is being produced
and can be used in the prediction pipeline. It is important, whenever
there are changes in the
`dazls.py <https://github.com/OpenSTEF/openstef/blob/main/openstef/model/regressors/dazls.py>`__,
to run again the notebook and use the
newly produced dazls_stored.sav file in the repository.
