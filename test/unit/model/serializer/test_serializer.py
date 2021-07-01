# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta
from pathlib import Path
from test.utils import BaseTestCase

from openstf.model.serializer import (
    PersistentStorageSerializer,
    MODEL_FILENAME,
    FOLDER_DATETIME_FORMAT,
)


class TestAbstractModelSerializer(BaseTestCase):
    def test_determine_model_age_from_path(self):
        expected_model_age = 7

        model_datetime = datetime.utcnow() - timedelta(days=expected_model_age)

        model_path = (
            Path(f"{model_datetime.strftime(FOLDER_DATETIME_FORMAT)}") / MODEL_FILENAME
        )

        model_age = PersistentStorageSerializer(
            trained_models_folder=""
        )._determine_model_age_from_path(model_path)

        self.assertEqual(model_age, expected_model_age)
