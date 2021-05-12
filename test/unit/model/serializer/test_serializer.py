# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import unittest
from datetime import datetime, timedelta
from pathlib import Path

from openstf.model.serializer import PersistentStorageSerializer, \
    MODEL_FILENAME, FOLDER_DATETIME_FORMAT


class TestAbstractModelSerializer(unittest.TestCase):
    def test_determine_model_age_from_path(self):
        expected_model_age = 7

        model_datetime = datetime.utcnow() - timedelta(days=expected_model_age)

        model_location = (
            Path(
                f"{model_datetime.strftime(FOLDER_DATETIME_FORMAT)}"
            )

        )

        model_age = PersistentStorageSerializer({'id':454})._determine_model_age_from_path(model_location)

        self.assertEqual(model_age, expected_model_age)
