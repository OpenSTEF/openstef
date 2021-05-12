import unittest
from unittest.mock import MagicMock, patch
from test.utils import TestData
from test.utils import BaseTestCase

from openstf.pipeline.train_model_sklearn import train_model_pipeline


# define constants

PJ = TestData.get_prediction_job(pid=307)
XGB_HYPER_PARAMS = {
    "subsample": 0.9,
    "min_child_weight": 4,
    "max_depth": 8,
    "gamma": 0.5,
    "colsample_bytree": 0.85,
    "eta": 0.1,
    "training_period_days": 90,
}


class TestTrainModel(BaseTestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == "__main__":
    unittest.main()
