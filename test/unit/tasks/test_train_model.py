from unittest import TestCase
from unittest.mock import MagicMock, patch

from openstf.tasks.train_model import train_model_task

from test.utils import TestData

FORECAST_MOCK = 'forecast_mock'

class TestTrainModelTask(TestCase):

    def setUp(self) -> None:
        self.pj = TestData.get_prediction_job(pid=307)

    @patch('openstf.tasks.train_model.train_model_pipeline')
    def test_create_train_model_task_happy_flow(self, train_model_pipeline_mock):
        # Test happy flow of create forecast task
        context = MagicMock()
        train_model_task(self.pj, context)

        self.assertEqual(train_model_pipeline_mock.call_count, 1)
        self.assertEqual(train_model_pipeline_mock.call_args_list[0][0][0]['id'], self.pj['id'])