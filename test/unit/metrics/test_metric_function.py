import unittest
from openstf.metrics.metrics import get_eval_metric_function, mae


class TestEvalMetricFunction(unittest.TestCase):
    def test_eval_metric(self):
        self.assertEqual(get_eval_metric_function("mae"), mae)  # add assertion here

    def test_eval_metric_exception(self):
        with self.assertRaises(KeyError):
            get_eval_metric_function("non-existing")


if __name__ == "__main__":
    unittest.main()
