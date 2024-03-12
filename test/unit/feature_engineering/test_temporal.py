import numpy as np
import pandas as pd

from openstef.feature_engineering.temporal_features import (
    add_time_of_the_day_cyclic_features,
)
from test.unit.utils.base import BaseTestCase


class TestTemporalFeaturesModule(BaseTestCase):
    def test_add_time_of_the_day_cyclic_features(self):
        # Two days of data every 15 minutes
        num_points = int(24 * 60 / 15 * 2)

        input_data = pd.DataFrame(
            index=pd.date_range(
                start="2023-01-01 00:00:00", freq="15T", periods=num_points
            ),
        )
        output_data = add_time_of_the_day_cyclic_features(input_data)

        # Two subsequent periods ranging from 0 to 2 * pi
        periods = (
            np.linspace(start=0.0, stop=2.0, num=num_points, endpoint=False)
            % 1.0
            * 2
            * np.pi
        )

        assert np.allclose(output_data["time_of_day_sine"], np.sin(periods))
        assert np.allclose(output_data["time_of_day_cosine"], np.cos(periods))
