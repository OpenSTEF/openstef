# SPDX-FileCopyrightText: 2017-2021 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

from datetime import datetime
from pathlib import Path

import pandas as pd

from openstf.model.capacity.train import train_capacity_prognosis
from openstf.tasks.utils.predictionjobloop import PredictionJobLoop
from openstf.tasks.utils.taskcontext import TaskContext
from openstf.enums import MLModelType


def main():
    taskname = Path(__file__).name.replace(".py", "")

    with TaskContext(taskname) as context:
        # training horizons
        y_hor = [0, 6, 13]

        # define input range
        datetime_end = datetime.utcnow()
        datetime_start = datetime_end - pd.Timedelta("400D")

        model_type = [ml.value for ml in MLModelType]
        PredictionJobLoop(context, model_type=model_type).map(
            train_capacity_prognosis,
            datetime_start=datetime_start,
            datetime_end=datetime_end,
            y_hor=y_hor,
        )


if __name__ == "__main__":
    main()
