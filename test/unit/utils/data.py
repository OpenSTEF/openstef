# SPDX-FileCopyrightText: 2017-2021 Contributors to the OpenSTF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0

import importlib
import json
import pickle
from pathlib import Path
from typing import Union

import pandas as pd
from openstef_dbc.services.prediction_job import PredictionJobDataClass

from openstef.data_classes.model_specifications import ModelSpecificationDataClass


class TestData:

    DATA_FILES_FOLDER = Path(__file__).parent.parent / "data"
    TRAINED_MODELS_FOLDER = Path(__file__).parent.parent / "trained_models"

    LAG_FUNCTIONS_KEYS = [
        "T-1d",
        "T-2d",
        "T-3d",
        "T-4d",
        "T-5d",
        "T-6d",
        "T-7d",
        "T-8d",
        "T-9d",
        "T-10d",
        "T-11d",
        "T-12d",
        "T-13d",
        "T-14d",
    ]

    _FILE_READER_KWARGS = {}
    _CSV_DEFAULT_KWARGS = {"index_col": 0, "parse_dates": True}

    _PY_ATTRIBUTE = {}
    _PY_DEFAULT_ATTRIBUTE = "data"

    @classmethod
    def save(cls, obj, filename):
        filepath = cls.DATA_FILES_FOLDER / filename
        if ".pickle" in filename:
            with open(filepath, "wb") as fh:
                pickle.dump(obj, fh)

    @classmethod
    def load(cls, filename):
        filepath = cls.DATA_FILES_FOLDER / filename
        if ".csv" in filename:
            reader = pd.read_csv
            reader_kwargs = cls._FILE_READER_KWARGS.get(
                filename, cls._CSV_DEFAULT_KWARGS
            )
            return reader(filepath, **reader_kwargs)

        if ".pickle" in filename:
            with open(filepath, "rb") as fh:
                data = pickle.load(fh)
            if filename == "input_data.pickle":
                data.index.freq = "15T"
            return data

        if ".json" in filename:
            reader = json.load
            with open(filepath, "r") as fp:
                return reader(fp)

        if ".py" in filename:
            module_name = f"test.unit.data.{filename.split('.py')[0]}"
            reader = importlib.import_module
            module = reader(module_name)
            return getattr(
                module, cls._PY_ATTRIBUTE.get(filename, cls._PY_DEFAULT_ATTRIBUTE)
            )

        raise ValueError(f"File type not support: {filename}")

    @classmethod
    def get_prediction_job(cls, pid: Union[int, str]):
        with open(cls.DATA_FILES_FOLDER / "prediction_jobs.json", "r") as fh:
            prediction_jobs = json.load(fh, object_hook=prediction_job_decoder)

            out_dict = prediction_jobs[str(pid)]

            # Change the typ column to forecast_type normally done after query
            out_dict["forecast_type"] = out_dict.pop("typ")

        return PredictionJobDataClass(**out_dict)

    @classmethod
    def get_prediction_job_and_modelspecs(cls, pid: Union[int, str]):
        with open(cls.DATA_FILES_FOLDER / "prediction_jobs.json", "r") as fh:
            prediction_jobs = json.load(fh, object_hook=prediction_job_decoder)

            out_dict = prediction_jobs[str(pid)]

            # Change the typ column to forecast_type normally done after query
            out_dict["forecast_type"] = out_dict.pop("typ")

        return (
            PredictionJobDataClass(**out_dict),
            ModelSpecificationDataClass(**out_dict),
        )

    @classmethod
    def get_prediction_jobs(cls):
        with open(cls.DATA_FILES_FOLDER / "prediction_jobs.json", "r") as fh:
            prediction_jobs = json.load(fh, object_hook=prediction_job_decoder)

        prediction_jobs_list = []
        for v in prediction_jobs.values():
            # Change the typ column to forecast_type normally done after query
            v["forecast_type"] = v.pop("typ")

            prediction_jobs_list.append(PredictionJobDataClass(**v))
        return prediction_jobs_list


def prediction_job_decoder(dct):
    if "created" in dct:
        dct["created"] = pd.Timestamp(dct["created"])
    return dct
