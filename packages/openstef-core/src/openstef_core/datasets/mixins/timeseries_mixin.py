from abc import abstractmethod
from datetime import timedelta

import pandas as pd


class TimeSeriesMixin:
    """Abstract base class for time series dataset functionality.

    This mixin defines the essential interface that all time series datasets
    must implement. It provides access to feature metadata, temporal properties,
    and the dataset's temporal index.

    Classes implementing this mixin must set the required attributes in their __init__ method.
    This interface enables consistent access patterns across different time series
    dataset implementations.

    Attributes:
        feature_names: Names of all available features, excluding metadata columns.
        sample_interval: The fixed interval between consecutive samples.
        index: Datetime index representing all timestamps in the dataset.
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]: ...

    @property
    @abstractmethod
    def sample_interval(self) -> timedelta: ...

    @property
    @abstractmethod
    def index(self) -> pd.DatetimeIndex: ...
