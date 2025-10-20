from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Concatenate, Self

import pandas as pd


class DatasetMixin(ABC):
    """Abstract base class for dataset persistence operations.

    This mixin defines the interface for saving and loading datasets to/from
    parquet files. It ensures datasets can be persisted with all their metadata
    and reconstructed exactly as they were saved.

    Classes implementing this mixin must:
    - Save all data and metadata necessary for complete reconstruction
    - Store metadata in parquet file attributes using attrs
    - Handle missing metadata gracefully with sensible defaults when loading

    See Also:
        TimeSeriesDataset: Implementation for standard time series datasets.
        VersionedTimeSeriesPart: Implementation for versioned dataset segments.
    """

    @abstractmethod
    def _to_pandas(self) -> pd.DataFrame:
        """Convert the dataset to a pandas DataFrame.

        This method provides a way to obtain the dataset's data in a standard
        pandas DataFrame format for interoperability with other libraries and tools.

        Returns:
            A pandas DataFrame representing the dataset's data.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _from_pandas(cls, df: pd.DataFrame) -> Self:
        """Create a dataset instance from a pandas DataFrame.

        This method allows for constructing the dataset from a standard
        pandas DataFrame, enabling easy integration with data sources
        that provide data in this format.

        Args:
            df: A pandas DataFrame containing the dataset's data.

        Returns:
            A new instance of the dataset constructed from the DataFrame.
        """
        raise NotImplementedError

    def to_parquet(self, path: Path) -> None:
        """Save the dataset to a parquet file.

        Stores both the dataset's data and all necessary metadata for complete
        reconstruction. Metadata should be stored in the parquet file's attrs
        dictionary.

        Args:
            path: File path where the dataset should be saved.

        See Also:
            read_parquet: Counterpart method for loading datasets.
        """
        self._to_pandas().to_parquet(path=path)

    @classmethod
    def read_parquet(cls, path: Path) -> Self:
        """Load a dataset from a parquet file.

        Reconstructs a dataset from a parquet file created with to_parquet,
        including all data and metadata. Should handle missing metadata
        gracefully with sensible defaults.

        Args:
            path: Path to the parquet file to load.

        Returns:
            New dataset instance reconstructed from the file.

        See Also:
            to_parquet: Counterpart method for saving datasets.
        """
        df = pd.read_parquet(path=path)  # type: ignore
        return cls._from_pandas(df)

    def pipe[T, **P](self, func: Callable[Concatenate[Self, P], T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Applies a function to the dataset and returns the result.

        This method allows for functional-style transformations and operations
        on the dataset, enabling method chaining and cleaner code.

        Args:
            func: A callable that takes the dataset instance and returns a value of type T.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of applying the function to the dataset.
        """
        return func(self, *args, **kwargs)
