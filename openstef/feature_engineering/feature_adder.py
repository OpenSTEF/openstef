# SPDX-FileCopyrightText: 2017-2023 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
"""This module provides functionality for defining custom feature adders."""
import inspect
import re
from abc import ABC, abstractmethod
from collections import Counter, namedtuple
from importlib import import_module
from typing import Optional, Sequence

import pandas as pd

ParsedFeature = namedtuple("ParsedFeature", ["name", "params"])


class FeatureAdder(ABC):
    """Abstract class that implement the FeatureAdder interface.

    It is the basic block that handles the logic for computing the specific feature and the syntactic sugar to load
    properly the feature adder according to the feature name.

    """

    @property
    @abstractmethod
    def _regex(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the FeatureAdder."""

    @abstractmethod
    def required_features(self, feature_names) -> list[str]:
        """List of features that are required to calculate this feature."""

    def __hash__(self):
        """Genearate hash of the name of this feature."""
        return hash(self.name)

    def parse_feature_name(self, feature_name: str) -> Optional[dict[str, str]]:
        """Parse a feature name.

        If the feature name is taken in charge by the feature adder, the method returns
        a dictionnary with the potentially parsed parameters contained the feature name. In the
        case the feature name does not contain parameters an empty dictionary is returned.
        Otherwise the method returns None.

        Args:
            feature_name (str): The feature name, this may contain parameter informations.

        Returns:
            Optional[dict[str, Any]]: The parsed parameters. If the feature name is recognized but has no parameters
            an empty dictionnary is returned. If the feature name is not recognized, None is
            returned.

        """
        reg = self._regex
        match = re.match(reg, feature_name)
        return None if match is None else match.groupdict()

    @abstractmethod
    def apply_features(
        self, df: pd.DataFrame, parsed_feature_names: Sequence[ParsedFeature]
    ) -> pd.DataFrame:
        """Apply or add the features to the input dataframe."""

    def __repr__(self):
        """Represent as string."""
        return "%s(<%s>)" % (self.__class__.__name__, self.name)


class FeatureDispatcher:
    """Orchestrator of the feature adders.

    It scans the feature_names to assign to each feature the proper feature adder and launch the effective computing of
    the features.

    """

    def __init__(self, feature_adders: Sequence[FeatureAdder]):
        """Initialize feature dispatcher."""
        self.feature_adders = list(feature_adders)
        self._check_feature_adder_names_unicity()

    def _check_feature_adder_names_unicity(self):
        names = Counter(adder.name for adder in self.feature_adders)
        duplicated_names = []
        for name, count in names.items():
            if count > 1:
                duplicated_names.append(name)

        if len(duplicated_names) > 0:
            raise RuntimeError(
                "There is at least one duplicated feature adder name: %s"
                % duplicated_names
            )

    def dispatch_features(
        self, feature_names: list[str]
    ) -> dict[FeatureAdder, list[ParsedFeature]]:
        """Dispatch features.

        Args:
            feature_names: The names of the features to be dispatched.

        Returns:
            Dictionary with parsed features.

        """
        recognized_features = set()
        dispatched_features = {}

        for feature_name in feature_names:
            for adder_obj in self.feature_adders:
                parsed_params = adder_obj.parse_feature_name(feature_name)
                if parsed_params is not None:
                    if feature_name in recognized_features:
                        raise RuntimeError(
                            "Ambiguous feature adder set detected. The feature name"
                            " '%s' is recognised by more than 1 feature adder"
                            % feature_names
                        )
                    recognized_features.add(feature_name)
                    features = dispatched_features.setdefault(adder_obj, [])
                    features.append(ParsedFeature(feature_name, parsed_params))

        return dispatched_features

    def apply_features(
        self, df: pd.DataFrame, feature_names: list[str]
    ) -> pd.DataFrame:
        """Applies features to the input DataFrame.

        Args:
            df: DataFrame to which the features have to be added.
            feature_names: Names of the features.

        Returns:
            DataFrame with the added features.

        """
        if feature_names is None:
            return df
        dispatched_features = self.dispatch_features(feature_names)

        applied_features = set()
        applied_features_num = 0

        while True:
            for adder, parsed_features in dispatched_features.items():
                parsed_feature_names = [pf.name for pf in parsed_features]
                required_features = adder.required_features(parsed_feature_names)

                if len(set(required_features) - set(df.columns)) == 0:
                    df = adder.apply_features(df, parsed_features)
                    applied_features |= set(parsed_feature_names)

            if (
                len(applied_features) == applied_features_num
            ):  # No new feature was treated
                break

            applied_features_num = len(applied_features)

        return df


def adders_from_module(module_name: str) -> list[FeatureAdder]:
    """Load all FeatureAdders classes on the fly from the module.

    Args:
        module_name: The name of the module from which to import.

    Returns:
        A list with all loaded FeatureAdders.

    """
    module = import_module(module_name)
    feature_adders = []

    for element_name in dir(module):
        element = getattr(module, element_name)
        if (
            isinstance(element, type)
            and issubclass(element, FeatureAdder)
            and not inspect.isabstract(element)
        ):
            feature_adders.append(element())

    return feature_adders


def adders_from_modules(module_names: list[str]) -> list[FeatureAdder]:
    """Load all FeatureAdders classes on the fly from multiple modules.

    Args:
        module_names: A list with names of the modules from which to import.

    Returns:
        A list with all loaded FeatureAdders.

    """
    return sum((adders_from_module(module_name) for module_name in module_names), [])
