# SPDX-FileCopyrightText: 2017-2022 Alliander N.V. <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from typing import List, Dict, Sequence, Optional
from collections import namedtuple, Counter
from importlib import import_module
import pandas as pd
import re
import inspect


ParsedFeature = namedtuple("ParsedFeature", ["name", "params"])


def adders_from_module(module_name: str):
    """Load all FeatureAdders classes on the fly from the module"""
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


def adders_from_modules(module_names: List[str]):
    return sum((adders_from_module(module_name) for module_name in module_names), [])


class FeatureAdder(ABC):
    """Abstract class that implement the FeatureAdder interface.
    It is the basic block that handles the logic for computing the specific feature
    and the syntactic sugar to load properly the feature adder according to the feature name."""

    @property
    @abstractmethod
    def _regex(self) -> str:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def required_features(self, feature_names) -> List[str]:
        pass

    def __hash__(self):
        return hash(self.name)

    def parse_feature_name(self, feature_name: str) -> Optional[Dict[str, str]]:
        """Parse a feature name

        If the feature name is taken in charge by the feature adder, the method returns
        a dictionnary with the potentially parsed parameters contained the feature name. In the
        case the feature name does not contain parameters an empty dictionary is returned.
        Otherwise the method returns None.

        Parameters
        ----------
        feature_name: str
            The feature name. The feature name may contain parameter informations

        Returns
        -------
        parameters: Optional[Dict[str, Any]]
            The parsed parameters. If the feature name is recognized but has no parameters
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
        pass

    def __repr__(self):
        return "%s(<%s>)" % (self.__class__.__name__, self.name)


class FeatureDispatcher:
    """Orchestrator of the feature adders.
    It scans the feature_names to assign to each feature the proper feature adder
    and launch the effective computing of the features.
    """

    def __init__(self, feature_adders: Sequence[FeatureAdder]):
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
        self, feature_names: List[str]
    ) -> Dict[FeatureAdder, List[ParsedFeature]]:
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

    def apply_features(self, df: pd.DataFrame, feature_names: List[str]):
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
