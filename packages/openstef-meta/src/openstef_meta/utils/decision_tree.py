# SPDX-FileCopyrightText: 2025 Contributors to the OpenSTEF project <short.term.energy.forecasts@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""A simple decision tree implementation for making decisions based on feature rules."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator


class Node(BaseModel):
    """A node in the decision tree, either a rule or a decision."""

    idx: int = Field(
        description="Index of the rule in the decision tree.",
    )


class Rule(Node):
    """A single rule in the decision tree."""

    idx: int = Field(
        description="Index of the decision in the decision tree.",
    )

    rule_type: Literal["greater_than", "less_than"] = Field(
        ...,
        description="Type of the rule to apply.",
    )
    feature_name: str = Field(
        ...,
        description="Name of the feature to which the rule applies.",
    )

    threshold: float | int = Field(
        ...,
        description="Threshold value for the rule.",
    )

    next_true: int = Field(
        ...,
        description="Index of the next rule if the condition is true.",
    )

    next_false: int = Field(
        ...,
        description="Index of the next rule if the condition is false.",
    )


class Decision(Node):
    """A leaf decision in the decision tree."""

    idx: int = Field(
        description="Index of the decision in the decision tree.",
    )

    decision: str = Field(
        ...,
        description="The prediction value at this leaf.",
    )


class DecisionTree(BaseModel):
    """A simple decision tree defined by a list of rules."""

    nodes: list[Node] = Field(
        ...,
        description="List of rules that define the decision tree.",
    )

    outcomes: set[str] = Field(
        ...,
        description="Set of possible outcomes from the decision tree.",
    )

    @model_validator(mode="after")
    def validate_tree_structure(self) -> "DecisionTree":
        """Validate that the tree structure is correct.

        Raises:
            ValueError: If tree is not built correctly.

        Returns:
            The validated DecisionTree instance.
        """
        node_idx = {node.idx for node in self.nodes}
        if node_idx != set(range(len(self.nodes))):
            raise ValueError("Rule indices must be consecutive starting from 0.")

        for node in self.nodes:
            if isinstance(node, Rule):
                if node.next_true not in node_idx:
                    msg = f"next_true index {node.next_true} not found in nodes."
                    raise ValueError(msg)
                if node.next_false not in node_idx:
                    msg = f"next_false index {node.next_false} not found in nodes."
                    raise ValueError(msg)
            if isinstance(node, Decision) and node.decision not in self.outcomes:
                msg = f"Decision '{node.decision}' not in defined outcomes {self.outcomes}."
                raise ValueError(msg)

        return self

    def get_decision(self, row: pd.Series) -> str:
        """Get decision from the decision tree based on input features.

        Args:
            row: Series containing feature values.

        Returns:
            The decision outcome as a string.

        Raises:
            ValueError: If the tree structure is invalid.
            TypeError: If a node type is invalid.
        """
        current_idx = 0
        while True:
            current_node = self.nodes[current_idx]
            if isinstance(current_node, Decision):
                return current_node.decision
            if isinstance(current_node, Rule):
                feature_value = row[current_node.feature_name]
                if current_node.rule_type == "greater_than":
                    if feature_value > current_node.threshold:
                        current_idx = current_node.next_true
                    else:
                        current_idx = current_node.next_false
                elif current_node.rule_type == "less_than":
                    if feature_value < current_node.threshold:
                        current_idx = current_node.next_true
                    else:
                        current_idx = current_node.next_false
                else:
                    msg = f"Invalid rule type '{current_node.rule_type}' at index {current_idx}."
                    raise ValueError(msg)
            else:
                msg = f"Invalid node type at index {current_idx}."
                raise TypeError(msg)

    __all__ = ["Node", "Rule", "Decision", "DecisionTree"]
