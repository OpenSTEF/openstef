import pandas as pd
import pytest

from openstef_meta.utils.decision_tree import Decision, DecisionTree, Node, Rule


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    data = {
        "feature_1": [1, 2, 3, 4, 5],
        "feature_2": [10, 20, 30, 40, 50],
    }
    return pd.DataFrame(data)


@pytest.fixture
def simple_decision_tree() -> DecisionTree:
    nodes: list[Node] = [
        Rule(
            idx=0,
            rule_type="less_than",
            feature_name="feature_1",
            threshold=3,
            next_true=1,
            next_false=2,
        ),
        Decision(idx=1, decision="Class_A"),
        Decision(idx=2, decision="Class_B"),
    ]
    return DecisionTree(nodes=nodes, outcomes={"Class_A", "Class_B"})


def test_decision_tree_prediction(sample_dataset: pd.DataFrame, simple_decision_tree: DecisionTree):

    decisions = sample_dataset.apply(simple_decision_tree.get_decision, axis=1)

    expected_decisions = pd.Series(
        ["Class_A", "Class_A", "Class_B", "Class_B", "Class_B"],
    )

    pd.testing.assert_series_equal(decisions, expected_decisions)
