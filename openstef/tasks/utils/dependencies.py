# SPDX-FileCopyrightText: 2017-2023 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import random
from typing import Iterable, Sequence, Set, Union

import networkx as nx

from openstef.data_classes.prediction_job import PredictionJobDataClass

NodeIdType = Union[str, int]
EdgeType = tuple[NodeIdType, NodeIdType]


def has_dependencies(pjs: Iterable[PredictionJobDataClass]) -> bool:
    """Test whether some prediction jobs have dependencies information.

    Args:
        pjs: The list of prediction jobs

    Returns:
        True if some dependency information was found.

    """
    for pj in pjs:
        if pj.depends_on is not None and len(pj.depends_on) > 0:
            return True
    return False


def build_graph_structure(
    pjs: Iterable[PredictionJobDataClass],
) -> tuple[Set[NodeIdType], Set[EdgeType]]:
    """Build the graph of dependencies between prediction jobs.

    Args:
        pjs: The Iterable of prediction jobs

    Returns:
        - The set of node ids of the graph
        - The set of edges in the graph

    """
    nodes = set()
    edges = set()

    for pj in pjs:
        nodes.add(pj["id"])
        if pj.depends_on is not None:
            for j in pj.depends_on:
                edges.add((j, pj["id"]))

    return nodes, edges


def build_nx_graph(
    nodes: Iterable[NodeIdType], edges: Iterable[EdgeType]
) -> nx.DiGraph:
    """Build a networkx Directed Graph.

    Args:
        nodes: The sequence of node ids
        edges: The sequence of edges

    Returns:
        The dependency graph

    """
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def find_groups(
    pjs: Sequence[PredictionJobDataClass], randomize_groups: bool = False
) -> tuple[nx.DiGraph, list[list[PredictionJobDataClass]]]:
    """Find a sequence of prediction job groups respecting dependencies.

    Compute groups of prediction jobs such that the prediction jobs in a group
    depend of at least one prediction job in the previous group and does not depend
    on a prediction job in the following groups.
    This means that all the prediction jobs in a group can be run in parallel and that
    if groups are treated in the given order, the dependencies of a prediction job have
    already been treated when the prediction job is run.

    Args:
        pjs: The sequence of prediction jobs
        randomize_groups: Wether subgroups should be randomized.

    Returns:
        - The dependency graph
        - The list of prediction job groups

    """
    nodes, edges = build_graph_structure(pjs)
    graph = build_nx_graph(nodes, edges)
    groups = list(nx.topological_generations(graph))

    if randomize_groups:
        for group in groups:
            random.shuffle(group)

    # Convert groups of pj ids to groups of pjs
    pj_id_map = {pj["id"]: i for i, pj in enumerate(pjs)}
    pj_groups = [[pjs[pj_id_map[pj_id]] for pj_id in group] for group in groups]
    return graph, pj_groups
