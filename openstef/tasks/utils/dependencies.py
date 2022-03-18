# SPDX-FileCopyrightText: 2017-2022 Contributors to the OpenSTEF project <korte.termijn.prognoses@alliander.com> # noqa E501>
#
# SPDX-License-Identifier: MPL-2.0
import random
from typing import Sequence
from ...data_classes.prediction_job import PredictionJobDataClass
import networkx as nx


def has_dependencies(pjs: Sequence[PredictionJobDataClass]) -> bool:
    for pj in pjs:
        if pj.depends_on is not None and len(pj.depends_on) > 0:
            return True
    return False


def build_graph_structure(pjs):
    nodes = set()
    edges = set()

    for pj in pjs:
        nodes.add(pj["id"])
        if pj.depends_on is not None:
            for j in pj.depends_on:
                edges.add((j, pj["id"]))

    return nodes, edges


def build_nx_graph(nodes, edges):
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph


def find_groups(pjs, randomize_groups=False):
    nodes, edges = build_graph_structure(pjs)
    graph = build_nx_graph(nodes, edges)
    groups = list(nx.topological_generations(graph))

    if randomize_groups:
        groups = [random.shuffle(group) for group in groups]

    # Convert groups of pj ids to groups of pjs
    pj_id_map = {pj["id"]: i for i, pj in enumerate(pjs)}
    pj_groups = [
        [pjs[pj_id_map[pj_id]] for pj_id in group] for group in groups
    ]
    return graph, pj_groups
