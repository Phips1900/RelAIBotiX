"""
Dynamic Reliability Assessment graph generator
@author Philipp Grimmeisen
@version 1.0
@date 01.08.2024
"""

import networkx as nx
import matplotlib.pyplot as plt


def create_mc_graph(mc_object):
    """@brief creates a MC graph"""
    mc_graph = nx.DiGraph()
    mc_states = mc_object.get_states()
    mc_states.extend(mc_object.get_absorbing_states())
    mc_graph.add_nodes_from(mc_states)
    mc_edges = mc_object.get_edges()
    mc_transitions = mc_object.get_transitions()
    for from_state, x in mc_transitions.items():
        for to_state, value in x.items():
            mc_graph.add_edge(from_state, to_state, transition_prob=value)
    return mc_graph


def create_ft_graph(ft_object):
    """@brief creates a ft graph"""
    ft_graph = nx.DiGraph()
    ft_basic_events = ft_object.get_basic_events()
    ft_gates = ft_object.get_gates()
    for be, prob in ft_basic_events.items():
        ft_graph.add_node(be, failure_prob=prob)
    for element, x in ft_gates.items():
        for gate, be in x.items():
            ft_graph.add_node(element, failure_prob="", gate_type=gate)
    for element, x in ft_gates.items():
        for gate, be in x.items():
            for basic_event in be:
                ft_graph.add_edge(element, basic_event)
    return ft_graph


def draw_graph(graph):
    """@brief draws a graph"""
    nx.draw(graph, with_labels=True)
    plt.show()