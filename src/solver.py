"""Description: This file contains the solvers for the fault tree and Markov chain models."""
import networkx as nx
import numpy as np
from scipy import linalg
import re
import math


def solve_ft(ft_graph, ft_object):
    """Bottom up fault tree solver"""
    result_map = {}
    ft_gates = nx.get_node_attributes(ft_graph, 'gate_type')
    ft_probs = nx.get_node_attributes(ft_graph, 'failure_prob')
    ft_basic_events = ft_object.get_basic_events()
    ft_top_event = ft_object.get_top_event()
    successor_dict = nx.dfs_successors(ft_graph, ft_top_event)
    result_map = solve_dfs(successor_dict, ft_gates, ft_basic_events, result_map)
    for element, value in result_map.items():
        for node in ft_probs:
            if node == element:
                ft_probs[node] = value
    if ft_top_event in result_map:
        ft_object.set_top_event_failure_prob(result_map[ft_top_event])


def solve_dfs(successor_dict, ft_gates, ft_basic_events, result_map):
    """Depth first search algorithm for the fault tree solver"""
    tmp_var = 0
    if successor_dict:
        for node, edges in successor_dict.items():
            for edge in edges:
                if edge in ft_gates:
                    if edge not in result_map:
                        tmp_var = 100
            if tmp_var == 100:
                tmp_var = 0
                continue
            else:
                gate_type = ft_gates[node]
                result_map[node] = solve_ft_gate(edges, gate_type, ft_basic_events, result_map)
                if node in successor_dict:
                    del successor_dict[node]
                break
        solve_dfs(successor_dict, ft_gates, ft_basic_events, result_map)
    return result_map


def solve_ft_gate(edges, gate_type, basic_events, result_map):
    """Solves the gate type of the fault tree"""
    result_and = 1
    result_or = 0
    if gate_type == 'AND':
        for edge in edges:
            if edge in basic_events:
                value = basic_events[edge]
            elif edge in result_map:
                value = result_map[edge]
            else:
                print("Error: Failure prob of node not found")
                return False
            result_and = result_and * value
        return result_and
    elif gate_type == 'OR':
        for edge in edges:
            if edge in basic_events:
                value = basic_events[edge]
            elif edge in result_map:
                value = result_map[edge]
            else:
                print("Error: Failure prob of node not found")
                return False
            result_or = result_or + (1 - result_or) * value
        return result_or
    else:
        print("Error: Wrong gate type")
        return False


def solve_mc(mc_object):
    """Numerical MC solver that computes probability of absorption and time to absorption"""
    state_list = mc_object.get_states()
    absorbing_state_list = mc_object.get_absorbing_states()
    transitions = mc_object.get_transitions()
    number_of_transient_states = len(state_list)
    p_matrix = create_mc_transition_matrix(state_list, absorbing_state_list, transitions)
    q_matrix, r_matrix = create_canonical_form(p_matrix, state_list, absorbing_state_list)
    i_matrix = np.identity(number_of_transient_states)
    c_vector = np.ones((number_of_transient_states, 1))

    # compute probability of absorption
    n_matrix = i_matrix - q_matrix
    lu, piv = linalg.lu_factor(n_matrix)
    b_matrix = linalg.lu_solve((lu, piv), r_matrix)

    # compute time to absorption
    lu_1, piv_1 = linalg.lu_factor(n_matrix)
    t_matrix = linalg.lu_solve((lu_1, piv_1), c_vector)

    return b_matrix, t_matrix


def create_mc_transition_matrix(state_list, absorbing_state_list, transitions):
    """Creates the transition matrix"""
    state_list.extend(absorbing_state_list)
    if absorbing_state_list:
        print("This is an absorbing Markov chain with " + str(len(absorbing_state_list)) +
              " absorbing states.")
    else:
        print("This is not an absorbing Markov chain.")
    p = np.zeros((len(state_list), len(state_list)))
    for first_edge, values in transitions.items():
        for second_edge, transition_prob in values.items():
            index_first_edge = state_list.index(first_edge)
            index_second_edge = state_list.index(second_edge)
            p[index_first_edge][index_second_edge] = transition_prob
    return p


def create_canonical_form(transition_matrix, state_list, absorbing_state_list):
    """Converts the transition matrix into the canonical form for further computations"""
    state_list.extend(absorbing_state_list)
    absorbing_indices = []
    transient_indices = []
    for state in state_list:
        if state in absorbing_state_list:
            absorbing_indices.append(state_list.index(state))
        else:
            transient_indices.append(state_list.index(state))

    q_matrix = np.delete(transition_matrix, absorbing_indices, axis=0)
    q_matrix = np.delete(q_matrix, absorbing_indices, axis=1)
    r_matrix = np.delete(transition_matrix, transient_indices, axis=1)
    r_matrix = np.delete(r_matrix, absorbing_indices, axis=0)

    return q_matrix, r_matrix


def hybrid_solver(ft_dict, mc_object):
    """Hybrid risk model solver. Solves first all FTs, then the MC"""
    ft_result_dict = {}
    mc_transitions = mc_object.get_transitions()
    for ft, ft_elements in ft_dict.items():
        ft_object = ft_elements[0]
        ft_graph = ft_elements[1]
        solve_ft(ft_graph, ft_object)

    for ft, ft_elements in ft_dict.items():
        ft_object = ft_elements[0]
        top_event = ft_object.get_top_event()
        ft_result_dict[top_event] = float(ft_object.get_top_event_failure_prob())

    for from_state, elements in mc_transitions.items():
        for to_state, value in list(elements.items()):
            prob = _eval_transition_expr(value, ft_result_dict)
            elements[to_state] = prob

    b_matrix, t_matrix = solve_mc(mc_object)
    return b_matrix, t_matrix


_NUM = r'(?P<num>[0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)'
_TOK = r'(?P<tok>[A-Za-z_][A-Za-z0-9_]*)'


def _eval_transition_expr(value, ft_result_dict):
    """Evaluate simple expressions using FT results."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == '1':
        return 1.0

    # direct token: "<TopEvent>"
    if s in ft_result_dict:
        return float(ft_result_dict[s])

    # (1 - <TopEvent>)  or  1 - <TopEvent>
    m = re.fullmatch(r'\(?\s*1\s*-\s*' + _TOK + r'\s*\)?', s)
    if m:
        tok = m.group('tok')
        if tok not in ft_result_dict:
            raise ValueError(f"Unknown FT token '{tok}' in '{s}'")
        return 1.0 - float(ft_result_dict[tok])

    # k * (1 - <TopEvent>)
    m = re.fullmatch(_NUM + r'\s*\*\s*\(?\s*1\s*-\s*' + _TOK + r'\s*\)?', s)
    if m:
        k = float(m.group('num'))
        tok = m.group('tok')
        return k * (1.0 - float(ft_result_dict[tok]))

    # (1 - <TopEvent>) * k
    m = re.fullmatch(r'\(?\s*1\s*-\s*' + _TOK + r'\s*\)?\s*\*\s*' + _NUM, s)
    if m:
        tok = m.group('tok')
        k = float(m.group('num'))
        return k * (1.0 - float(ft_result_dict[tok]))

    # k * <TopEvent>
    m = re.fullmatch(_NUM + r'\s*\*\s*' + _TOK, s)
    if m:
        k = float(m.group('num'))
        tok = m.group('tok')
        return k * float(ft_result_dict[tok])

    # <TopEvent> * k
    m = re.fullmatch(_TOK + r'\s*\*\s*' + _NUM, s)
    if m:
        tok = m.group('tok')
        k = float(m.group('num'))
        return k * float(ft_result_dict[tok])

    raise ValueError(f"Unrecognized transition expression: '{s}'")