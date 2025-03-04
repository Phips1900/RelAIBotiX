# Description: This file contains the solvers for the fault tree and Markov chain models.
import networkx as nx
import numpy as np
from scipy import linalg


def solve_ft(ft_graph, ft_object):
    """@brief bottom up fault tree solver"""
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
    """@brief MC solver that computes probability of absorption and time to absorption"""
    state_list = mc_object.get_states()
    absorbing_state_list = mc_object.get_absorbing_states()
    transitions = mc_object.get_transitions()
    number_of_transient_states = len(state_list)
    p_matrix = create_mc_transition_matrix(state_list, absorbing_state_list, transitions)
    q_matrix, r_matrix = create_canonical_form(p_matrix, state_list, absorbing_state_list)
    i_matrix = np.identity(number_of_transient_states)
    c_vector = np.ones((number_of_transient_states, 1))

    """compute probability of absorption"""
    n_matrix = i_matrix - q_matrix
    lu, piv = linalg.lu_factor(n_matrix)
    b_matrix = linalg.lu_solve((lu, piv), r_matrix)

    """compute time to absorption"""
    lu_1, piv_1 = linalg.lu_factor(n_matrix)
    t_matrix = linalg.lu_solve((lu_1, piv_1), c_vector)

    return b_matrix, t_matrix


def create_mc_transition_matrix(state_list, absorbing_state_list, transitions):
    """@brieg creates the transition matrix"""
    state_list.extend(absorbing_state_list)
    if absorbing_state_list:
        print("This is an absorbing Markov chain with " + str(len(absorbing_state_list)) + " absorbing states.")
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
    """@brief brings the transition matrix into the canonical form for further computations"""
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


def hybrid_solver(ft_dict, mc_object, repeat_dict={}):
    """@brief Hybrid risk model solver. Solves first all FTs, then the MC"""
    ft_result_dict = {}
    mc_transitions = mc_object.get_transitions()
    for ft, ft_elements in ft_dict.items():
        ft_object = ft_elements[0]
        ft_graph = ft_elements[1]
        solve_ft(ft_graph, ft_object)

    for ft, ft_elements in ft_dict.items():
        ft_object = ft_elements[0]
        top_event = ft_object.get_top_event()
        top_event_prob = ft_object.get_top_event_failure_prob()
        ft_result_dict[top_event] = top_event_prob

    for from_state, elements in mc_transitions.items():
        for to_state, value in elements.items():
            if value in ft_result_dict:
                transition_prob = ft_result_dict[value]
                mc_object.add_single_transition(from_state, to_state, transition_prob)
            elif value == '1':
                continue
            else:
                for top_event, prob in ft_result_dict.items():
                    if ("1 - " + top_event) == value:
                        prob = 1 - prob
                        mc_object.add_single_transition(from_state, to_state, prob)
                        if repeat_dict:
                            for s, v in repeat_dict.items():
                                if s == to_state:
                                    prob = prob * v
                                    mc_object.add_single_transition(from_state, to_state, prob)

    b_matrix, t_matrix = solve_mc(mc_object)
    return b_matrix, t_matrix
