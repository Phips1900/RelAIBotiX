"""
Dynamic Reliability Assessment graph generator
@author Philipp Grimmeisen
@version 1.0
@date 01.08.2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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


def create_custom_spider_chart(data_dict, title='Spider Diagram', save_path=None):
    """
    Creates a custom spider (radar) diagram based on a dictionary of components and their associated probabilities.

    Parameters:
    - data_dict (dict): A dictionary where keys are components (str) and values are probabilities (float).
    - title (str): The title of the spider diagram.

    Returns:
    - None: Displays the spider diagram.
    """
    # Extract components and their corresponding probabilities
    labels = list(data_dict.keys())
    values = list(data_dict.values())

    # Number of variables we're plotting.
    num_vars = len(labels)

    # Split the circle into even parts and save the angles.
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # The plot is a circle, so we need to "complete the loop"
    values += values[:1]
    angles += angles[:1]
    labels += labels[:1]

    # Create the figure
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Draw one axe per variable and add labels
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw the outline of the data
    ax.plot(angles, values, linewidth=2, linestyle='-', color='red', label='Components')

    # Fill the area under the curve
    ax.fill(angles, values, color='red', alpha=0.25)

    # Add labels to the chart
    ax.set_xticks(angles[:-1])

    # Combine labels and values for display
    labels_with_values = [f"{label}\n{value:.2E}" for label, value in zip(labels[:-1], values[:-1])]
    ax.set_xticklabels(labels_with_values, color='grey', size=12, fontweight='bold')

    # Customize the grid
    ax.yaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelcolor='blue')
    ax.grid(color='blue', linestyle='-', linewidth=1)

    # Add labels to the radial axes in scientific notation
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2E}'))

    # Add a title with a custom style
    plt.title(title, size=16, color='black', y=1.1, fontweight='bold')

    # Set the background color to white
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')

    # Show the plot
    plt.show()


def plot_absorbing_markov_chain(mc_object, save_path=None):
    """
    Creates and displays a graph of an absorbing Markov chain.

    Parameters:
    - mc_object: An object representing the Markov chain. Must have methods:
        - get_states(): Returns a list of all states.
        - get_absorbing_states(): Returns a list of absorbing states.
        - get_transitions(): Returns a nested dictionary where the first key is the from_state,
                             the second key is the to_state, and the value is the transition weight (float).
        - get_edges(): Returns a dictionary where keys are from_states and values are either a single to_state or a list of to_states.

    Returns:
    - None: Displays the Markov chain graph.
    """
    # Initialize the directed graph
    G = nx.DiGraph()

    # Add nodes (states)
    states = mc_object.get_states()
    absorbing_states = mc_object.get_absorbing_states()
    G.add_nodes_from(states)

    # Add edges (transitions)
    edges = mc_object.get_edges()
    transitions = mc_object.get_transitions()

    for from_state, to_states in edges.items():
        # Ensure to_states is a list even if it's a single state
        if isinstance(to_states, str):
            to_states = [to_states]

        for to_state in to_states:
            if from_state in transitions and to_state in transitions[from_state]:
                weight = transitions[from_state][to_state]
                try:
                    weight = float(weight)
                except ValueError:
                    print(f"Warning: Transition weight for {from_state} -> {to_state} could not be converted to float.")
                    continue
                # Add the edge with the float weight
                G.add_edge(from_state, to_state, weight=weight)
            else:
                print(f"Warning: No transition probability for {from_state} -> {to_state}")

    # Ensure that color_map has the same length as the number of nodes
    color_map = ['red' if state in absorbing_states else 'blue' for state in G.nodes]

    # Set edge labels to be the transition probabilities
    edge_labels = {(from_state, to_state): f"{G[from_state][to_state]['weight']:.2f}"
                   for (from_state, to_state) in G.edges()}

    pos = {
        'object_detection': (0, 0),
        'move': (0, -20),
        'pick': (0, -40),
        'carry': (0, -60),
        'place': (0, -80),
        'reset': (0, -100),
        'done': (0, -120),
        'object_detection_failure': (6, 0),
        'move_failure': (6, -20),
        'pick_failure': (6, -40),
        'carry_failure': (6, -60),
        'place_failure': (6, -80),
        'reset_failure': (6, -100),
    }

    # Draw the graph with the manually defined positions
    nx.draw(G, pos, with_labels=True, node_color=color_map, node_size=5000, font_size=12, font_weight='bold',
            edge_color='gray', arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='green', font_size=10)

    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')

    # Display the graph
    plt.title('Absorbing Markov Chain')
    plt.show()


