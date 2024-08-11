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


def create_custom_spider_chart(data_dict, title='Spider Diagram'):
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

    # Show the plot
    plt.show()

