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


def plot_data(data, save_path=None):
    """@brief plots data"""
    plt.plot(data[:, 3], label="x")
    plt.plot(data[:, 4], label="y")
    plt.plot(data[:, 5], label="z")
    plt.plot(data[:, 19], label="Joint_2_velocity")
    plt.plot(data[:, 28] / 10, label="Joint_2_torque")
    plt.plot(data[:, 0], label="camera")
    plt.plot(data[:, 1], label="gripper")
    plt.plot(data[:, 25], label="gripper_velocity")
    # plt.plot(data[:, 17] / 10, label="label")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
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


def create_custom_mc_graph(states, absorbing_states, transitions, output_file=None):
    """
    Creates a custom Markov Chain graph including transitions between all states.

    Parameters:
    - states (list): List of all states.
    - absorbing_states (list): List of absorbing states.
    - transitions (dict): Nested dictionary where the first key is the from_state,
                          the second key is the to_state, and the value is the transition probability (float).
    - output_file (str): Optional. If provided, saves the graph as an image file with the given name.

    Returns:
    - None: Displays or saves the Markov Chain graph.
    """
    # Initialize the directed graph
    G = nx.DiGraph()

    # Add nodes
    G.add_nodes_from(states + absorbing_states)

    # Add edges with transition probabilities
    for from_state, to_transitions in transitions.items():
        for to_state, probability in to_transitions.items():
            G.add_edge(from_state, to_state, weight=probability)

    # Define manual positions for a left-right layout
    pos = {}
    y_offset = 0
    for i, state in enumerate(states):
        pos[state] = (0, y_offset)
        y_offset -= 2  # Increase vertical spacing

    y_offset = 0
    for i, state in enumerate(absorbing_states):
        pos[state] = (4, y_offset)
        y_offset -= 2  # Increase vertical spacing

    # Draw the graph
    plt.figure(figsize=(10, len(states) * 1.5))
    nx.draw(
        G, pos, with_labels=True,
        node_color=['white' if state not in absorbing_states else 'red' for state in G.nodes()],
        node_size=5000, font_size=14, font_weight='bold', edge_color='green', arrows=True, width=2, edgecolors='black'
    )

    # Draw edge labels
    edge_labels = {(from_state, to_state): f"{G[from_state][to_state]['weight']:.2f}"
                   for from_state, to_state in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=14, label_pos=0.5)

    # Remove axis
    plt.axis('off')

    # Save the plot to a file if specified
    if output_file:
        plt.savefig(output_file, format='png', bbox_inches='tight')

    # Show the plot
    plt.show()


# Example usage:
states = ["object_detection", "move", "pick", "carry", "place", "reset", "done"]
absorbing_states = ["object_detection_failure", "move_failure", "pick_failure", "carry_failure", "place_failure",
                    "reset_failure"]

# Define all transitions, including those between transient states
transitions = {
    "object_detection": {"move": 0.8, "object_detection_failure": 0.2},
    "move": {"pick": 0.7, "move_failure": 0.3},
    "pick": {"carry": 0.6, "pick_failure": 0.4},
    "carry": {"place": 0.5, "carry_failure": 0.5},
    "place": {"reset": 0.9, "place_failure": 0.1},
    "reset": {"done": 0.8, "reset_failure": 0.2},
}

create_custom_mc_graph(states, absorbing_states, transitions)

print(states)


