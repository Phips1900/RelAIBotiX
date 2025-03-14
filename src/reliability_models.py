"""This module provides the classes HybridReliabilityModel, MarkovChain, FaultTree, and FailureMode."""
from typing import Optional, List, Tuple, Dict
from solver import *


class HybridReliabilityModel:
    """
    A model that combines Markov chain and fault tree methods to evaluate system reliability.

    This class provides methods to add, remove, and access the individual components (Markov chain
    and fault trees) and to compute the overall system reliability.
    """
    def __init__(self, name):
        """
        Initialize the HybridReliabilityModel with a name.

        Parameters:
            name (str): The identifier for this reliability model.
        """
        self.name = name
        self.markov_chain = None    # Instance of a MarkovChain
        self.fault_trees: list[FaultTree] = []      # List to hold FaultTree instances

    def clear(self) -> None:
        """
          Reset the model by clearing its name, Markov chain, and fault trees.
        """
        self.name = ""
        self.markov_chain = None
        self.fault_trees.clear()

    def add_markov_chain(self, markov_chain) -> bool:
        """
        Set the MarkovChain for the model.

        Parameters:
            markov_chain (MarkovChain): The MarkovChain instance to assign.

        Returns:
            bool: True upon successful assignment.
        """
        self.markov_chain = markov_chain
        return True

    def add_fault_tree(self, fault_tree) -> bool:
        """
        Append a FaultTree to the model.

        Parameters:
            fault_tree: The FaultTree instance to add.

        Returns:
            bool: True upon successful addition.
        """
        self.fault_trees.append(fault_tree)
        return True

    def get_markov_chain(self):
        """
        Retrieve the currently set MarkovChain.

        Returns:
            MarkovChain or None: The MarkovChain instance, or None if not set.
        """
        return self.markov_chain

    def get_fault_trees(self):
        """
        Retrieve the list of FaultTree instances.

        Returns:
            list: A list containing all FaultTree instances added to the model.
        """
        return self.fault_trees

    def remove_fault_tree(self, fault_tree) -> bool:
        """
        Remove a FaultTree instance from the model.

        Parameters:
            fault_tree (FaultTree): The FaultTree instance to remove.

        Returns:
            bool: True if removal was successful, False if the FaultTree was not found.
        """
        if fault_tree not in self.fault_trees:
            return False
        self.fault_trees.remove(fault_tree)
        return True

    def remove_markov_chain(self) -> bool:
        """
        Remove the MarkovChain from the model.

        Returns:
            bool: True upon successful removal.
        """
        self.markov_chain = None
        return True

    def get_name(self) -> str:
        """
        Get the name of the reliability model.

        Returns:
            str: The name of the model.
        """
        return self.name

    def set_name(self, name: str) -> bool:
        """
        Set or update the model's name.

        Parameters:
            name (str): The new name for the model.

        Returns:
            bool: True upon successful update.
        """
        self.name = name
        return True

    def compute_system_reliability(self, ft_dict, repeat_dict: Optional[Dict] = None) -> Tuple[float, List[float], List[float]]:
        """Function to compute the system reliability using the hybrid solver."""
        if self.markov_chain is None:
            raise ValueError("Cannot compute reliability: Markov chain is not set.")

        # Call the hybrid_solver function to compute the probability and time to absorption
        absorption_prob, absorption_time = hybrid_solver(ft_dict=ft_dict, mc_object=self.markov_chain, repeat_dict=repeat_dict)
        num_cols = absorption_prob.shape[1]
        num_cols = num_cols - 1

        absorption_prob = absorption_prob[0, 0:num_cols]
        system_reliability = absorption_prob.sum()
        return system_reliability, absorption_prob, absorption_time


class MarkovChain:
    """
    MarkovChain class
    """
    def __init__(self, name):
        """constructor"""
        self.name = name
        self.states = []
        self.absorbing_states = []
        self.edges = {}
        self.transitions = {}
        self.transition_matrix = []

    def clear_mc(self):
        """Clears all elements of the class MarkovChain"""
        self.name = ""
        self.states.clear()
        self.absorbing_states.clear()
        self.edges.clear()
        self.transitions.clear()
        self.transition_matrix.clear()

    def auto_create_mc(self, states, done_state=False, repeat_info=0):
        """Automatically creates a Markov Chain"""
        self.add_states(states)
        self.add_absorbing_states(done_state=done_state)
        self.add_edges(repeat_info=repeat_info)
        self.add_transitions()
        return True

    def add_states(self, states):
        """Adds a list of states to the Markov Chain"""
        self.states = states
        return True

    def add_single_state(self, state):
        """Adds a single state to the Markov Chain"""
        self.states.append(state)
        return True

    def add_absorbing_states(self, done_state=False):
        """Adds an absorbing state for each state to the Markov Chain. Optional done state can be added"""
        for state in self.states:
            self.absorbing_states.append(state + '_failure')
        if done_state:
            self.absorbing_states.append('done')
        return True

    def add_single_absorbing_state(self, state):
        """Adds a single absorbing state to the Markov Chain"""
        self.absorbing_states.append(state)
        return True

    def get_states(self):
        """Returns the states of the Markov Chain"""
        return self.states

    def get_absorbing_states(self):
        """Returns the absorbing states of the Markov Chain"""
        return self.absorbing_states

    def remove_state(self, state):
        """Removes a state from the Markov Chain"""
        if state not in self.states:
            return False
        self.states.remove(state)
        return True

    def remove_absorbing_state(self, state):
        """Removes an absorbing state from the Markov Chain"""
        if state not in self.absorbing_states:
            return False
        self.absorbing_states.remove(state)
        return True

    def add_edges(self, repeat_info=0):
        """Adds edges automatically to the Markov Chain"""
        num_states = len(self.states)
        for i in range(num_states):
            try:
                next_state = self.states[i + 1]
                self.edges[self.states[i]] = []
                self.edges[self.states[i]].append(next_state)
                for a_state in self.absorbing_states:
                    if self.states[i] in a_state:
                        self.edges[self.states[i]].append(a_state)
            except (ValueError, IndexError):
                if i == num_states - 1:
                    self.edges[self.states[i]] = []
                    for a_state in self.absorbing_states:
                        if self.states[i] in a_state:
                            self.edges[self.states[i]].append(a_state)
                    if 'done' in self.absorbing_states:
                        self.edges[self.states[i]].append('done')
                    if repeat_info == 1:
                        self.edges[self.states[i]].append(self.states[0])
                else:
                    return False
        for a_state in self.absorbing_states:
            self.edges[a_state] = a_state
        return True

    def add_single_edge(self, from_state, to_state):
        """Aadds a single edge to the Markov Chain"""
        self.edges[from_state].append(to_state)
        return True

    def get_edges(self):
        """Returns the edges of the Markov Chain as dictionary"""
        return self.edges

    def remove_edge(self, from_state, to_state):
        """Removes a single edge from the Markov Chain"""
        if from_state not in self.edges:
            return False
        if to_state not in self.edges[from_state]:
            return False
        self.edges[from_state].remove(to_state)
        return True

    def add_single_transition(self, from_state, to_state, transition_value):
        """Adds a single transition to the Markov Chain"""
        if not self.transitions:
            self.transitions[from_state] = {}
        self.transitions[from_state][to_state] = transition_value
        return True

    def add_transitions(self):
        """Adds transitions automatically to the Markov Chain"""
        for from_state, to_states in self.edges.items():
            if from_state not in self.transitions:
                self.transitions[from_state] = {}
            if from_state == to_states:
                self.transitions[from_state][to_states] = '1'
            else:
                for s in to_states:
                    index = s.find('failure')
                    if index != -1:
                        self.transitions[from_state][s] = s.lower()
                    else:
                        self.transitions[from_state][s] = '1 - ' + from_state.lower() + '_failure'
        return True

    def get_transitions(self):
        """Returns the transitions of the Markov Chain as dictionary"""
        return self.transitions

    def remove_transition(self, from_state, to_state):
        """Removes a single transition from the Markov Chain"""
        if from_state not in self.transitions:
            return False
        if to_state not in self.transitions[from_state]:
            return False
        del self.transitions[from_state][to_state]
        return True


class FaultTree:
    """
    FaultTree class
    """
    def __init__(self, name, top_event='', skill=''):
        """Constructor"""
        self.name = name
        self.top_event = top_event
        self.skill = skill
        self.basic_events = {}
        self.gates = {}
        self.top_event_failure_prob = 0.0

    def clear_ft(self):
        """Clears all elements of the class FaultTree"""
        self.name = ""
        self.top_event = ""
        self.skill = ""
        self.basic_events.clear()
        self.gates.clear()
        self.top_event_failure_prob = 0.0

    def auto_create_ft(self, basic_events, top_event='', skill='', redundancy=False, redundant_components={}):
        """Automatically creates a Fault Tree"""
        if not self.top_event and top_event:
            self.set_top_event(top_event)
        if not self.skill and skill:
            self.set_skill(skill)
        if not self.top_event and not top_event:
            print("Error! No top-event")
            return False
        if not self.skill and not skill:
            print("Error! No skill link")
            return False
        self.add_basic_events(basic_events)
        self.add_gates(redundancy=redundancy, redundant_components=redundant_components)
        return True

    def set_top_event(self, top_event):
        """Sets the top event of the Fault Tree"""
        self.top_event = top_event
        return True

    def get_top_event(self):
        """Returns the top event of the Fault Tree"""
        return self.top_event

    def set_skill(self, skill):
        """Sets the linked skill of the Fault Tree"""
        self.skill = skill
        return True

    def get_skill(self):
        """Returns the linked skill of the Fault Tree"""
        return self.skill

    def set_top_event_failure_prob(self, top_event_failure_prob):
        """Sets the top event failure probability of the Fault Tree"""
        self.top_event_failure_prob = top_event_failure_prob
        return True

    def get_top_event_failure_prob(self):
        """Returns the top event failure probability of the Fault Tree"""
        return self.top_event_failure_prob

    def add_single_basic_event(self, basic_event, prob):
        """Adds a single basic event to the Fault Tree"""
        self.basic_events[basic_event] = prob
        return True

    def add_basic_events(self, basic_events):
        """Adds automatically a dictionary of basic events with failure probs to the Fault Tree"""
        self.basic_events = basic_events
        return True

    def get_basic_events(self):
        """Returns the basic events of the Fault Tree as dictionary"""
        return self.basic_events

    def remove_basic_event(self, basic_event):
        """Removes a single basic event from the Fault Tree"""
        if basic_event not in self.basic_events:
            return False
        del self.basic_events[basic_event]
        return True

    def add_single_gate(self, gate_name, gate_type, basic_events):
        """Adds a single gate to the Fault Tree"""
        self.gates[self.top_event]['OR'].append(gate_name)
        self.gates[gate_name] = {}
        for be in basic_events:
            self.gates[gate_name][gate_type].append(be)
        return True

    def add_gates(self, redundancy=False, redundant_components=None):
        """Adds automatically gates to the Fault Tree"""
        if redundant_components is None:
            redundant_components = {}
        self.gates[self.top_event] = {}
        if not redundancy:
            self.gates[self.top_event]['OR'] = []
            for be in self.basic_events:
                self.gates[self.top_event]['OR'].append(be)
        else:
            basic_events = self.basic_events
            self.gates[self.top_event]['OR'] = []
            for key, value in redundant_components.items():
                self.gates[self.top_event]['OR'].append('loss_of_' + key)
                self.gates['loss_of_' + key] = {}
                self.gates['loss_of_' + key]['AND'] = []
                for component_name in value:
                    if component_name in basic_events:
                        self.gates['loss_of_' + key]['AND'].append(component_name)
                        del basic_events[component_name]
            for be in basic_events:
                self.gates[self.top_event]['OR'].append(be)
        return True

    def get_gates(self):
        """Returns the gates of the Fault Tree as dictionary"""
        return self.gates

    def remove_gate(self, gate_name):
        """Removes a single gate from the Fault Tree"""
        if gate_name not in self.gates:
            return False
        del self.gates[gate_name]
        return True


class FailureMode:
    """
    FailureMode class
    """
    def __init__(self, name):
        """Constructor"""
        self.name = name
        self.severity = ''
        self.likelihood = ''

    def clear_fm(self):
        """Clears all elements of the class FailureMode"""
        self.name = ""
        self.severity = ""
        self.likelihood = ""

    def set_name(self, name):
        """Sets the name of the FailureMode"""
        self.name = name
        return True

    def get_name(self):
        """Returns the name of the FailureMode"""
        return self.name

    def set_severity(self, severity):
        """Sets the severity of the FailureMode"""
        self.severity = severity
        return True

    def get_severity(self):
        """Returns the severity of the FailureMode"""
        return self.severity

    def set_likelihood(self, likelihood):
        """Sets the likelihood of the FailureMode"""
        self.likelihood = likelihood
        return True

    def get_likelihood(self):
        """Returns the likelihood of the FailureMode"""
        return self.likelihood

    def remove_severity(self):
        """Removes the severity of the FailureMode"""
        self.severity = ''
        return True

    def remove_likelihood(self):
        """Removes the likelihood of the FailureMode"""
        self.likelihood = ''
        return True
