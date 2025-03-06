import numpy as np
import os
from typing import Dict, Union, Any, List
from robotic_system import *
from graph import *
from reliability_models import *
from behavioral_analysis import *

LOOKUP_TABLES: Dict[str, Dict[int, str]] = {
    'Franka Emika Panda': {
        1: 'object_detection',
        2: 'move',
        3: 'pick',
        4: 'carry',
        5: 'place',
        6: 'shake',
        7: 'pour',
        8: 'rotate',
        9: 'reset',
    },
    'UR5': {
        1: 'move',
        2: 'pick',
        3: 'carry',
        4: 'place',
        5: 'reset',
    },
    'Jaco 2': {
        0: 'place',
        1: 'carry',
        2: 'pick',
    },
    'OpenManipulator': {
        1: 'move',
        2: 'pick',
        3: 'carry',
        4: 'place',
        5: 'reset',
    },
    'Franka Emika Cable': {
        0: 'pick',
        1: 'route',
        2: 'perturb',
        3: 'next',
    }
}

COMPONENT_LOOKUP_TABLES: Dict[str, Dict[int, str]] = {
    'Franka Emika Panda': {
        0: 'Camera',
        1: 'Gripper',
        2: 'Gripper',
        3: 'Joint_1',
        4: 'Joint_2',
        5: 'Joint_3',
        6: 'Joint_4',
        7: 'Joint_5',
        8: 'Joint_6',
        9: 'Joint_7',
    },
    'UR5': {
        0: 'Joint_1',
        1: 'Joint_2',
        2: 'Joint_3',
        3: 'Joint_4',
        4: 'Joint_5',
        5: 'Joint_6',
        6: 'Gripper',
    },
    'Jaco 2': {
        0: 'Joint_1',
        1: 'Joint_2',
        2: 'Joint_3',
        3: 'Joint_4',
        4: 'Joint_5',
        5: 'Joint_6',
        6: 'Gripper',
    },
    'OpenManipulator': {
        0: 'Joint_1',
        1: 'Joint_2',
        2: 'Joint_3',
        3: 'Joint_4',
        4: 'Joint_5',
        5: 'Joint_6',
        6: 'Gripper',
    },
    'Franka Emika Cable': {
        0: 'Joint_1',
        1: 'Joint_2',
        2: 'Joint_3',
        3: 'Joint_4',
        4: 'Joint_5',
        5: 'Joint_6',
        6: 'Joint_7',
    }
}

COMPONENT_PROPERTY_LINKS: Dict[str, int] = {
    'Joint_1': 0,
    'Joint_2': 1,
    'Joint_3': 2,
    'Joint_4': 3,
    'Joint_5': 4,
    'Joint_6': 5,
    'Joint_7': 6,
    'Gripper': 7
}


def load_data(file_path: str) -> np.ndarray:
    """
    Load data from a .npy file.

    Args:
        file_path (str): Path to the numpy file (.npy).

    Returns:
        np.ndarray: The data loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        Exception: For any error encountered during loading.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        data = np.load(file_path, allow_pickle=True)
    except Exception as e:
        raise Exception(f"Error loading file {file_path}: {e}")

    return data


def get_skill_str(skill_id: int, robotic_system: str) -> str:
    """
    Retrieve the skill string for a given robotic system and skill ID.

    Args:
        skill_id (int): The identifier for the skill.
        robotic_system (str): The name of the robotic system.

    Returns:
        str: The corresponding skill string, or 'Unknown skill' if not found.
    """
    lookup = LOOKUP_TABLES.get(robotic_system)
    if lookup is None:
        # Handle unknown robotic systems explicitly.
        return 'Unknown skill'

    return lookup.get(skill_id, 'Unknown skill')


def get_component_str(component_id: int, robotic_system: str) -> str:
    """
    Retrieve the component string for a given robotic system and component ID.

    Args:
        component_id (int): The identifier for the component.
        robotic_system (str): The name of the robotic system.

    Returns:
        str: The corresponding component string, or 'Unknown_component' if not found.
    """
    lookup = COMPONENT_LOOKUP_TABLES.get(robotic_system)
    if lookup is None:
        return 'Unknown_component'
    return lookup.get(component_id, 'Unknown_component')


def get_component_property_link(component_name: str) -> Union[int, str]:
    """
    Retrieve the property link for a given component name.

    Args:
        component_name (str): The name of the component.

    Returns:
        int or str: The corresponding property link (as an integer) if found,
        or 'Unknown_component' if the component name is not in the lookup.
    """
    return COMPONENT_PROPERTY_LINKS.get(component_name, 'Unknown_component')


def add_skills(skills: List[int], robotic_system: Any) -> None:
    """
    Adds skills to the robotic system.

    Args:
        skills (List[int]): A list of skill IDs.
        robotic_system: The robotic system instance that has a 'name' attribute and an 'add_skill' method.
    """
    for skill_id in skills:
        skill_name = get_skill_str(skill_id, robotic_system=robotic_system.name)
        # Create a new Skill object with the resolved name and id, then add it to the system.
        robotic_system.add_skill(Skill(name=skill_name, id=skill_id))


def add_properties(classified_properties: Dict[Any, Dict[str, Any]], robotic_system: Any) -> None:
    """
    Adds properties to each component of the robotic system based on classified properties.

    Args:
        classified_properties (Dict[Any, Dict[str, Any]]): A dictionary where each key is a skill (or skill id)
            and its value is another dictionary mapping property names to property values.
            Each property value is expected to be an indexable object (e.g., list or dict).
        robotic_system: The robotic system instance with a 'components' attribute (list of components).
    """
    for skill, properties in classified_properties.items():
        for component in robotic_system.components:
            for property_key, property_value in properties.items():
                index: Union[int, str] = get_component_property_link(component.name)
                if index != 'Unknown_component':
                    # property_value is assumed to be indexable by the returned index.
                    component.add_property(property_name=property_key,
                                           value=property_value[index],
                                           skill=skill)
                else:
                    # Default value if the component is not found in the lookup.
                    component.add_property(property_name=property_key,
                                           value='low',
                                           skill=skill)


def add_components_to_skill(robotic_system: Any, active_components: Dict[Any, List[int]]) -> None:
    """
    Adds components to each skill of the robotic system based on the active components mapping.

    Args:
        robotic_system: The robotic system instance that has a 'skills' attribute (list of skills)
            and a 'name' attribute.
        active_components (Dict[Any, List[int]]): A dictionary mapping a skill (or skill id) to a list of
            component IDs that should be added to that skill.
    """
    for skill_id, component_ids in active_components.items():
        for skill in robotic_system.skills:
            if skill.id == skill_id:
                # Add default components to the skill.
                skill.add_component('Power_Supply')
                skill.add_component('Controller')
                skill.add_component('Sensors')
                # Add the active components using the lookup table.
                for component_id in component_ids:
                    component_name = get_component_str(component_id, robotic_system=robotic_system.name)
                    skill.add_component(component_name)


def classify_property_dict(property_dict: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, List[Union[str, float]]]]:
    """
    Classifies properties for each skill based on type-specific thresholds.

      - For 'velocity':
          * 'low'    if 0.0 <= abs(value) < 0.3
          * 'medium' if 0.3 <= abs(value) < 0.7
          * 'high'   if 0.7 <= abs(value) <= 2.01
          * Otherwise, the raw absolute value is appended.

      - For 'torque':
          * 'low'         if 0.0 <= abs(value) < 10.0
          * 'medium'      if 10.0 <= abs(value) < 30.0
          * 'high'        if 30.0 <= abs(value) <= 50.1
          * Otherwise, 'out of range' is appended.

      - For any other property type, 'invalid property type' is appended for each value.

    Args:
        property_dict (Dict[str, Dict[str, List[float]]]):
            A nested dictionary where each key is a skill identifier mapping to another dictionary.
            The inner dictionary's keys are property types and values are lists of float values.

    Returns:
        Dict[str, Dict[str, List[Union[str, float]]]]:
            A nested dictionary with the same structure, but each value replaced by its classification.
    """
    classifications: Dict[str, Dict[str, List[Union[str, float]]]] = {}

    if not property_dict:
        return classifications

    for skill, properties in property_dict.items():
        classifications[skill] = {}
        for property_type, values in properties.items():
            classifications[skill][property_type] = []
            for value in values:
                abs_value = abs(value)
                if property_type == 'velocity':
                    if 0.0 <= abs_value < 0.3:
                        classifications[skill][property_type].append('low')
                    elif 0.3 <= abs_value < 0.7:
                        classifications[skill][property_type].append('medium')
                    elif 0.7 <= abs_value <= 2.01:
                        classifications[skill][property_type].append('high')
                    else:
                        classifications[skill][property_type].append(abs_value)
                elif property_type == 'torque':
                    if 0.0 <= abs_value < 10.0:
                        classifications[skill][property_type].append('low')
                    elif 10.0 <= abs_value < 30.0:
                        classifications[skill][property_type].append('medium')
                    elif 30.0 <= abs_value <= 50.1:
                        classifications[skill][property_type].append('high')
                    else:
                        classifications[skill][property_type].append('out of range')
                else:
                    classifications[skill][property_type].append('invalid property type')

    return classifications


def create_skill_list(skill_objects: List[Any]) -> List[str]:
    """
    Extracts and returns a list of skill names from a list of skill objects.

    Args:
        skill_objects (List[Any]): A list of objects that have a 'name' attribute.

    Returns:
        List[str]: A list containing the 'name' attribute of each skill.
    """
    return [skill.name for skill in skill_objects]


def create_ft_dict(hybrid_model: Any) -> Dict[str, List[Any]]:
    """
    Creates a dictionary of fault trees from the hybrid model.

    Args:
        hybrid_model (Any): An object that contains a 'fault_trees' attribute (iterable of fault tree objects).
                            Each fault tree is expected to have a 'name' attribute.

    Returns:
        Dict[str, List[Any]]: A dictionary mapping fault tree names to a list [fault_tree, fault_tree_graph].
    """
    ft_dict: Dict[str, List[Any]] = {}
    for ft in hybrid_model.fault_trees:
        # Assumes create_ft_graph is defined elsewhere and returns the fault tree graph for ft.
        ft_graph = create_ft_graph(ft)
        ft_dict[ft.name] = [ft, ft_graph]
    return ft_dict


def update_be_probability(robotic_system_obj: Any, skill: int, component: str) -> float:
    """
    Update the basic event probability for a given component in the robotic system based on
    the properties corresponding to a given skill (used as an index).

    Args:
        robotic_system_obj (RobotSystem): An object that has an attribute `components` (a list
            of component objects). Each component must have a `name` attribute, a method
            `get_failure_prob() -> float`, and a method `get_properties()` that returns a dict
            with keys 'torque' and 'velocity' whose values are indexable.
        skill (int): The skill identifier used as an index to access property values.
        component (str): The name of the component for which to update the probability.

    Returns:
        float: The updated basic event probability for the component.

    Raises:
        ValueError: If the specified component is not found in the robotic system.
    """
    # Find the target component(s)
    target_components = [c for c in robotic_system_obj.components if c.name == component]
    if not target_components:
        raise ValueError(f"Component '{component}' not found in the robotic system.")
    # Assume the first matching component is used
    comp = target_components[0]
    old_probability = comp.get_failure_prob()
    properties = comp.get_properties()
    if properties:
        # Expect properties['torque'] and properties['velocity'] to be indexable by `skill`
        properties_dict = {
            'torque': properties['torque'][skill],
            'velocity': properties['velocity'][skill]
        }
        updated_probability = get_prob_factor(properties_dict) * old_probability
    else:
        updated_probability = old_probability

    return updated_probability


def create_fault_trees(robotic_system_obj: Any, hybrid_model_obj: Any) -> Any:
    """
    Creates a fault tree for each skill of the robotic system containing the used components.

    Args:
        robotic_system_obj (RobotSystem): An object representing the robotic system that contains
            attributes/methods such as `get_skills()`, `skills` (a list of Skill objects), and
            `components` (a list of Component objects).
        hybrid_model_obj (HybridModel): An object that manages fault trees. It must support
            `add_fault_tree()` and have a `fault_trees` attribute.

    Returns:
        HybridModel: The updated hybrid model with fault trees added and processed.
    """
    # Create a list of skill names from the robotic system.
    states = create_skill_list(robotic_system_obj.get_skills())
    be_dict: Dict[str, Dict[str, float]] = {}

    # Create fault trees for each skill.
    for state in states:
        ft_name = f"{state}_failure"
        ft_top_event = ft_name.lower()
        ft_skill = state
        # Assumes FaultTree is defined and imported elsewhere.
        hybrid_model_obj.add_fault_tree(FaultTree(ft_name, ft_top_event, ft_skill))

    # Compute basic event probabilities for each skill.
    for skill in robotic_system_obj.skills:
        be_dict[skill.name] = {}
        active_components = skill.get_components()
        for component in robotic_system_obj.components:
            if component.name in active_components:
                prob = update_be_probability(robotic_system_obj, skill.id, component.name)
                be_dict[skill.name][component.name] = prob

    # Auto-create fault trees using the computed probabilities.
    for ft in hybrid_model_obj.fault_trees:
        # Assumes ft.skill is a string matching a key in be_dict.
        ft.auto_create_ft(basic_events=be_dict[ft.skill])

    return hybrid_model_obj


def add_skill_failure_prob(hybrid_model: Any, robotic_system: Any) -> bool:
    """
    Sets the skill failure probabilities for the Markov Chain by matching each skill
    with its corresponding fault tree in the hybrid model.

    Args:
        hybrid_model (HybridModel): An object that contains fault trees with a `fault_trees` attribute.
        robotic_system (RobotSystem): An object representing the robotic system with a `skills` attribute.

    Returns:
        bool: True once the operation is complete.
    """
    for skill in robotic_system.skills:
        for ft in hybrid_model.fault_trees:
            if skill.name == ft.skill:
                skill.set_skill_failure_prob(ft.get_top_event_failure_prob())
    return True


def perform_sensitivity_analysis(hybrid_model: Any,
                                 robotic_system: Any,
                                 sensitivity_analysis_data: Dict[str, float]
                                 ) -> Dict[str, float]:
    """
    Performs sensitivity analysis on the hybrid reliability model for each component in the robotic system.

    Args:
        hybrid_model (Any): The initial hybrid reliability model.
        robotic_system (Any): The robotic system containing components and skills.
        sensitivity_analysis_data (Dict[str, float]): A dictionary to store computed system reliability
            for each component, keyed by component name.

    Returns:
        Dict[str, float]: The updated sensitivity analysis data mapping each component's name to its
                          computed system reliability.
    """
    for component in robotic_system.components:
        # Create new states based on the current skills of the robotic system.
        new_states = create_skill_list(robotic_system.get_skills())

        # Create a new hybrid model and associated Markov chain for the component.
        hybrid_model_new = HybridReliabilityModel(component.name)
        mc_new = MarkovChain(component.name)
        mc_new.auto_create_mc(states=new_states, done_state=True, repeat_info=1)
        hybrid_model_new.add_markov_chain(mc_new)

        # Create fault trees based on the new hybrid model.
        hybrid_model_new = create_fault_trees(robotic_system, hybrid_model_new)

        # Update the basic event probability for the component in each fault tree.
        for ft in hybrid_model_new.fault_trees:
            if component.name in ft.basic_events:
                old_prob = ft.basic_events[component.name]
                new_prob = old_prob * 10.0
                ft.basic_events[component.name] = new_prob

        # Create a fault tree dictionary and compute system reliability.
        new_fts = create_ft_dict(hybrid_model_new)
        new_system_reliability, new_absorption_prob, new_absorption_time = hybrid_model_new.compute_system_reliability(
            ft_dict=new_fts,
            repeat_dict={'done': 0.1, 'object_detection': 0.9}
        )

        # Store the computed reliability in the sensitivity analysis data.
        sensitivity_analysis_data[component.name] = new_system_reliability

    return sensitivity_analysis_data
