from dra_robotic_system import *
from dra_reliability_models import *
from dra_graph import *
from dra_json import *
from dra_solver import *
from dra_behavioral_analysis import *
import numpy as np
import matplotlib.pyplot as plt


def load_data(file_path):
    data = np.load(file_path)
    return data


def get_skill_str(skill_id):
    # Function to get the skill string
    # Define the lookup table
    skill_lookup = {
        1: 'move',
        2: 'pick',
        3: 'carry',
        4: 'place',
        5: 'object_detection',
        6: 'reset',
        7: 'pour',
        8: 'shake',
    }
    return skill_lookup.get(skill_id, 'Unknown skill')


def get_component_str(component_id):
    # Function to get the component string
    # Define the lookup table
    component_lookup = {
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
    }
    return component_lookup.get(component_id, 'Unknown_component')


def get_component_property_link(component_name):
    # Function to get the component string
    # Define the lookup table
    component_lookup = {
        'Joint_1': 0,
        'Joint_2': 1,
        'Joint_3': 2,
        'Joint_4': 3,
        'Joint_5': 4,
        'Joint_6': 5,
        'Joint_7': 6,
        'Gripper': 7
    }
    return component_lookup.get(component_name, 'Unknown_component')


def classify_property_dict(property_dict):
    """
    Classifies properties based on their type and values provided in a dictionary for each skill.
    """
    classifications = {}

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
                    elif 0.7 <= abs_value <= 1.55:
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


def create_skill_list(skill_objects):
    skill_list = []
    for skill in skill_objects:
        skill_list.append(skill.name)
    return skill_list


def update_be_probability(robotic_system_obj, skill, component):
    """@brief updates the basic event probabilities depending on the properties of the components of the robotic system"""
    properties_dict = {}
    for c in robotic_system_obj.components:
        if c.name == component:
            old_probability = c.get_failure_prob()
            properties = c.get_properties()
            properties_dict['torque'] = properties['torque'][skill]
            properties_dict['velocity'] = properties['velocity'][skill]
            updated_probability = get_prob_factor(properties_dict) * old_probability
    return updated_probability


def perform_sensitivity_analysis(self, robotic_system):
    """@brief performs sensitivity analysis on the HybridReliabilityModel"""
    for component in robotic_system.components:
        old_prob = component.get_failure_prob()
        component.set_failure_prob(old_prob * 10)
    return True


def create_fault_trees(robotic_system_obj, hybrid_model_obj):
    """@brief creates a fault tree for each skill of the robotic system containing the used components"""
    states = create_skill_list(robotic_system_obj.get_skills())
    be_dict = {}
    for state in states:
        ft_name = state + '_failure'
        ft_top_event = ft_name.lower()
        ft_skill = state
        hybrid_model_obj.add_fault_tree(FaultTree(ft_name, ft_top_event, ft_skill))

    for skill in robotic_system_obj.skills:
        be_dict[skill.name] = {}
        active_components = skill.get_components()
        for component in robotic_system_obj.components:
            if component.name in active_components:
                prob = update_be_probability(robotic_system_obj, skill.id, component.name)
                be_dict[skill.name][component.name] = prob

    for ft in hybrid_model_obj.fault_trees:
        ft.auto_create_ft(basic_events=be_dict[ft.skill])

    return hybrid_model_obj



robotic_data = read_json('franka_config.json')
component_names = robotic_data['components']
robotic_name = robotic_data['robot']
robotic_type = robotic_data['robot_type']

robotic_system = RoboticSystem(robotic_name, robot_type=robotic_type)
for component, reliability_data in component_names.items():
    failure_probability = reliability_data['failure_probability']
    redundancy = reliability_data['redundancy']
    robotic_system.add_component(Component(component, failure_prob=failure_probability, redundancy=redundancy))

print(robotic_system.get_name())


robot_data = load_data('pick_place_data.npy')
behavioral_profile = BehavioralAnalysis(name=robotic_system.get_name(), time_series=robot_data, robot_type=robotic_system.get_robot_type())
behavioral_profile.detect_skill_sequence()
skills = behavioral_profile.get_skill_sequence()
for skill in skills:
    robotic_system.add_skill(Skill(name=get_skill_str(skill), id=skill))
behavioral_profile.analyze_active_components()
active_components = behavioral_profile.get_active_components()
for skill, a_c in active_components.items():
    for s in robotic_system.skills:
        if s.id == skill:
            s.add_component('Power_Supply')
            s.add_component('Controller')
            s.add_component('Sensors')
            for c in a_c:
                s.add_component(get_component_str(c))
behavioral_profile.extract_properties()
extracted_properties = behavioral_profile.get_extracted_properties()
classified_properties = classify_property_dict(extracted_properties)
for skill, properties in classified_properties.items():
    for c in robotic_system.components:
        for property_key, property_value in properties.items():
            index = get_component_property_link(c.name)
            if index != 'Unknown_component':
                if property_key == 'torque' and index == 7:
                    c.add_property(property_name=property_key, value='low', skill=skill)
                else:
                    c.add_property(property_name=property_key, value=property_value[index], skill=skill)
            else:
                c.add_property(property_name=property_key, value='low', skill=skill)
hybrid_model = HybridReliabilityModel('Franka_hybrid')
mc = MarkovChain('Franka')
states = create_skill_list(robotic_system.get_skills())
mc.auto_create_mc(states=states, done_state=True, repeat_info=1)
hybrid_model.add_markov_chain(mc)
hybrid_model = create_fault_trees(robotic_system, hybrid_model)
""""
for state in states:
    ft_name = state + '_failure'
    ft_top_event = ft_name.lower()
    ft_skill = state
    hybrid_model.add_fault_tree(FaultTree(ft_name, ft_top_event, ft_skill))

for skill in robotic_system.skills:
    ft_dict = {}
    for ft in hybrid_model.fault_trees:
        if skill.name == ft.skill:
            be = skill.get_components()
            for x in robotic_system.components:
                for b in be:
                    if x.name == b:
                        ft_dict[b] = x.get_failure_prob()
        ft.auto_create_ft(basic_events=ft_dict)

"""
ft_dict = {}
for ft in hybrid_model.fault_trees:
    ft_graph = create_ft_graph(ft)
    ft_dict[ft.name] = [ft, ft_graph]

system_reliability, absorption_prob, absorption_time = hybrid_model.compute_system_reliability(ft_dict=ft_dict, repeat_dict={'done': 0.1, 'object_detection': 0.9})

plot_absorbing_markov_chain(mc_object=mc, save_path='absorbing_mc.png')
# Example usage:
data = {
    'Gripper': 2e-4,
    'Joint_1': 1.23e-4,
    'Joint_2': 3.16e-4,
    'Joint_3': 1.23e-4,
    'Joint_4': 2.39e-4,
    'Joint_5': 1.26e-4,
    'Joint_6': 1.71e-4,
    'Joint_7': 1.26e-4,
    'Camera': 8.74e-5,
    'Sensors': 1.14e-4,
    'Controller': 8.75e-5,
    'Power_supply': 1.23e-4,
    'None': 8.74e-5
}

create_custom_spider_chart(data, title='System failure probability control policy (A)', save_path='spider_chart.png')







