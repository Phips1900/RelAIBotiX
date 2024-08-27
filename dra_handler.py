from dra_robotic_system import *
from dra_reliability_models import *
from dra_graph import *
from dra_json import *
from dra_solver import *
from dra_behavioral_analysis import *
import numpy as np
import matplotlib.pyplot as plt
from dra_pdf import *


def load_data(file_path):
    data = np.load(file_path)
    shaped_date = data[0, 2:2980, :]
    return shaped_date


def get_skill_str(skill_id):
    # Function to get the skill string
    # Define the lookup table
    skill_lookup = {
        1: 'object_detection',
        2: 'move',
        3: 'pick',
        4: 'carry',
        5: 'place',
        6: 'shake',
        7: 'pour',
        8: 'rotate',
        9: 'reset',
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


def add_skills(skills, robotic_system):
    for skill in skills:
        robotic_system.add_skill(Skill(name=get_skill_str(skill), id=skill))


def add_properties(classified_properties, robotic_system):
    for skill, properties in classified_properties.items():
        for c in robotic_system.components:
            for property_key, property_value in properties.items():
                index = get_component_property_link(c.name)
                if index != 'Unknown_component':
                    c.add_property(property_name=property_key, value=property_value[index], skill=skill)
                else:
                    c.add_property(property_name=property_key, value='low', skill=skill)


def add_components_to_skill(robotic_system, active_components):
    for skill, a_c in active_components.items():
        for s in robotic_system.skills:
            if s.id == skill:
                s.add_component('Power_Supply')
                s.add_component('Controller')
                s.add_component('Sensors')
                for c in a_c:
                    s.add_component(get_component_str(c))


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


def create_skill_list(skill_objects):
    skill_list = []
    for skill in skill_objects:
        skill_list.append(skill.name)
    return skill_list


def create_ft_dict(hybrid_model):
    ft_dict = {}
    for ft in hybrid_model.fault_trees:
        ft_graph = create_ft_graph(ft)
        ft_dict[ft.name] = [ft, ft_graph]
    return ft_dict


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


def add_skill_failure_prob(hybrid_model, robotic_system):
    """@brief sets the skill failure probabilities for the Markov Chain"""
    for skill in robotic_system.skills:
        for ft in hybrid_model.fault_trees:
            if skill.name == ft.skill:
                skill.set_skill_failure_prob(ft.get_top_event_failure_prob())
    return True


def perform_sensitivity_analysis(hybrid_model, robotic_system, sensitivity_analysis_data):
    for component in robotic_system.components:
        new_states = create_skill_list(robotic_system.get_skills())
        hybrid_model_new = HybridReliabilityModel(component.name)
        mc_new = MarkovChain(component.name)
        mc_new.auto_create_mc(states=new_states, done_state=True, repeat_info=1)
        hybrid_model_new.add_markov_chain(mc_new)
        hybrid_model_new = create_fault_trees(robotic_system, hybrid_model_new)
        for ft in hybrid_model_new.fault_trees:
            if component.name in ft.basic_events:
                old_prob = ft.basic_events[component.name]
                new_prob = old_prob * 10.0
                ft.basic_events[component.name] = new_prob
        new_fts = create_ft_dict(hybrid_model_new)
        new_system_reliability, new_absorption_prob, new_absorption_time = hybrid_model_new.compute_system_reliability(ft_dict=new_fts, repeat_dict={'done': 0.1, 'object_detection': 0.9})
        sensitivity_analysis_data[component.name] = new_system_reliability
    return sensitivity_analysis_data


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


robot_data = load_data('pick_place_philipp.npy')
plot_data(robot_data, save_path='robot_data.png')
behavioral_profile = BehavioralAnalysis(name=robotic_system.get_name(), time_series=robot_data, robot_type=robotic_system.get_robot_type())
behavioral_profile.detect_skill_sequence()
skills = behavioral_profile.get_skill_sequence()
add_skills(skills, robotic_system)
behavioral_profile.analyze_active_components()
active_components = behavioral_profile.get_active_components()
add_components_to_skill(robotic_system, active_components)
behavioral_profile.extract_properties()
extracted_properties = behavioral_profile.get_extracted_properties()
classified_properties = classify_property_dict(extracted_properties)
add_properties(classified_properties, robotic_system)
hybrid_model = HybridReliabilityModel('Franka_hybrid')
mc = MarkovChain('Franka')
states = create_skill_list(robotic_system.get_skills())
mc.auto_create_mc(states=states, done_state=True, repeat_info=1)
hybrid_model.add_markov_chain(mc)
hybrid_model = create_fault_trees(robotic_system, hybrid_model)

ft_dict = create_ft_dict(hybrid_model)

system_reliability, absorption_prob, absorption_time = hybrid_model.compute_system_reliability(ft_dict=ft_dict, repeat_dict={'done': 0.1, 'object_detection': 0.9})
robotic_system.set_system_failure_prob(float(system_reliability))
add_skill_failure_prob(hybrid_model, robotic_system)
write_json(robotic_system, 'robotic_system.json')

sensitivity_analysis_data = {'None': system_reliability}
sensitivity_analysis_data = perform_sensitivity_analysis(hybrid_model, robotic_system, sensitivity_analysis_data)
create_custom_spider_chart(sensitivity_analysis_data, title='System failure probability', save_path='spider_chart.png')
json_file = 'robotic_system.json'
plot_files = ['spider_chart.png']

create_pdf_from_json_and_plots(json_file, plot_files, filename='robot_report.pdf')









