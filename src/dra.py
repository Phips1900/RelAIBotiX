from robotic_system import *
from reliability_models import *
from graph import *
from json_handler import *

component_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5', 'Joint_6', 'Joint_7', 'Controller', 'Power_Supply', 'Gripper', 'Camera', 'Sensors']
components = {}
skill_names = ['Move', 'Pick', 'Carry', 'Place', 'Reset']
skills = {}

for component_name in component_names:
    component = Component(component_name)
    components[component_name] = component

for skill_name in skill_names:
    skill = Skill(skill_name)
    skills[skill_name] = skill

franka = RoboticSystem('Franka')
for component in components.values():
    franka.add_component(component)
for skill in skills.values():
    franka.add_skill(skill)

mc = MarkovChain('test')
skill_list = []
s = franka.get_skills()
for s_ in s:
    skill_list.append(s_.name)
mc.auto_create_mc(skill_list, repeat_info=1)

redundant_component = {'1': ['joint_1', 'joint_2'], '2': ['cpu', 'gpu']}

basic_events = {'joint_1': 0.00000003577, 'joint_2': 0.00000003577, 'joint_3': 0.00000003577, 'joint_4': 0.00000003577,
                'joint_5': 0.00000003577, 'joint_6': 0.00000003577, 'joint_7': 0.00000003577,
                'power_supply': 0.00000006674, 'gripper': 0.00000007344, 'cpu': 0.00000002167, 'gpu': 0.00000002167}

ft = FaultTree('test')
ft.auto_create_ft(top_event='test', skill='test', basic_events=basic_events, redundancy=True,
                  redundant_components=redundant_component)

ft_graph = create_ft_graph(ft)
draw_graph(ft_graph)

data = read_json('franka_config.json')
print(data)

print(components)
