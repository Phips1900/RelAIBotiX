from dra_robotic_system import *

component_names = ['Joint_1', 'Joint_2', 'Joint_3', 'Joint_4', 'Joint_5', 'Joint_6', 'Joint_7', 'Controller', 'Power_Supply', 'Gripper', 'Camera', 'Sensors']
components = {}

for component_name in component_names:
    component = Component(component_name)
    components[component_name] = component

print(components)