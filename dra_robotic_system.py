"""
Dynamic Reliability Assessment
robotic system class
@author Philipp Grimmeisen
@version 1.0
@date 31.07.2024
"""


class RoboticSystem:
    """
    @brief RoboticSystem class
    """
    def __init__(self, name, robot_type=''):
        """@brief constructor"""
        self.name = name
        self.type = robot_type
        self.components = []
        self.skills = []
        self.system_failure_prob = 0
        self.failure_modes = []

    def clear(self):
        """@brief clears all elements of the class RoboticSystem"""
        self.name = ""
        self.type = ""
        self.components.clear()
        self.skills.clear()
        self.system_failure_prob = 0.0
        self.failure_modes.clear()

    def add_skill(self, skill):
        """@brief adds a skill to the list of skills"""
        self.skills.append(skill)
        return True

    def add_system_failure_prob(self, prob):
        """@brief adds the system failure probability"""
        self.system_failure_prob = prob
        return True

    def add_component(self, component):
        """@brief adds a component object to the list of component objects"""
        self.components.append(component)
        return True

    def add_failure_mode(self, failure_mode):
        """@brief adds a failure mode object to the list of failure mode objects"""
        self.failure_modes.append(failure_mode)
        return True

    def get_skills(self):
        """@brief returns the list of skill objects"""
        return self.skills

    def get_system_failure_prob(self):
        """@brief returns the system failure probability"""
        return self.system_failure_prob

    def get_name(self):
        """@brief returns the name of the system"""
        return self.name

    def set_name(self, name):
        """@brief sets the name of the system"""
        self.name = name
        return True

    def get_components(self):
        """@brief returns the list of component objects"""
        return self.components

    def get_failure_modes(self):
        """@brief returns the list of failure mode objects"""
        return self.failure_modes

    def set_robot_type(self, robot_type):
        """@brief sets the robot type"""
        self.type = robot_type
        return True

    def get_robot_type(self):
        """@brief returns the robot type"""
        return self.type
# End of class RoboticSystem


class Skill:
    """
    @brief Skill class
    """
    def __init__(self, name, id=None):
        """@brief constructor"""
        self.name = name
        self.id = id
        self.components = []
        self.failure_modes = []
        self.skill_failure_prob = 0.0

    def clear(self):
        """@brief clears all elements of the class Skill"""
        self.name = ""
        self.id = None
        self.components.clear()
        self.failure_modes.clear()
        self.skill_failure_prob = 0.0

    def get_name(self):
        """@brief returns the name of the skill"""
        return self.name

    def set_name(self, name):
        """@brief sets the name of the skill"""
        self.name = name
        return True

    def get_id(self):
        """@brief returns the id of the skill"""
        return self.id

    def set_id(self, id):
        """@brief sets the id of the skill"""
        self.id = id
        return True

    def set_skill_failure_prob(self, prob):
        """@brief sets the skill failure probability"""
        self.skill_failure_prob = prob
        return True

    def get_skill_failure_prob(self):
        """@brief returns the skill failure probability"""
        return self.skill_failure_prob

    def add_component(self, component):
        """@brief adds a component to the list of components"""
        if isinstance(component, str):
            if component not in self.components:
                self.components.append(component)
        elif isinstance(component, list):
            self.components = component
        else:
            raise ValueError("Input must be either a string or a list")
        return True

    def remove_component(self, component_name):
        """@brief removes a component from the list of components"""
        if component_name not in self.components:
            return False
        self.components.remove(component_name)
        return True

    def add_failure_mode(self, failure_mode):
        """@brief adds a failure mode to the list of failure modes"""
        self.failure_modes.append(failure_mode.name)
        return True

    def remove_failure_mode(self, failure_mode):
        """@brief removes a failure mode from the list of failure modes"""
        if failure_mode.name not in self.failure_modes:
            return False
        self.failure_modes.remove(failure_mode.name)
        return True
# End of class Skill


class Component:
    """
    @brief Component class
    """
    def __init__(self, name, failure_prob=0.0, redundancy=False):
        """@brief constructor"""
        self.name = name
        self.properties = {}
        self.skills = []
        self.failure_prob = failure_prob
        self.redundancy = redundancy

    def clear(self):
        """@brief clears all elements of the class Component"""
        self.name = ""
        self.properties.clear()
        self.skills.clear()
        self.failure_prob = 0.0
        self.redundancy = False

    def get_name(self):
        """@brief returns the name of the component"""
        return self.name

    def set_name(self, name):
        """@brief sets the name of the component"""
        self.name = name
        return True

    def add_property(self, property_name, value, skill):
        """@brief adds a property to the list of properties"""
        if property_name not in self.properties:
            self.properties[property_name] = {}
        self.properties[property_name][skill] = value
        return True

    def remove_property(self, property_name):
        """@brief removes a property from the list of properties"""
        if property_name not in self.properties:
            return False
        del self.properties[property_name]
        return True

    def get_properties(self):
        """@brief returns the list of properties"""
        return self.properties

    def get_property(self, property_name):
        """@brief returns a specific property"""
        return self.properties[property_name]

    def add_skill(self, skill):
        """@brief adds a skill to the list of skills"""
        self.skills.append(skill.name)
        return True

    def remove_skill(self, skill):
        """@brief removes a skill from the list of skills"""
        if skill.name not in self.skills:
            return False
        self.skills.remove(skill.name)
        return True

    def set_failure_prob(self, prob):
        """@brief sets the failure probability"""
        self.failure_prob = prob
        return True

    def get_failure_prob(self):
        """@brief returns the failure probability"""
        return self.failure_prob

    def set_redundancy(self, value):
        """@brief sets the redundancy"""
        self.redundancy = value
        return True

    def get_redundancy(self):
        """@brief returns the redundancy"""
        return self.redundancy
# End of class Component
