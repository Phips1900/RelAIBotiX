import numpy as np


class BehavioralAnalysis:
    def __init__(self, name, time_series, robot_type):
        self.name = name
        self.time_series = time_series
        self.robot_type = robot_type
        self.skill_sequence = []
        self.skill_data_points = {}
        self.active_components = {}
        self.extracted_properties = {}

    def clear(self):
        """@brief clears all elements of the class BehavioralAnalysis"""
        self.name = ""
        self.time_series = ""
        self.robot_type = ""
        self.skill_sequence.clear()
        self.skill_data_points.clear()
        self.active_components.clear()
        self.extracted_properties.clear()

    def get_all(self):
        return self.skill_sequence, self.skill_data_points, self.active_components, self.extracted_properties

    def set_time_series_data(self, time_series):
        self.time_series = time_series
        return True

    def get_time_series_data(self):
        return self.time_series

    def get_name(self):
        return self.name

    def set_name(self, name):
        self.name = name
        return True

    def get_robot_type(self):
        return self.robot_type

    def set_robot_type(self, robot_type):
        self.robot_type = robot_type
        return True

    def detect_skill_sequence(self):
        skill_column = get_columns(self.name)['skills']
        time_series = self.time_series[:, skill_column]
        iterator = 0

        if len(time_series) > 0:
            prev_value = time_series[0]
            self.skill_sequence.append(int(prev_value))
            self.skill_data_points[int(prev_value)] = {}
            self.skill_data_points[int(prev_value)]['start'] = iterator

            for value in time_series[1:]:
                iterator += 1
                if value != prev_value:
                    self.skill_sequence.append(int(value))
                    self.skill_data_points[int(prev_value)]['end'] = iterator - 1
                    self.skill_data_points[int(value)] = {}
                    self.skill_data_points[int(value)]['start'] = iterator
                    prev_value = value
                if iterator == (len(time_series) - 1):
                    self.skill_data_points[int(prev_value)]['end'] = iterator + 1
        return True

    def get_skill_sequence(self):
        return self.skill_sequence

    def get_skill_data_points(self):
        return self.skill_data_points

    def analyze_active_components(self):
        component_columns = get_columns(self.name)['components']
        time_series = self.time_series[:, component_columns]
        threshold = 0.0001
        for skill, time_points in self.skill_data_points.items():
            self.active_components[skill] = []
            start_value = time_points['start']
            end_value = time_points['end']
            data_new = time_series[start_value:end_value, 0:len(time_series[0])]
            for i in range(0, len(time_series[0])):
                previous_value = data_new[0, i]
                for value in data_new[1:, i]:
                    difference = abs(value - previous_value)
                    if difference > threshold:
                        if i not in self.active_components[skill]:
                            self.active_components[skill].append(i)
                            break
                    if np.all(data_new[:, i] == 1.0):
                        if i not in self.active_components[skill]:
                            self.active_components[skill].append(i)
                            break
        return True

    def get_active_components(self):
        return self.active_components

    def extract_properties(self):
        property_columns_vel = get_columns(self.name)['properties_velocity']
        property_columns_torque = get_columns(self.name)['properties_torque']
        property_columns = property_columns_vel + property_columns_torque
        if not property_columns:
            print('No properties found for the robot type')
            return True
        time_series = self.time_series[:, property_columns]
        for skill, time_points in self.skill_data_points.items():
            self.extracted_properties[skill] = {}
            start_value = time_points['start']
            end_value = time_points['end']
            data_new = time_series[start_value:end_value, 0:len(time_series[0])]
            max_value_column = np.max(np.abs(data_new), axis=0)
            max_value_list = max_value_column.tolist()
            if property_columns_vel:
                len_vel = len(property_columns_vel)
                self.extracted_properties[skill]['velocity'] = max_value_list[:len_vel]
            if property_columns_torque:
                len_torque = len(property_columns_torque)
                self.extracted_properties[skill]['torque'] = max_value_list[len_torque:]
        return True

    def get_extracted_properties(self):
        return self.extracted_properties


# Define the lookup table
lookup_table = {
    ('low', 'low'): 1,
    ('medium', 'low'): 4,
    ('high', 'low'): 8,
    ('low', 'medium'): 5,
    ('low', 'high'): 10,
    ('medium', 'medium'): 9,
    ('high', 'medium'): 13,
    ('medium', 'high'): 14,
    ('high', 'high'): 18
}


# Function to get the probability factor
def get_prob_factor(properties):
    key = (properties.get('velocity'), properties.get('torque'))
    return lookup_table.get(key, "Invalid properties")


def get_columns(robot_type):
    """
    Returns a dictionary of lists containing column indices that are interesting for components, properties, and skills,
    depending on the robot type.

    Parameters:
    - robot_type (str): The type of the robot ('Manipulator', 'Mobile Robot', 'Undefined').

    Returns:
    - dict: A dictionary with keys 'components', 'properties', 'skills' and values as lists of column indices.
    """
    robot_column_mapping = {
        'Franka Emika Panda': {
            'components': [0, 1, 2, 10, 11, 12, 13, 14, 15, 16],
            'properties_velocity': [18, 19, 20, 21, 22, 23, 24, 25, 26],
            'properties_torque': [27, 28, 29, 30, 31, 32, 33, 34, 35],
            'skills': [17]
        },
        'UR5': {
            'components': [0, 1, 2, 3, 4, 5, 13],
            'properties_velocity': [],
            'properties_torque': [],
            'skills': [15]
        },
        'Jaco 2': {
            'components': [],
            'properties_velocity': [],
            'properties_torque': [],
            'skills': []
        },
        'OpenManipulator': {
            'components': [3, 4, 5, 6, 7, 8, 9],
            'properties_velocity': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'properties_torque': [],
            'skills': [22, 23, 24]
        },
        'Franka Emika Cable': {
            'components': [0, 1, 2, 3, 4, 5, 6],
            'properties_velocity': [],
            'properties_torque': [],
            'skills': [7]
        }
    }
    # Return the corresponding dictionary of lists or an empty dictionary if the robot type is not found
    return robot_column_mapping.get(robot_type, {'components': [], 'properties': [], 'skills': []})
