
def classify_property_dict(property_dict):
    """
    Classifies properties based on their type and values provided in a dictionary for each skill.

    Parameters:
    - property_dict (dict): A dictionary where the key is the property type ('velocity' or 'torque') and the value is a list of float values.

    Returns:
    - dict: A dictionary with the same keys as input and the classified values as a list of strings.
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
