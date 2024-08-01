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
