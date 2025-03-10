"""This module provides functions for reading and writing JSON files using jsonpickle."""
import json
import jsonpickle


def write_json(data, file):
    """
    Serialize the reliability data to a JSON file using jsonpickle.

    Parameters:
        data (any): The reliability data to be serialized.
        file (str): The file path where the JSON data will be stored.
    """
    jsonpickle.set_encoder_options('json', indent=4)
    encoded_data = jsonpickle.encode(data, unpicklable=False)
    json_data = json.dumps(encoded_data)
    decoded_data = jsonpickle.decode(json_data)
    with open(file, 'w') as outfile:
        outfile.write(decoded_data)


def read_json(file):
    """
    Deserialize JSON data from a file back into a Python object using jsonpickle.

    Parameters:
        file (str): The file path from which the JSON data should be read.

    Returns:
        any: The Python object reconstructed from the JSON file.
    """
    with open(file, 'r') as infile:
        data = json.load(infile)
    return data
