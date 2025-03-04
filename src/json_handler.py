import json
import jsonpickle


def write_json(data, file):
    jsonpickle.set_encoder_options('json', indent=4)
    encoded_data = jsonpickle.encode(data, unpicklable=False)
    json_data = json.dumps(encoded_data)
    decoded_data = jsonpickle.decode(json_data)
    with open(file, 'w') as outfile:
        outfile.write(decoded_data)


def read_json(file):
    with open(file, 'r') as infile:
        data = json.load(infile)
    return data

