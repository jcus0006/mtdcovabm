import os
import json
from npencoder import NpEncoder

def jsonKeys2int(x):
    new_dict = {}
    for k, v in x.items():
        try:
            new_key = int(k)
        except:
            new_key = k
        if type(v) == dict:
            v = jsonKeys2int(v)
        new_dict[new_key] = v
    return new_dict

def convert_to_json_file(current_directory, folder, subfolder, filename, array, custom_encoder=None):
    temp_filepath = os.path.join(current_directory, folder, subfolder, filename)
    if os.path.exists(temp_filepath):
        os.remove(temp_filepath)

    if custom_encoder is None:
        custom_encoder = NpEncoder

    with open(temp_filepath, "w", encoding="utf-8") as fp:
        json.dump(array, fp, ensure_ascii=False, indent=4, cls=custom_encoder)

    return temp_filepath

def convert_to_json_str(array, custom_encoder=None, is_class=False):
    if custom_encoder is None:
        custom_encoder = NpEncoder

    if not is_class:
        return json.dumps(array, ensure_ascii=False, indent=4, cls=custom_encoder)
    else:
        return json.dumps(array, default=lambda o: o.__dict__, indent=4)