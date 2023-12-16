from collections import UserDict

class CustomDict(UserDict):
    pass

class PartialCustomDict:
    def __init__(self, original_dict, keys_to_keep, raise_key_error=False):
        self.original_dict = original_dict
        self.keys_to_keep = keys_to_keep
        self.raise_key_error = raise_key_error

    def __getitem__(self, key):
        if key in self.keys_to_keep:
            return self.original_dict[key]
        else:
            if self.raise_key_error:
                raise KeyError(f"Key '{key}' not found in partial dictionary")
        
    def __setitem__(self, key, value):
        if key in self.keys_to_keep:
            self.original_dict[key] = value
        else:
            if self.raise_key_error:
                raise KeyError(f"Key '{key}' not found in partial dictionary")