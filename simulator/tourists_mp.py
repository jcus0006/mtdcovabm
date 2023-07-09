import numpy as np
from enum import IntEnum
from simulator import util
import sys

class Tourists:
    def __init__(self) -> None:
        self.n_total = None

        self.groupid = []
        self.subgroupid = []
        self.age = []
        self.gender = []

        self.shm_groupid = []
        self.shm_subgroupid = []
        self.shm_age = []
        self.shm_gender = []

    def clear_non_shared_memory_readonly(self):
        self.groupid = []
        self.subgroupid = []
        self.age = []
        self.gender = []
    
    def populate(self, data):
        for _, properties in data.items():
            self.groupid.append(properties["groupid"] if "groupid" in properties else None)
            self.subgroupid.append(properties["subgroupid"] if "subgroupid" in properties else None)
            self.age.append(properties["age"] if "age" in properties else None)
            self.gender.append(properties["gender"] if "gender" in properties else None)

        self.n_total = len(data)

    def convert_to_shared_memory_readonly(self, clear_normal_memory=False):
        self.shm_groupid = util.generate_shared_memory_int(self.groupid)
        self.shm_subgroupid = util.generate_shared_memory_int(self.subgroupid)
        self.shm_age = util.generate_shared_memory_int(self.age)
        self.shm_gender = util.generate_shared_memory_int(self.gender)

        if clear_normal_memory:
            self.clear_non_shared_memory_readonly()

    def convert_from_shared_memory_readonly(self):
        self.groupid = util.generate_ndarray_from_shared_memory_int(self.shm_groupid, self.n_total)
        self.subgroupid = util.generate_ndarray_from_shared_memory_int(self.shm_subgroupid, self.n_total)
        self.age = util.generate_ndarray_from_shared_memory_int(self.shm_age, self.n_total)
        self.gender = util.generate_ndarray_from_shared_memory_int(self.shm_gender, self.n_total)

    def clone(self, cell_to_clone):
        self.n_total = cell_to_clone.n_total

        self.groupid = cell_to_clone.type
        self.subgroupid = cell_to_clone.sub_type
        self.age = cell_to_clone.indid
        self.gender = cell_to_clone.resident_uids

    def cleanup_shared_memory_readonly(self):
        util.close_shm(self.shm_groupid)
        util.close_shm(self.shm_subgroupid)
        util.close_shm(self.shm_age)
        util.close_shm(self.shm_gender)

    def get(self, index, name):
        return getattr(self, name)[index]
    
    def calculate_memory_size(self, attr_name=None):
        total_size = sum([sys.getsizeof(getattr(self, attr)) for attr in dir(self) if attr_name is None or attr==attr_name])
        return total_size

