import numpy as np
from enum import IntEnum
from simulator import util
import sys

class TouristsGroups:
    def __init__(self) -> None:
        self.n_total = None

        self.arr = []
        self.dep = []
        self.purpose = []
        self.accomtype = []
        self.reftourid = []
        self.accominfo = [] # [[1581, 1540]]
        self.subgroupsmemberids = [] # [[0, 14, 2]]

        self.shm_arr = []
        self.shm_dep = []
        self.shm_purpose = []
        self.shm_accomtype = []
        self.shm_reftourid = []
        self.shm_accominfo = []
        self.shm_subgroupsmemberids = []

    def clear_non_shared_memory_readonly(self):
        self.arr = []
        self.dep = []
        self.purpose = []
        self.accomtype = []
        self.reftourid = []
        self.accominfo = [] 
        self.subgroupsmemberids = [] 
    
    def populate(self, data):
        for _, properties in data.items():
            self.arr.append(properties["arr"] if "arr" in properties else None)
            self.dep.append(properties["dep"] if "dep" in properties else None)
            self.purpose.append(properties["purpose"] if "purpose" in properties else None)
            self.accomtype.append(properties["accomtype"] if "accomtype" in properties else None)
            self.reftourid.append(properties["reftourid"] if "reftourid" in properties else None)
            self.accominfo.append(properties["accominfo"] if "accominfo" in properties else None)
            self.subgroupsmemberids.append(properties["subgroupsmemberids"] if "subgroupsmemberids" in properties else None)

        self.n_total = len(data)

    def convert_to_shared_memory_readonly(self, clear_normal_memory=False):
        self.shm_arr = util.generate_shared_memory_int(self.arr)
        self.shm_dep = util.generate_shared_memory_int(self.dep)
        self.shm_purpose = util.generate_shared_memory_int(self.purpose)
        self.shm_accomtype = util.generate_shared_memory_int(self.accomtype)
        self.shm_reftourid = util.generate_shared_memory_int(self.reftourid)
        self.shm_accominfo = util.generate_shared_memory_multidim_varying(self.accominfo, 3)
        self.shm_subgroupsmemberids = util.generate_shared_memory_varyingdims_varyinglength(self.subgroupsmemberids)

        if clear_normal_memory:
            self.clear_non_shared_memory_readonly()

    def convert_from_shared_memory_readonly(self):
        self.arr = util.generate_ndarray_from_shared_memory_int(self.shm_arr, self.n_total)
        self.dep = util.generate_ndarray_from_shared_memory_int(self.shm_dep, self.n_total)
        self.purpose = util.generate_ndarray_from_shared_memory_int(self.shm_purpose, self.n_total)
        self.accomtype = util.generate_ndarray_from_shared_memory_int(self.shm_accomtype, self.n_total)
        self.reftourid = util.generate_ndarray_from_shared_memory_int(self.shm_reftourid, self.n_total)
        self.accominfo = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_accominfo, self.n_total)
        self.subgroupsmemberids = util.generate_ndarray_varyingdims_varyinglength(self.shm_subgroupsmemberids, self.n_total)

    def clone(self, cell_to_clone):
        self.n_total = cell_to_clone.n_total

        self.arr = cell_to_clone.type
        self.dep = cell_to_clone.sub_type
        self.purpose = cell_to_clone.indid
        self.accomtype = cell_to_clone.resident_uids
        self.reftourid = cell_to_clone.reftourid
        self.accominfo = cell_to_clone.accominfo
        self.subgroupsmemberids = cell_to_clone.subgroupsmembersids

    def cleanup_shared_memory_readonly(self):
        util.close_shm(self.shm_arr)
        util.close_shm(self.shm_dep)
        util.close_shm(self.shm_purpose)
        util.close_shm(self.shm_accomtype)
        util.close_shm(self.shm_reftourid)
        util.close_shm(self.shm_accominfo)
        util.close_shm(self.shm_subgroupsmemberids)

    def get(self, index, name):
        return getattr(self, name)[index]
    
    def calculate_memory_size(self, attr_name=None):
        total_size = sum([sys.getsizeof(getattr(self, attr)) for attr in dir(self) if attr_name is None or attr==attr_name])
        return total_size

