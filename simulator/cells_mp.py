import numpy as np
from enum import IntEnum
from simulator import util

class Cells:
    def __init__(self) -> None:
        self.n_total = None

        self.type = []
        self.sub_type = []
        self.indid = []
        self.resident_uids = []
        self.member_uids = []
        self.staff_uids = []

        self.shm_type = []
        self.shm_sub_type = []
        self.shm_indid = []
        self.shm_resident_uids = []
        self.shm_member_uids = []
        self.shm_staff_uids = []

    def clear_non_shared_memory_readonly(self):
        self.type = []
        self.sub_type = []
        self.indid = []
        self.resident_uids = []
        self.member_uids = []
        self.staff_uids = []
    
    def populate(self, data, testinghubcells, vaccinationhubcells, restaurantcells):
        for cellid, properties in data.items():
            type = util.convert_celltype_str_to_enum(properties["type"])
            self.type.append(type)

            place = properties["place"]
            indid = place["indid"] if "indid" in place else None
            self.indid.append(indid)
            self.resident_uids.append(place["resident_uids"] if "resident_uids" in place else None)
            self.member_uids.append(place["member_uids"] if "member_uids" in place else None)
            self.staff_uids.append(place["staff_uids"] if "staff_uids" in place else None)

            if type == CellType.Hospital:
                if cellid in testinghubcells:
                    self.sub_type.append(CellSubType.TestingHub)
                elif cellid in vaccinationhubcells:
                    self.sub_type.append(CellSubType.VaccinationHub)
                else:
                    self.sub_type.append(None)
            elif indid == 9 and cellid in restaurantcells:
                self.sub_type.append(CellSubType.Restaurant)
            else:
                self.sub_type.append(None)

        self.n_total = len(data)

    def convert_to_shared_memory_readonly(self, clear_normal_memory=False):
        self.shm_type = util.generate_shared_memory_int(self.type)
        self.shm_sub_type = util.generate_shared_memory_int(self.sub_type)
        self.shm_indid = util.generate_shared_memory_int(self.indid)
        self.shm_resident_uids = util.generate_shared_memory_singledim_varying(self.resident_uids, 1)
        self.shm_member_uids = util.generate_shared_memory_singledim_varying(self.member_uids, 1)
        self.shm_staff_uids = util.generate_shared_memory_singledim_varying(self.staff_uids, 1)

        if clear_normal_memory:
            self.clear_non_shared_memory_readonly()

    def convert_from_shared_memory_readonly(self):
        self.type = util.generate_ndarray_from_shared_memory_int(self.shm_type, self.n_total)
        self.sub_type = util.generate_ndarray_from_shared_memory_int(self.shm_sub_type, self.n_total)
        self.indid = util.generate_ndarray_from_shared_memory_int(self.shm_indid, self.n_total)
        self.resident_uids = util.generate_ndarray_from_shared_memory_singledim_varying(self.shm_resident_uids, self.n_total)
        self.member_uids = util.generate_ndarray_from_shared_memory_singledim_varying(self.shm_member_uids, self.n_total)
        self.staff_uids = util.generate_ndarray_from_shared_memory_singledim_varying(self.shm_staff_uids, self.n_total)

    def clone(self, cell_to_clone):
        self.n_total = cell_to_clone.n_total

        self.type = cell_to_clone.type
        self.sub_type = cell_to_clone.sub_type
        self.indid = cell_to_clone.indid
        self.resident_uids = cell_to_clone.resident_uids
        self.member_uids = cell_to_clone.member_uids
        self.staff_uids = cell_to_clone.staff_uids

    def cleanup_shared_memory_readonly(self):
        util.close_shm(self.shm_type)
        util.close_shm(self.shm_sub_type)
        util.close_shm(self.shm_indid)
        util.close_shm(self.shm_resident_uids)
        util.close_shm(self.shm_member_uids)
        util.close_shm(self.shm_staff_uids)

    def get(self, index, name):
        return getattr(self, name)[index]
    
    def get_keys(self, type=None, sub_type=None):
        if type is not None:
            return np.array([i for i, t in enumerate(self.type) if t is not None and t == type])
        elif sub_type is not None:
            return np.array([i for i, t in enumerate(self.sub_type) if t is not None and t == sub_type])
        else:
            return []
    
    # def set(self, index, name, value):
    #     if name == "type":
    #         self.type[index] = value
    #     elif name == "sub_type":
    #         self.sub_type[index] = value
    #     elif name == "indid":
    #         self.indid[index] = value
    #     elif name == "resident_uids":
    #         self.resident_uids[index] = value
    #     elif name == "member_uids":
    #         self.member_uids[index] = value
    #     elif name == "staff_uids":
    #         self.staff_uids[index] = value

class CellType(IntEnum):
    Undefined = 0
    Household = 1
    Workplace = 2
    Accommodation = 3
    Hospital = 4
    Entertainment = 5
    School = 6
    Institution = 7
    Transport = 8
    Religion = 9
    Airport = 10

class CellSubType(IntEnum):
    Undefined = 0
    Restaurant = 1
    TestingHub = 2
    VaccinationHub = 3

