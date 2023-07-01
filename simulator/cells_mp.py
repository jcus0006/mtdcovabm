from enum import IntEnum
from simulator import util

class Cells:
    def __init__(self) -> None:
        self.type = []
        self.indid = []
        self.resident_uids = []
        self.member_uids = []
        self.staff_uids = []

        self.cells_agents_timesteps = [] # [[agentid, starttimestep, endtimestep]]

        self.shm_type = []
        self.shm_indid = []
        self.shm_resident_uids = []
        self.shm_member_uids = []
        self.shm_staff_uids = []

        # self.hhid = []
        # self.wpid = []
        # self.scid = []
        # self.clid = []
        # self.indid = []
        # self.activityid = []
        # self.accomid = []
        # self.accomtypeid = []
        # self.roomid = []
        # self.roomsize = []
        # self.guest_uids = []
        # self.accomcellid = []
        # self.resident_uids = []
        # self.staff_uids = []
        # self.visitor_uids = []
        # self.teacher_uids = []
        # self.student_uids = []
        # self.non_teaching_staff_uids = []
        # self.busid = []
        # self.capacity = []
        # self.passenger_uids = []
        # self.churchid = []
    
    def populate(self, data):
        for _, properties in data.items():
            self.type.append(util.convert_celltype_str_to_enum(properties["type"]))

            place = properties["place"]
            self.indid.append(place["indid"] if "indid" in place else None)
            self.resident_uids.append(place["resident_uids"] if "resident_uids" in place else None)
            self.member_uids.append(place["member_uids"] if "member_uids" in place else None)
            self.staff_uids.append(place["staff_uids"] if "staff_uids" in place else None)


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

