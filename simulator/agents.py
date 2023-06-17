class Agents:
    def __init__(self) -> None:
        self.age = [] # int
        self.gender = [] # int
        self.hhid = [] # int
        self.scid = [] # int
        self.sc_student = []
        self.sc_type = []
        self.wpid = []
        self.empstatus = []
        self.empind = []
        self.empftpt = []
        # self.edu = []
        # self.lti = []
        # self.bmi = []
        self.res_cellid = []
        self.work_cellid = []
        self.school_cellid = []
        self.inst_cellid = []
        self.age_bracket_index = []
        self.epi_age_bracket_index = []
        self.working_age_bracket_index = []
        self.soc_rate = []
        self.guardian_id = []
        self.working_schedule = [] # work or school schedule - to change to array
        self.isshiftbased = []
        self.pub_transp_reg = []
        self.ent_activity = []

        self.itinerary = []
        self.itinerary_nextday = []
        self.non_daily_activity_recurring = []
        self.prevday_non_daily_activity_recurring = []

        # self.busdriver - to be handled differently
        # self.state_transition_by_day - will need to handle differently
        # self.test_day
        # self.test_result_day
        # self.quarantine_days
        # self.hospitalisation_days
        # self.vaccination_days

    def populate(self, data):
        for agent_id, properties in data.items():
            self.age.append(properties["age"] if "age" in properties else None)
            self.gender.append(properties["gender"] if "gender" in properties else None)
            self.hhid.append(properties["hhid"] if "hhid" in properties else None) # int
            self.scid.append(properties["scid"] if "scid" in properties else None) # int
            self.sc_student.append(properties["sc_student"] if "sc_student" in properties else None)
            self.sc_type.append(properties["sc_type"] if "sc_type" in properties else None)
            self.wpid.append(properties["wpid"] if "wpid" in properties else None)
            self.empstatus.append(properties["empstatus"] if "empstatus" in properties else None)
            self.empind.append(properties["empind"] if "empind" in properties else None)
            self.empftpt.append(properties["empftpt"] if "empftpt" in properties else None)
            # self.edu.append(properties["edu"] if "edu" in properties else None)
            # self.lti.append(properties["lti"] if "lti" in properties else None)
            # self.bmi.append(properties["bmi"] if "bmi" in properties else None)
            self.res_cellid.append(properties["res_cellid"])
            self.work_cellid.append(properties["work_cellid"])
            self.school_cellid.append(properties["school_cellid"])
            self.inst_cellid.append(properties["inst_cellid"])
            self.age_bracket_index.append(properties["age_bracket_index"] if "age_bracket_index" in properties else None)
            self.epi_age_bracket_index.append(properties["epi_age_bracket_index"] if "epi_age_bracket_index" in properties else None)
            self.working_age_bracket_index.append(properties["working_age_bracket_index"] if "working_age_bracket_index" in properties else None)
            self.soc_rate.append(properties["soc_rate"] if "soc_rate" in properties else None)
            self.guardian_id.append(properties["guardian_id"] if "guardian_id" in properties else None)
            self.working_schedule.append(properties["working_schedule"]) # work or school schedule
            self.isshiftbased.append(None) # this is calculated in working schedule generation on first day
            self.pub_transp_reg.append(properties["pub_transp_reg"])
            self.ent_activity.append(properties["ent_activity"])

            self.itinerary.append(None)
            self.itinerary_nextday.append(None)
            self.non_daily_activity_recurring.append(None)
            self.prevday_non_daily_activity_recurring.append(None)

    def get(self, index, name):
        return getattr(self, name)[index]

    def set(self, index, name, value):
        if name == "age":
            self.age[index] = value
        elif name == "res_cellid":
            self.res_cellid[index] = value
        elif name == "work_cellid":
            self.work_cellid[index] = value
        elif name == "school_cellid":
            self.school_cellid[index] = value
        elif name == "inst_cellid":
            self.inst_cellid[index] = value
        elif name == "age_bracket_index":
            self.age_bracket_index[index] = value
        elif name == "epi_age_bracket_index":
            self.epi_age_bracket_index[index] = value
        elif name == "working_age_bracket_index":
            self.working_age_bracket_index[index] = value
        elif name == "soc_rate":
            self.soc_rate[index] = value
        elif name == "guardian_id":
            self.guardian_id[index] = value
        elif name == "working_schedule":
            self.working_schedule[index] = value
        elif name == "isshiftbased":
            self.isshiftbased[index] = value
        elif name == "pub_transp_reg":
            self.pub_transp_reg[index] = value
        elif name == "ent_activity":
            self.ent_activity[index] = value
        elif name == "itinerary":
            self.itinerary[index] = value
        elif name == "itinerary_nextday":
            self.itinerary_nextday[index] = value
        elif name == "non_daily_activity_recurring":
            self.non_daily_activity_recurring[index] = value
        elif name == "prevday_non_daily_activity_recurring":
            self.prevday_non_daily_activity_recurring[index] = value
        elif name == "gender": # from here onwards, these values are supposed to be readonly
            self.gender[index] = value
        elif name == "hhid":
            self.hhid[index] = value
        elif name == "scid":
            self.scid[index] = value
        elif name == "sc_student":
            self.sc_student[index] = value
        elif name == "sc_type":
            self.sc_type[index] = value
        elif name == "wpid":
            self.wpid[index] = value
        elif name == "empstatus":
            self.empstatus[index] = value