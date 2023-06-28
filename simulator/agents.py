import numpy as np
import sys
import struct
import time
import multiprocessing.shared_memory as shm
from simulator import util

class Agents:
    def __init__(self) -> None:
        self.n_total = None
        self.n_locals = None
        self.n_tourists = None
        # self.age = [] # int
        # self.gender = [] # int
        # self.hhid = [] # int
        # self.scid = [] # int
        self.sc_student = []
        # self.sc_type = []
        # self.wpid = []
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
        self.working_schedule = [] # work or school schedule - to change to array (done)
        self.isshiftbased = []
        self.pub_transp_reg = []
        self.ent_activity = []

        self.itinerary = []
        self.itinerary_nextday = []
        self.non_daily_activity_recurring = []
        self.prevday_non_daily_activity_recurring = []

        self.busdriver = []
        self.state_transition_by_day = []
        self.test_day = []
        self.test_result_day = []
        self.quarantine_days = []
        self.hospitalisation_days = []
        self.vaccination_days = []

        # self.shm_age = [] # int
        # self.shm_gender = [] # int
        # self.shm_hhid = [] # int
        # self.shm_scid = [] # int
        self.shm_sc_student = []
        # self.sc_type = []
        # self.shm_wpid = []
        self.shm_empstatus = []
        self.shm_empind = []
        self.shm_empftpt = []
        # self.shm_edu = []
        # self.shm_lti = []
        # self.shm_bmi = []
        self.shm_res_cellid = []
        self.shm_work_cellid = []
        self.shm_school_cellid = []
        self.shm_inst_cellid = []
        self.shm_age_bracket_index = []
        self.shm_epi_age_bracket_index = []
        self.shm_working_age_bracket_index = []
        self.shm_soc_rate = []
        self.shm_guardian_id = []
        self.shm_working_schedule = [] # work or school schedule - to change to array (done)
        self.shm_isshiftbased = []
        self.shm_pub_transp_reg = []
        self.shm_ent_activity = []

        self.shm_itinerary = []
        self.shm_itinerary_nextday = []
        self.shm_non_daily_activity_recurring = []
        self.shm_prevday_non_daily_activity_recurring = []

        self.shm_busdriver = []
        self.shm_state_transition_by_day = []
        self.shm_test_day = []
        self.shm_test_result_day = []
        self.shm_quarantine_days = []
        self.shm_hospitalisation_days = []
        self.shm_vaccination_days = []

    def populate(self, data, n_locals, n_tourists):
        start = time.time()

        n_total = n_locals + n_tourists

        self.n_total = n_total
        self.n_locals = n_locals
        self.n_tourists = n_tourists

        for _, properties in data.items():
            # self.age.append(properties["age"] if "age" in properties else None)
            # self.gender.append(properties["gender"] if "gender" in properties else None)
            # self.hhid.append(properties["hhid"] if "hhid" in properties else None) # int
            # self.scid.append(properties["scid"] if "scid" in properties else None) # int
            self.sc_student.append(properties["sc_student"] if "sc_student" in properties else None)
            # self.sc_type.append(properties["sc_type"] if "sc_type" in properties else None)
            # self.wpid.append(properties["wpid"] if "wpid" in properties else None)
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

            self.busdriver.append(None)
            self.state_transition_by_day.append(None)
            self.test_day.append(None)
            self.test_result_day.append(None)
            self.quarantine_days.append(None)
            self.hospitalisation_days.append(None)
            self.vaccination_days.append(None)

        time_taken = time.time() - start
        print("agents_mp populate time taken: " + str(time_taken))
        # self.convert_to_ndarray()

    def clone(self, agents_mp_to_clone, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True

        self.n_total = agents_mp_to_clone.n_total
        self.n_locals = agents_mp_to_clone.n_locals
        self.n_tourists = agents_mp_to_clone.n_tourists

        if loadall:
            self.sc_student, self.shm_sc_student = agents_mp_to_clone.sc_student, agents_mp_to_clone.shm_sc_student
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.shm_sc_type))
            # self.wpid = self.generate_ndarray_from_shared_memory_int(self.shm_wpid)
            self.empstatus, self.shm_empstatus = agents_mp_to_clone.empstatus, agents_mp_to_clone.shm_empstatus
            self.empind, self.shm_empind = agents_mp_to_clone.empind, agents_mp_to_clone.shm_empind
            self.empftpt, self.shm_empftpt = agents_mp_to_clone.empftpt, agents_mp_to_clone.shm_empftpt
            self.res_cellid, self.shm_res_cellid = agents_mp_to_clone.res_cellid, agents_mp_to_clone.shm_res_cellid
            self.work_cellid, self.shm_work_cellid = agents_mp_to_clone.work_cellid, agents_mp_to_clone.shm_work_cellid
            self.school_cellid, self.shm_school_cellid = agents_mp_to_clone.school_cellid, agents_mp_to_clone.shm_school_cellid
            self.inst_cellid, self.shm_inst_cellid = agents_mp_to_clone.inst_cellid, agents_mp_to_clone.shm_inst_cellid
            self.age_bracket_index, self.shm_age_bracket_index = agents_mp_to_clone.age_bracket_index, agents_mp_to_clone.shm_age_bracket_index
            self.epi_age_bracket_index, self.shm_epi_age_bracket_index = agents_mp_to_clone.epi_age_bracket_index, agents_mp_to_clone.shm_epi_age_bracket_index
            self.working_age_bracket_index, self.shm_working_age_bracket_index = agents_mp_to_clone.working_age_bracket_index, agents_mp_to_clone.shm_working_age_bracket_index
            self.soc_rate, self.shm_soc_rate = agents_mp_to_clone.soc_rate, agents_mp_to_clone.shm_soc_rate
            self.guardian_id, self.shm_guardian_id = agents_mp_to_clone.guardian_id, agents_mp_to_clone.shm_guardian_id
            self.pub_transp_reg, self.shm_pub_transp_reg = agents_mp_to_clone.pub_transp_reg, agents_mp_to_clone.shm_pub_transp_reg
            self.ent_activity, self.shm_ent_activity = agents_mp_to_clone.ent_activity, agents_mp_to_clone.shm_ent_activity
            self.isshiftbased, self.shm_isshiftbased = agents_mp_to_clone.isshiftbased, agents_mp_to_clone.shm_isshiftbased
            self.busdriver, self.shm_busdriver = agents_mp_to_clone.busdriver, agents_mp_to_clone.shm_busdriver
            self.working_schedule, self.shm_working_schedule = agents_mp_to_clone.working_schedule, agents_mp_to_clone.shm_working_schedule
        elif contactnetwork:
            self.age_bracket_index, self.shm_age_bracket_index = agents_mp_to_clone.age_bracket_index, agents_mp_to_clone.shm_age_bracket_index
            self.soc_rate, self.shm_soc_rate = agents_mp_to_clone.soc_rate, agents_mp_to_clone.shm_soc_rate
            self.epi_age_bracket_index, self.shm_epi_age_bracket_index = agents_mp_to_clone.epi_age_bracket_index, agents_mp_to_clone.shm_epi_age_bracket_index
            self.res_cellid, self.shm_res_cellid = agents_mp_to_clone.res_cellid, agents_mp_to_clone.shm_res_cellid
        elif itinerary:
            self.sc_student, self.shm_sc_student = agents_mp_to_clone.sc_student, agents_mp_to_clone.shm_sc_student
            self.empstatus, self.shm_empstatus = agents_mp_to_clone.empstatus, agents_mp_to_clone.shm_empstatus
            self.empind, self.shm_empind = agents_mp_to_clone.empind, agents_mp_to_clone.shm_empind
            self.ent_activity, self.shm_ent_activity = agents_mp_to_clone.ent_activity, agents_mp_to_clone.shm_ent_activity
            self.isshiftbased, self.shm_isshiftbased = agents_mp_to_clone.isshiftbased, agents_mp_to_clone.shm_isshiftbased
            self.empftpt, self.shm_empftpt = agents_mp_to_clone.empftpt, agents_mp_to_clone.shm_empftpt
            self.guardian_id, self.shm_guardian_id = agents_mp_to_clone.guardian_id, agents_mp_to_clone.shm_guardian_id
            self.age_bracket_index, self.shm_age_bracket_index = agents_mp_to_clone.age_bracket_index, agents_mp_to_clone.shm_age_bracket_index
            self.epi_age_bracket_index, self.shm_epi_age_bracket_index = agents_mp_to_clone.epi_age_bracket_index, agents_mp_to_clone.shm_epi_age_bracket_index
            self.working_age_bracket_index, self.shm_working_age_bracket_index = agents_mp_to_clone.working_age_bracket_index, agents_mp_to_clone.shm_working_age_bracket_index
            self.res_cellid, self.shm_res_cellid = agents_mp_to_clone.res_cellid, agents_mp_to_clone.shm_res_cellid
            self.work_cellid, self.shm_work_cellid = agents_mp_to_clone.work_cellid, agents_mp_to_clone.shm_work_cellid
            self.school_cellid, self.shm_school_cellid = agents_mp_to_clone.school_cellid, agents_mp_to_clone.shm_school_cellid
            self.pub_transp_reg, self.shm_pub_transp_reg = agents_mp_to_clone.pub_transp_reg, agents_mp_to_clone.shm_pub_transp_reg
            self.working_schedule, self.shm_working_schedule = agents_mp_to_clone.working_schedule, agents_mp_to_clone.shm_working_schedule

    def clear_non_shared_memory_readonly(self):
        # self.age = None
        # self.gender = None
        # self.hhid = None
        # self.scid = None
        self.sc_student = None
        # self.sc_type = None
        # self.wpid = None
        self.empstatus = None
        self.empind = None
        self.empftpt = None
        # self.edu = None
        # self.lti = None
        # self.bmi = None
        self.res_cellid = None
        self.work_cellid = None
        self.school_cellid = None
        self.inst_cellid = None
        self.age_bracket_index = None
        self.epi_age_bracket_index = None
        self.working_age_bracket_index = None
        self.soc_rate = None
        self.guardian_id = None
        # self.working_schedule = None
        # self.isshiftbased = None
        self.pub_transp_reg = None
        self.ent_activity = None
    
    def clear_non_shared_memory_workingschedule(self):
        self.working_schedule = None

    def clear_non_shared_memory_isshiftbased(self):
        self.isshiftbased = None

    def clear_non_shared_memory_dynamic(self):
        self.itinerary = None
        self.itinerary_nextday = None
        self.non_daily_activity_recurring = None
        self.prevday_non_daily_activity_recurring = None

        self.busdriver = None
        self.state_transition_by_day = None
        self.test_day = None
        self.test_result_day = None
        self.quarantine_days = None
        self.hospitalisation_days = None
        self.vaccination_days = None


    # def convert_to_ndarray(self):
    #     self.age = np.array(self.age)
    #     self.gender = np.array(self.gender)
    #     self.hhid = np.array(self.hhid)
    #     self.scid = np.array(self.scid)
    #     self.sc_student = np.array(self.sc_student)
    # #     self.sc_type = np.array(self.sc_type)
    #     self.wpid = np.array(self.wpid)
    #     self.empstatus = np.array(self.empstatus)
    #     self.empind = np.array(self.empind)
    #     self.empftpt = np.array(self.empftpt)
    #     # self.edu = []
    #     # self.lti = []
    #     # self.bmi = []
    #     self.res_cellid = np.array(self.res_cellid)
    #     self.work_cellid = np.array(self.work_cellid)
    #     self.school_cellid = np.array(self.school_cellid)
    #     self.inst_cellid = np.array(self.inst_cellid)
    #     self.age_bracket_index = np.array(self.age_bracket_index)
    #     self.epi_age_bracket_index = np.array(self.epi_age_bracket_index)
    #     self.working_age_bracket_index = np.array(self.working_age_bracket_index)
    #     self.soc_rate = np.array(self.soc_rate)
    #     self.guardian_id = np.array(self.guardian_id)
    #     self.working_schedule = np.array(self.working_schedule) # work or school schedule - to change to array (done)
    #     self.isshiftbased = np.array(self.isshiftbased)
    #     self.pub_transp_reg = np.array(self.pub_transp_reg)
    #     self.ent_activity = np.array(self.ent_activity)

    #     self.itinerary = np.array(self.itinerary)
    #     self.itinerary_nextday = np.array(self.itinerary_nextday)
    #     self.non_daily_activity_recurring = np.array(self.non_daily_activity_recurring)
    #     self.prevday_non_daily_activity_recurring = np.array(self.prevday_non_daily_activity_recurring)

    #     self.busdriver = np.array(self.busdriver)
    #     self.state_transition_by_day = np.array(self.state_transition_by_day)
    #     self.test_day = np.array(self.test_day)
    #     self.test_result_day = np.array(self.test_result_day)
    #     self.quarantine_days = np.array(self.quarantine_days)
    #     self.hospitalisation_days = np.array(self.hospitalisation_days)
    #     self.vaccination_days = np.array(self.vaccination_days)

    # def get_shm(self, index, name):
    #     combined = getattr(self, name)

    #     if combined is not None:
    #         arr, mask = combined

    #         if mask[index]:
    #             return arr[index]
        
    #     return None
    
    # def set_shm(self, index, name, value):
    #     if name == "age":
    #         self.age[0][index] = value
    #     elif name == "res_cellid":
    #         self.res_cellid[0][index] = value
    #     elif name == "work_cellid":
    #         self.work_cellid[0][index] = value
    #     elif name == "school_cellid":
    #         self.school_cellid[0][index] = value
    #     elif name == "inst_cellid":
    #         self.inst_cellid[0][index] = value
    #     elif name == "age_bracket_index":
    #         self.age_bracket_index[0][index] = value
    #     elif name == "epi_age_bracket_index":
    #         self.epi_age_bracket_index[0][index] = value
    #     elif name == "working_age_bracket_index":
    #         self.working_age_bracket_index[0][index] = value
    #     elif name == "soc_rate":
    #         self.soc_rate[0][index] = value
    #     elif name == "guardian_id":
    #         self.guardian_id[0][index] = value
    #     elif name == "working_schedule":
    #         self.working_schedule[0][index] = value
    #     elif name == "isshiftbased":
    #         self.isshiftbased[0][index] = value
    #     elif name == "pub_transp_reg":
    #         self.pub_transp_reg[0][index] = value
    #     elif name == "ent_activity":
    #         self.ent_activity[0][index] = value
    #     elif name == "itinerary":
    #         self.itinerary[0][index] = value
    #     elif name == "itinerary_nextday":
    #         self.itinerary_nextday[0][index] = value
    #     elif name == "non_daily_activity_recurring":
    #         self.non_daily_activity_recurring[0][index] = value
    #     elif name == "prevday_non_daily_activity_recurring":
    #         self.prevday_non_daily_activity_recurring[0][index] = value
    #     elif name == "gender": # from here onwards, these values are supposed to be readonly
    #         self.gender[0][index] = value
    #     elif name == "hhid":
    #         self.hhid[0][index] = value
    #     elif name == "scid":
    #         self.scid[0][index] = value
    #     elif name == "sc_student":
    #         self.sc_student[0][index] = value
    #     # elif name == "sc_type":
    #     #     self.sc_type[index] = value
    #     elif name == "wpid":
    #         self.wpid[0][index] = value
    #     elif name == "empstatus":
    #         self.empstatus[0][index] = value
    #     elif name == "busdriver":
    #         self.busdriver[0][index] = value
    #     elif name =="state_transition_by_day":
    #         self.state_transition_by_day[0][index] = value
    #     elif name == "test_day":
    #         self.test_day[0][index] = value
    #     elif name == "test_result_day":
    #         self.test_result_day[0][index] = value
    #     elif name == "quarantine_days":
    #         self.quarantine_days[0][index] = value
    #     elif name == "hospitalisation_days":
    #         self.hospitalisation_days[0][index] = value
    #     elif name == "vaccination_days":
    #         self.vaccination_days[0][index] = value

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
        # elif name == "sc_type":
        #     self.sc_type[index] = value
        elif name == "wpid":
            self.wpid[index] = value
        elif name == "empstatus":
            self.empstatus[index] = value
        elif name == "busdriver":
            self.busdriver[index] = value
        elif name =="state_transition_by_day":
            self.state_transition_by_day[index] = value
        elif name == "test_day":
            self.test_day[index] = value
        elif name == "test_result_day":
            self.test_result_day[index] = value
        elif name == "quarantine_days":
            self.quarantine_days[index] = value
        elif name == "hospitalisation_days":
            self.hospitalisation_days[index] = value
        elif name == "vaccination_days":
            self.vaccination_days[index] = value

    def convert_to_shared_memory_readonly(self, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            # self.shm_age = self.generate_shared_memory_int(self.age)
            # self.shm_gender = self.generate_shared_memory_int(self.gender)
            # self.shm_hhid = self.generate_shared_memory_int(self.hhid)
            # self.shm_scid = self.generate_shared_memory_int(self.scid)
            self.shm_sc_student = self.generate_shared_memory_int(self.sc_student)
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.sc_type))
            # self.shm_wpid = self.generate_shared_memory_int(self.wpid)
            self.shm_empstatus = self.generate_shared_memory_int(self.empstatus)
            self.shm_empind = self.generate_shared_memory_int(self.empind)
            self.shm_empftpt = self.generate_shared_memory_int(self.empftpt)

            self.shm_work_cellid = self.generate_shared_memory_int(self.work_cellid)
            self.shm_school_cellid = self.generate_shared_memory_int(self.school_cellid)
            self.shm_inst_cellid = self.generate_shared_memory_int(self.inst_cellid)
            self.shm_working_age_bracket_index = self.generate_shared_memory_int(self.working_age_bracket_index)

            self.shm_guardian_id = self.generate_shared_memory_int(self.guardian_id)
            self.shm_pub_transp_reg = self.generate_shared_memory_int(self.pub_transp_reg)
            self.shm_ent_activity = self.generate_shared_memory_int(self.ent_activity)
            self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)
            self.shm_busdriver = self.generate_shared_memory_int(self.busdriver)

            self.shm_res_cellid = self.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = self.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_shared_memory_int(self.soc_rate, float)
        elif contactnetwork:
            self.shm_res_cellid = self.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = self.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_shared_memory_int(self.soc_rate, float)
        elif itinerary:
            self.shm_sc_student = self.generate_shared_memory_int(self.sc_student)
            self.shm_empstatus = self.generate_shared_memory_int(self.empstatus)
            self.shm_empind = self.generate_shared_memory_int(self.empind)
            self.shm_empftpt = self.generate_shared_memory_int(self.empftpt)
            self.shm_ent_activity = self.generate_shared_memory_int(self.ent_activity)
            self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)
            self.shm_guardian_id = self.generate_shared_memory_int(self.guardian_id)
            self.shm_age_bracket_index = self.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_working_age_bracket_index = self.generate_shared_memory_int(self.working_age_bracket_index)
            self.shm_res_cellid = self.generate_shared_memory_int(self.res_cellid)
            self.shm_work_cellid = self.generate_shared_memory_int(self.work_cellid)
            self.shm_school_cellid = self.generate_shared_memory_int(self.school_cellid)
            self.shm_inst_cellid = self.generate_shared_memory_int(self.inst_cellid)
            self.shm_pub_transp_reg = self.generate_shared_memory_int(self.pub_transp_reg)

        time_taken = time.time() - start
        print("agents_mp convert_to_shared_memory_readonly time taken: " + str(time_taken))

    # def convert_to_shared_memory_workingschedule(self):            
    #     self.shm_working_schedule = self.generate_shared_memory_threedim_varying(self.working_schedule)

    # def convert_to_shared_memory_isshiftbased(self):
    #     self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)

    def convert_to_shared_memory_dynamic(self, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True

        if loadall or itinerary:
            start = time.time()

            self.shm_itinerary = self.generate_shared_memory_threedim_varying(self.itinerary)
            self.shm_itinerary_nextday = self.generate_shared_memory_threedim_varying(self.itinerary_nextday)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (itinerary) time taken: " + str(time_taken))

            start = time.time()

            self.shm_non_daily_activity_recurring = self.generate_shared_memory_threedim_single(self.non_daily_activity_recurring)
            self.shm_prevday_non_daily_activity_recurring = self.generate_shared_memory_threedim_single(self.prevday_non_daily_activity_recurring)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (non_daily_activity_recurring) time taken: " + str(time_taken))

            start = time.time()

            self.shm_state_transition_by_day = self.generate_shared_memory_threedim_varying(self.state_transition_by_day)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.shm_test_day = self.generate_shared_memory_twodim_single(self.test_day)
            self.shm_test_result_day = self.generate_shared_memory_twodim_single(self.test_result_day)
            self.shm_quarantine_days = self.generate_shared_memory_twodim_varying(self.quarantine_days)
            self.shm_hospitalisation_days = self.generate_shared_memory_threedim_single(self.hospitalisation_days)
            self.shm_vaccination_days = self.generate_shared_memory_threedim_single(self.hospitalisation_days)  

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (epi) time taken: " + str(time_taken))
        elif contactnetwork:
            start = time.time()

            self.shm_state_transition_by_day = self.generate_shared_memory_threedim_varying(self.state_transition_by_day)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.shm_test_day = self.generate_shared_memory_twodim_single(self.test_day)
            self.shm_test_result_day = self.generate_shared_memory_twodim_single(self.test_result_day)
            self.shm_quarantine_days = self.generate_shared_memory_twodim_varying(self.quarantine_days)
            self.shm_hospitalisation_days = self.generate_shared_memory_threedim_single(self.hospitalisation_days)
            self.shm_vaccination_days = self.generate_shared_memory_threedim_single(self.vaccination_days)  

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (epi) time taken: " + str(time_taken))

    def convert_from_shared_memory_readonly(self, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            # self.age = self.generate_ndarray_from_shared_memory_int(self.shm_age)
            # self.gender = self.generate_ndarray_from_shared_memory_int(self.shm_gender)
            # self.hhid = self.generate_ndarray_from_shared_memory_int(self.shm_hhid)
            # self.scid = self.generate_ndarray_from_shared_memory_int(self.shm_scid)
            self.sc_student = self.generate_ndarray_from_shared_memory_int(self.shm_sc_student)
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.shm_sc_type))
            # self.wpid = self.generate_ndarray_from_shared_memory_int(self.shm_wpid)
            self.empstatus = self.generate_ndarray_from_shared_memory_int(self.shm_empstatus)
            self.empind = self.generate_ndarray_from_shared_memory_int(self.shm_empind)
            self.empftpt = self.generate_ndarray_from_shared_memory_int(self.shm_empftpt)
            self.res_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_res_cellid)
            self.work_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_work_cellid)
            self.school_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_school_cellid)
            self.inst_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_inst_cellid)
            self.age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index)
            self.epi_age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index)
            self.working_age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_working_age_bracket_index)
            self.soc_rate = self.generate_ndarray_from_shared_memory_int(self.shm_soc_rate, float)
            self.guardian_id = self.generate_ndarray_from_shared_memory_int(self.shm_guardian_id)
            self.pub_transp_reg = self.generate_ndarray_from_shared_memory_int(self.shm_pub_transp_reg)
            self.ent_activity = self.generate_ndarray_from_shared_memory_int(self.shm_ent_activity)
            self.isshiftbased = self.generate_ndarray_from_shared_memory_int(self.shm_isshiftbased)  
            self.busdriver = self.generate_ndarray_from_shared_memory_int(self.shm_busdriver)
        elif contactnetwork:
            self.age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index)
            self.soc_rate = self.generate_ndarray_from_shared_memory_int(self.shm_soc_rate, float)
            self.epi_age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index)
            self.res_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_res_cellid)
        elif itinerary:
            self.sc_student = self.generate_ndarray_from_shared_memory_int(self.shm_sc_student)
            self.empstatus = self.generate_ndarray_from_shared_memory_int(self.shm_empstatus)
            self.empind = self.generate_ndarray_from_shared_memory_int(self.shm_empind)
            self.ent_activity = self.generate_ndarray_from_shared_memory_int(self.shm_ent_activity)
            self.isshiftbased = self.generate_ndarray_from_shared_memory_int(self.shm_isshiftbased)
            self.empftpt = self.generate_ndarray_from_shared_memory_int(self.shm_empftpt)
            self.guardian_id = self.generate_ndarray_from_shared_memory_int(self.shm_guardian_id)
            self.age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index)
            self.working_age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_working_age_bracket_index)
            self.res_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_res_cellid)
            self.epi_age_bracket_index = self.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index)
            self.work_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_work_cellid)
            self.school_cellid = self.generate_ndarray_from_shared_memory_int(self.shm_school_cellid)
            self.pub_transp_reg = self.generate_ndarray_from_shared_memory_int(self.shm_pub_transp_reg)

        time_taken = time.time() - start
        print("agents_mp convert_from_shared_memory_readonly time taken: " + str(time_taken))

    # def convert_from_shared_memory_workingschedule(self):
    #     self.working_schedule = self.generate_ndarray_from_shared_memory_threedim_varying(self.shm_working_schedule)

    # def convert_from_shared_memory_isshiftbased(self):
    #     self.isshiftbased = self.generate_ndarray_from_shared_memory_int(self.shm_isshiftbased)        

    def convert_from_shared_memory_dynamic(self, loadall=False, itinerary=False, contactnetwork=False): 
        if not loadall and not itinerary and not contactnetwork:
            loadall = True

        if loadall or itinerary:
            start = time.time()

            self.itinerary = self.generate_ndarray_from_shared_memory_threedim_varying(self.shm_itinerary)
            self.itinerary_nextday = self.generate_ndarray_from_shared_memory_threedim_varying(self.shm_itinerary_nextday)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (itinerary) time taken: " + str(time_taken))

            start = time.time()

            self.non_daily_activity_recurring = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_non_daily_activity_recurring)
            self.prevday_non_daily_activity_recurring = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_prevday_non_daily_activity_recurring)
            
            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (non_daily_activity_recurring) time taken: " + str(time_taken))

            start = time.time()

            self.state_transition_by_day = self.generate_ndarray_from_shared_memory_threedim_varying(self.shm_state_transition_by_day)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()

            self.test_day = self.generate_ndarray_from_shared_memory_twodim_single(self.shm_test_day)
            self.test_result_day = self.generate_ndarray_from_shared_memory_twodim_single(self.shm_test_result_day)
            self.quarantine_days = self.generate_ndarray_from_shared_memory_twodim_varying(self.shm_quarantine_days)
            self.hospitalisation_days = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_hospitalisation_days)
            self.vaccination_days = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_vaccination_days)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (epi) time taken: " + str(time_taken))
        else:
            start = time.time()

            self.state_transition_by_day = self.generate_ndarray_from_shared_memory_threedim_varying(self.shm_state_transition_by_day)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.test_day = self.generate_ndarray_from_shared_memory_twodim_single(self.shm_test_day)
            self.test_result_day = self.generate_ndarray_from_shared_memory_twodim_single(self.shm_test_result_day)
            self.quarantine_days = self.generate_ndarray_from_shared_memory_twodim_varying(self.shm_quarantine_days)
            self.hospitalisation_days = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_hospitalisation_days)
            self.vaccination_days = self.generate_ndarray_from_shared_memory_threedim_single(self.shm_vaccination_days)  

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (epi) time taken: " + str(time_taken))

    def generate_shared_memory_int(self, data, type=int):
        # Create a separate Boolean array to track valid/invalid elements
        valid_mask = [x is not None for x in data]

        valid_len = sum(bool(x) for x in valid_mask)

        if valid_len == 0:
            return None
        
        # Create a shared memory block for the data array
        data_shm = shm.SharedMemory(create=True, size=valid_len * np.dtype(type).itemsize)

        # Store the data in the shared memory
        data_array = np.ndarray(valid_len, dtype=type, buffer=data_shm.buf)

        data_array_index = 0
        for i in range(len(data)):
            if valid_mask[i]:
                data_array[data_array_index] = data[i]
                data_array_index += 1

        # data_array[valid_mask] = [x for x in data if x is not None]

        # Create a shared memory block for the valid mask
        mask_shm = shm.SharedMemory(create=True, size=len(valid_mask) * np.dtype(bool).itemsize)

        # Store the valid mask in the shared memory
        mask_array = np.ndarray(len(valid_mask), dtype=bool, buffer=mask_shm.buf)
        mask_array[:] = valid_mask

        return [data_shm, mask_shm, data_array.shape, mask_array.shape]

    def generate_ndarray_from_shared_memory_int(self, data, type=int):
        original_structured_data = []

        if data is None:
            for i in range(self.n_total):
                original_structured_data.append(None)
        else:    
            data_shm, mask_shm, data_shape, mask_shape = data[0], data[1], data[2], data[3]

            data_array = np.ndarray(data_shape, dtype=type, buffer=data_shm.buf)
            mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

            data_array_index = 0
            for i in range(self.n_total):
                if mask_array[i]:
                    original_structured_data.append(data_array[data_array_index])
                    data_array_index += 1
                else:
                    original_structured_data.append(None)

        return original_structured_data
    
    def generate_shared_memory_twodim_single(self, data):
        # Flatten and prepare the data
        flattened_data = []
        mask = []

        for i, sublist in enumerate(data):
            if sublist is not None:
                flattened_data.append(tuple(sublist))
                mask.append(1)           
            else:
                mask.append(0)

        total_size = len(flattened_data) * np.dtype([('a', int), ('b', int)]).itemsize

        if total_size > 0:
            # Create shared memory for data
            shm_data = shm.SharedMemory(create=True, size=total_size)
            data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int)],
                                    buf=shm_data.buf)

            # Assign values to the shared memory data array
            for i, value in enumerate(flattened_data):
                data_array[i] = value

            # Create shared memory for mask
            shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(int).itemsize)
            mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

            # Assign values to the shared memory mask array
            for i, value in enumerate(mask):
                mask_array[i] = value

            # Get the names of the shared memory blocks
            return [shm_data, shm_mask, data_array.shape, mask_array.shape]
        
        return None
    
    def generate_ndarray_from_shared_memory_twodim_single(self, data):
        original_structured_data = []

        if data is None:
            for i in range(self.n_total):
                original_structured_data.append(None)
        else: 
            data_shm, mask_shm, data_shape, mask_shape = data[0], data[1], data[2], data[3]

            data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int)])
            data_array.data = data_shm.buf
            mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

            data_array_index = 0
            for i in range(self.n_total):
                if mask_array[i]:
                    original_structured_data.append(data_array[data_array_index])
                    data_array_index += 1
                else:
                    original_structured_data.append(None)

        return original_structured_data
    
    def generate_shared_memory_twodim_varying(self, data):
        # Flatten and prepare the data
        flattened_data = []
        mask = []
        indices = []

        for i, sublist in enumerate(data):
            if sublist is not None:
                mask.append(1)
                for j, item in enumerate(sublist):
                    flattened_data.append(tuple(item))
                    indices.append((i, j))
            else:
                mask.append(0)

        total_size = len(flattened_data) * np.dtype([('a', int), ('b', int)]).itemsize
        indices_total_size = len(indices) * np.dtype([('a', int), ('b', int)]).itemsize
        
        if total_size > 0:
            # Create shared memory for data
            shm_data = shm.SharedMemory(create=True, size=total_size)
            data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int)],
                                    buf=shm_data.buf)

            # Assign values to the shared memory data array
            for i, value in enumerate(flattened_data):
                data_array[i] = value

            # Created shared memory for indices
            shm_indices = shm.SharedMemory(create=True, size=indices_total_size)
            indices_array = np.recarray(len(indices), dtype=[('a', int), ('b', int)], buf=shm_indices.buf)

            # Assign values to the shared memory mask array
            for i, value in enumerate(indices):
                indices_array[i] = value

            # Get the names of the shared memory blocks
            return [shm_data, shm_indices, data_array.shape, indices_array.shape]
        
        return None
    
    def generate_ndarray_from_shared_memory_twodim_varying(self, data):
        original_structured_data = []

        if data is None:
            for i in range(self.n_total):
                original_structured_data.append(None)
        else:       
            data_shm, indices_shm, data_shape, indices_shape = data[0], data[1], data[2], data[3]

            data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int)])
            data_array.data = data_shm.buf
            indices_array = np.recarray(indices_shape, dtype=[('a', int), ('b', int)])
            indices_array.data = indices_shm.buf

            original_structured_data = self.generate_original_structure(data_array, indices_array)

        return original_structured_data
    # def generate_shared_memory_str(self, data):
    #     # Create a separate Boolean array to track valid/invalid elements
    #     valid_mask = [x is not None for x in data]
        
    #     size_sum = sum([len(x) for i, x in enumerate(data) if valid_mask[i]])
    #     # Create a shared memory block for the data array
    #     data_shm = shm.SharedMemory(create=True, size=size_sum * np.dtype(object).itemsize)

    #     # Store the data in the shared memory
    #     data_array = np.ndarray(len(data), dtype=int, buffer=data_shm.buf)
    #     data_array[valid_mask] = [x for x in data if x is not None]

    #     # Create a shared memory block for the valid mask
    #     mask_shm = shm.SharedMemory(create=True, size=len(valid_mask) * np.dtype(bool).itemsize)

    #     # Store the valid mask in the shared memory
    #     mask_array = np.ndarray(len(valid_mask), dtype=bool, buffer=mask_shm.buf)
    #     mask_array[:] = valid_mask

    #     return [data_shm, mask_shm, data_array.shape, mask_array.shape]
    
    # def generate_shared_memory_twodim_varying(self, data):
    #     if data is None:
    #         return None

    #     # Flatten and prepare the data
    #     flattened_data = []
    #     mask = []
    #     indices = []

    #     for i, sublist in enumerate(data):
    #         if sublist is not None:
    #             mask.append(1)
    #             for j, item in enumerate(sublist):
    #                 flattened_data.append(tuple(item))
    #                 indices.append((i, j))
    #         else:
    #             mask.append(0)

    #     total_size = len(flattened_data) * np.dtype([('a', int), ('b', int)]).itemsize

    #     if total_size > 0:
    #         # Create shared memory for data
    #         shm_data = shm.SharedMemory(create=True, size=total_size)
    #         data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int)],
    #                                 buf=shm_data.buf)

    #         # Assign values to the shared memory data array
    #         for i, value in enumerate(flattened_data):
    #             data_array[i] = value

    #         # Create shared memory for mask
    #         shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(int).itemsize)
    #         mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

    #         # Assign values to the shared memory mask array
    #         for i, value in enumerate(mask):
    #             mask_array[i] = value

    #         # Get the names of the shared memory blocks
    #         return [shm_data, shm_mask, data_array.shape, mask_array.shape, indices]
        
    #     return None
    
    # def generate_ndarray_from_shared_memory_twodim_varying(self, data):
    #     if data is None:
    #         return None
        
    #     data_shm, mask_shm, data_shape, mask_shape, indices = data[0], data[1], data[2], data[3], data[4]

    #     data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int)])
    #     data_array.data = data_shm.buf
    #     mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

    #     original_structured_data = []
    #     for i in range(len(data_array)):
    #         start_index, end_index = indices[i]
    #         sliced_data = data_array[start_index:end_index]
    #         sliced_mask = mask_array[start_index:end_index]
            
    #         reconstructed_array = []
    #         for value, is_valid in zip(sliced_data, sliced_mask):
    #             if is_valid:
    #                 reconstructed_array.append(value)
    #             else:
    #                 reconstructed_array.append(None)
            
    #         original_structured_data.append(reconstructed_array)

    #     return original_structured_data
    
    def generate_shared_memory_threedim_varying(self, data):
        # Flatten and prepare the data
        flattened_data = []
        mask = []
        indices = []

        for i, sublist in enumerate(data):
            if sublist is not None:
                mask.append(1)
                for j, item in enumerate(sublist):
                    flattened_data.append(tuple(item))
                    indices.append((i, j))
            else:
                mask.append(0)

        total_size = len(flattened_data) * np.dtype([('a', int), ('b', int), ('c', int)]).itemsize
        indices_total_size = len(indices) * np.dtype([('a', int), ('b', int)]).itemsize
        
        if total_size > 0:
            # Create shared memory for data
            shm_data = shm.SharedMemory(create=True, size=total_size)
            data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int), ('c', int)],
                                    buf=shm_data.buf)

            # Assign values to the shared memory data array
            for i, value in enumerate(flattened_data):
                data_array[i] = value

            # Create shared memory for mask
            # shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(int).itemsize)
            # mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

            # # Assign values to the shared memory mask array
            # for i, value in enumerate(mask):
            #     mask_array[i] = value

            # Created shared memory for indices
            shm_indices = shm.SharedMemory(create=True, size=indices_total_size)
            indices_array = np.recarray(len(indices), dtype=[('a', int), ('b', int)], buf=shm_indices.buf)

            # Assign values to the shared memory mask array
            for i, value in enumerate(indices):
                indices_array[i] = value

            # Get the names of the shared memory blocks
            return [shm_data, shm_indices, data_array.shape, indices_array.shape]
        
        return None
    
    def generate_ndarray_from_shared_memory_threedim_varying(self, data):
        original_structured_data = []

        if data is None:
            for i in range(self.n_total):
                original_structured_data.append(None)
        else:       
            data_shm, indices_shm, data_shape, indices_shape = data[0], data[1], data[2], data[3]

            data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int), ('c', int)])
            data_array.data = data_shm.buf
            indices_array = np.recarray(indices_shape, dtype=[('a', int), ('b', int)])
            indices_array.data = indices_shm.buf

            original_structured_data = self.generate_original_structure(data_array, indices_array)

        return original_structured_data
    
    def generate_shared_memory_threedim_single(self, data):
        # Flatten and prepare the data
        flattened_data = []
        mask = []

        for i, sublist in enumerate(data):
            if sublist is not None:
                flattened_data.append(tuple(sublist))
                mask.append(1)           
            else:
                mask.append(0)

        total_size = len(flattened_data) * np.dtype([('a', int), ('b', int), ('c', int)]).itemsize

        if total_size > 0:
            # Create shared memory for data
            shm_data = shm.SharedMemory(create=True, size=total_size)
            data_array = np.recarray(len(flattened_data), dtype=[('a', int), ('b', int), ('c', int)],
                                    buf=shm_data.buf)

            # Assign values to the shared memory data array
            for i, value in enumerate(flattened_data):
                data_array[i] = value

            # Create shared memory for mask
            shm_mask = shm.SharedMemory(create=True, size=len(mask) * np.dtype(int).itemsize)
            mask_array = np.frombuffer(shm_mask.buf, dtype=bool)

            # Assign values to the shared memory mask array
            for i, value in enumerate(mask):
                mask_array[i] = value

            # Get the names of the shared memory blocks
            return [shm_data, shm_mask, data_array.shape, mask_array.shape]
        
        return None
    
    def generate_ndarray_from_shared_memory_threedim_single(self, data):
        original_structured_data = []

        if data is None:
            for i in range(self.n_total):
                original_structured_data.append(None)
        else: 
            data_shm, mask_shm, data_shape, mask_shape = data[0], data[1], data[2], data[3]

            data_array = np.recarray(shape=data_shape, dtype=[('a', int), ('b', int), ('c', int)])
            data_array.data = data_shm.buf
            mask_array = np.ndarray(mask_shape, dtype=bool, buffer=mask_shm.buf)

            data_array_index = 0
            for i in range(self.n_total):
                if mask_array[i]:
                    original_structured_data.append(data_array[data_array_index])
                    data_array_index += 1
                else:
                    original_structured_data.append(None)

        return original_structured_data
    
    def generate_original_structure(self, data_array, indices_array):
        original_structure = []
        
        # data_array_index = 0

        indices_rows = []

        inner_arr_index = 0

        distinct_keys = util.get_distinct_first_indices(indices_array)

        for i in range(self.n_total):
            inner_arr = []

            if len(indices_rows) == 0 and i in distinct_keys:
                indices_rows = util.get_all_rows_by_key(indices_array, i)

            if len(indices_rows) > 0:
                start_index = inner_arr_index + indices_rows[0][1]
                end_index = start_index + indices_rows[-1][1] + 1

                inner_arr.extend(data_array[start_index:end_index]) # non-inclusive

                inner_arr_index = end_index 

                original_structure.append(inner_arr)

                indices_rows = []
            else:
                original_structure.append(None)
        
        return original_structure
    
    # def convert_from_shared_memory(self, shape, shared_memory_names):
        shm_age = shm.SharedMemory(name=shared_memory_names[0][0])
        self.age = np.ndarray(shape, dtype=shared_memory_names[0][1], buffer=shm_age.buf)

        shm_gender = shm.SharedMemory(name=shared_memory_names[1][0])
        self.gender = np.ndarray(shape, dtype=shared_memory_names[1][1], buffer=shm_gender.buf)

        shm_hhid = shm.SharedMemory(name=shared_memory_names[2][0])
        self.hhid = np.ndarray(shape, dtype=shared_memory_names[2][1], buffer=shm_hhid.buf)

        shm_scid = shm.SharedMemory(name=shared_memory_names[3][0])
        self.scid = np.ndarray(shape, dtype=shared_memory_names[3][1], buffer=shm_scid.buf)

        shm_sc_student = shm.SharedMemory(name=shared_memory_names[4][0])
        self.sc_student = np.ndarray(shape, dtype=shared_memory_names[4][1], buffer=shm_sc_student.buf)

        shm_sc_type = shm.SharedMemory(name=shared_memory_names[5][0])
        self.sc_type = np.ndarray(shape, dtype=shared_memory_names[5][1], buffer=shm_sc_type.buf)

        shm_wpid = shm.SharedMemory(name=shared_memory_names[6][0])
        self.wpid = np.ndarray(shape, dtype=shared_memory_names[6][1], buffer=shm_wpid.buf)

        shm_empstatus = shm.SharedMemory(name=shared_memory_names[7][0])
        self.empstatus = np.ndarray(shape, dtype=shared_memory_names[7][1], buffer=shm_empstatus.buf)

        shm_empind = shm.SharedMemory(name=shared_memory_names[8][0])
        self.empind = np.ndarray(shape, dtype=shared_memory_names[8][1], buffer=shm_empind.buf)

        shm_empftpt = shm.SharedMemory(name=shared_memory_names[9][0])
        self.empftpt = np.ndarray(shape, dtype=shared_memory_names[9][1], buffer=shm_empftpt.buf)

        shm_res_cellid = shm.SharedMemory(name=shared_memory_names[10][0])
        self.res_cellid = np.ndarray(shape, dtype=shared_memory_names[10][1], buffer=shm_res_cellid.buf)

        shm_work_cellid = shm.SharedMemory(name=shared_memory_names[11][0])
        self.work_cellid = np.ndarray(shape, dtype=shared_memory_names[11][1], buffer=shm_work_cellid.buf)

        shm_school_cellid = shm.SharedMemory(name=shared_memory_names[12][0])
        self.school_cellid = np.ndarray(shape, dtype=shared_memory_names[12][1], buffer=shm_school_cellid.buf)

        shm_inst_cellid = shm.SharedMemory(name=shared_memory_names[13][0])
        self.inst_cellid = np.ndarray(shape, dtype=shared_memory_names[13][1], buffer=shm_inst_cellid.buf)

        shm_age_bracket_index = shm.SharedMemory(name=shared_memory_names[14][0])
        self.age_bracket_index = np.ndarray(shape, dtype=shared_memory_names[14][1], buffer=shm_age_bracket_index.buf)

        shm_epi_age_bracket_index = shm.SharedMemory(name=shared_memory_names[15][0])
        self.epi_age_bracket_index = np.ndarray(shape, dtype=shared_memory_names[15][1], buffer=shm_epi_age_bracket_index.buf)

        shm_working_age_bracket_index = shm.SharedMemory(name=shared_memory_names[0][0])
        self.working_age_bracket_index = np.ndarray(shape, dtype=shared_memory_names[0][1], buffer=shm_working_age_bracket_index.buf)

        shm_soc_rate = shm.SharedMemory(name=shared_memory_names[1][0])
        self.soc_rate = np.ndarray(shape, dtype=shared_memory_names[1][1], buffer=shm_soc_rate.buf)

        shm_guardian_id = shm.SharedMemory(name=shared_memory_names[2][0])
        self.guardian_id = np.ndarray(shape, dtype=shared_memory_names[2][1], buffer=shm_guardian_id.buf)

        shm_working_schedule = shm.SharedMemory(name=shared_memory_names[3][0])
        self.working_schedule = np.ndarray(shape, dtype=shared_memory_names[3][1], buffer=shm_working_schedule.buf)

        shm_isshiftbased = shm.SharedMemory(name=shared_memory_names[4][0])
        self.isshiftbased = np.ndarray(shape, dtype=shared_memory_names[4][1], buffer=shm_isshiftbased.buf)

        shm_pub_transp_reg = shm.SharedMemory(name=shared_memory_names[5][0])
        self.pub_transp_reg = np.ndarray(shape, dtype=shared_memory_names[5][1], buffer=shm_pub_transp_reg.buf)

        shm_ent_activity = shm.SharedMemory(name=shared_memory_names[6][0])
        self.ent_activity = np.ndarray(shape, dtype=shared_memory_names[6][1], buffer=shm_ent_activity.buf)

        shm_itinerary = shm.SharedMemory(name=shared_memory_names[7][0])
        self.itinerary = np.ndarray(shape, dtype=shared_memory_names[7][1], buffer=shm_itinerary.buf)

        shm_itinerary_nextday = shm.SharedMemory(name=shared_memory_names[8][0])
        self.itinerary_nextday = np.ndarray(shape, dtype=shared_memory_names[8][1], buffer=shm_itinerary_nextday.buf)

        shm_non_daily_activity_recurring = shm.SharedMemory(name=shared_memory_names[9][0])
        self.non_daily_activity_recurring = np.ndarray(shape, dtype=shared_memory_names[9][1], buffer=shm_non_daily_activity_recurring.buf)

        shm_prevday_non_daily_activity_recurring = shm.SharedMemory(name=shared_memory_names[10][0])
        self.prevday_non_daily_activity_recurring = np.ndarray(shape, dtype=shared_memory_names[10][1], buffer=shm_prevday_non_daily_activity_recurring.buf)

        shm_busdriver = shm.SharedMemory(name=shared_memory_names[11][0])
        self.busdriver = np.ndarray(shape, dtype=shared_memory_names[11][1], buffer=shm_busdriver.buf)

        shm_state_transition_by_day = shm.SharedMemory(name=shared_memory_names[12][0])
        self.state_transition_by_day = np.ndarray(shape, dtype=shared_memory_names[12][1], buffer=shm_state_transition_by_day.buf)

        shm_test_day = shm.SharedMemory(name=shared_memory_names[13][0])
        self.test_day = np.ndarray(shape, dtype=shared_memory_names[13][1], buffer=shm_test_day.buf)

        shm_test_result_day = shm.SharedMemory(name=shared_memory_names[14][0])
        self.test_result_day = np.ndarray(shape, dtype=shared_memory_names[14][1], buffer=shm_test_result_day.buf)

        shm_quarantine_days = shm.SharedMemory(name=shared_memory_names[15][0])
        self.quarantine_days = np.ndarray(shape, dtype=shared_memory_names[15][1], buffer=shm_quarantine_days.buf)

        shm_hospitalisation_days = shm.SharedMemory(name=shared_memory_names[15][0])
        self.hospitalisation_days = np.ndarray(shape, dtype=shared_memory_names[15][1], buffer=shm_hospitalisation_days.buf)

        shm_vaccination_days = shm.SharedMemory(name=shared_memory_names[15][0])
        self.vaccination_days = np.ndarray(shape, dtype=shared_memory_names[15][1], buffer=shm_vaccination_days.buf)

    def calculate_memory_size(self, attr_name=None):
        total_size = sum(sys.getsizeof(getattr(self, attr)) for attr in dir(self) if attr_name is None or attr==attr_name)
        return total_size
    

