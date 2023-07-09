import numpy as np
import sys
import struct
import time
from simulator import util
from simulator.epidemiology import SEIRState, SEIRStateTransition, InfectionType, Severity

class Agents:
    def __init__(self) -> None:
        self.n_total = None
        self.n_locals = None
        self.n_tourists = None
        self.age = [] # int
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

        self.seir_state = [] # states -> 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
        self.seir_state_transition_for_day = []
        self.infection_type = [] # was dict {agentid: infectiontype}
        self.infection_severity = [] # was dict {agentid: infectionseverity}
        self.vaccination_doses = [] # number of doses per agent

        self.shm_age = [] # int
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

        self.shm_seir_state = [] # states -> 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
        self.shm_seir_state_transition_for_day = []
        self.shm_infection_type = [] # was dict {agentid: infectiontype}
        self.shm_infection_severity = [] # was dict {agentid: infectionseverity}
        self.shm_vaccination_doses = [] # number of doses per agent

    def populate(self, data, n_locals, n_tourists, agents_seir_state):
        start = time.time()

        n_total = n_locals + n_tourists

        self.n_total = n_total
        self.n_locals = n_locals
        self.n_tourists = n_tourists

        for _, properties in data.items():
            self.age.append(properties["age"] if "age" in properties else None)
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

            self.infection_type.append(InfectionType.Undefined)
            self.infection_severity.append(InfectionType.Undefined)
            self.seir_state_transition_for_day.append(None)

        self.seir_state = agents_seir_state
        self.vaccination_doses = np.array([0 if i < n_locals else None for i in range(n_total)]) # not applicable to tourists

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
            self.age, self.shm_age = agents_mp_to_clone.age, agents_mp_to_clone.shm_age
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

            self.itinerary, self.shm_itinerary = agents_mp_to_clone.itinerary, agents_mp_to_clone.shm_itinerary
            self.itinerary_nextday, self.shm_itinerary_nextday = agents_mp_to_clone.itinerary_nextday, agents_mp_to_clone.shm_itinerary_nextday
            self.non_daily_activity_recurring, self.shm_non_daily_activity_recurring = agents_mp_to_clone.non_daily_activity_recurring, agents_mp_to_clone.shm_non_daily_activity_recurring
            self.prevday_non_daily_activity_recurring, self.shm_prevday_non_daily_activity_recurring = agents_mp_to_clone.prevday_non_daily_activity_recurring, agents_mp_to_clone.shm_prevday_non_daily_activity_recurring
        
            self.seir_state, self.shm_seir_state = agents_mp_to_clone.seir_state, agents_mp_to_clone.shm_seir_state
            self.seir_state_transition_for_day, self.shm_seir_state_transition_for_day = agents_mp_to_clone.seir_state_transition_for_day, agents_mp_to_clone.shm_seir_state_transition_for_day
            self.infection_type, self.shm_infection_type = agents_mp_to_clone.infection_type, agents_mp_to_clone.shm_infection_type
            self.infection_severity, self.shm_infection_severity = agents_mp_to_clone.infection_severity, agents_mp_to_clone.shm_infection_severity
            self.vaccination_doses, self.shm_vaccination_doses = agents_mp_to_clone.vaccination_doses, agents_mp_to_clone.shm_vaccination_doses
        elif contactnetwork:
            self.age_bracket_index, self.shm_age_bracket_index = agents_mp_to_clone.age_bracket_index, agents_mp_to_clone.shm_age_bracket_index
            self.soc_rate, self.shm_soc_rate = agents_mp_to_clone.soc_rate, agents_mp_to_clone.shm_soc_rate
            self.epi_age_bracket_index, self.shm_epi_age_bracket_index = agents_mp_to_clone.epi_age_bracket_index, agents_mp_to_clone.shm_epi_age_bracket_index
            self.res_cellid, self.shm_res_cellid = agents_mp_to_clone.res_cellid, agents_mp_to_clone.shm_res_cellid

            self.seir_state, self.shm_seir_state = agents_mp_to_clone.seir_state, agents_mp_to_clone.shm_seir_state
            self.seir_state_transition_for_day, self.shm_seir_state_transition_for_day = agents_mp_to_clone.seir_state_transition_for_day, agents_mp_to_clone.shm_seir_state_transition_for_day
            self.infection_type, self.shm_infection_type = agents_mp_to_clone.infection_type, agents_mp_to_clone.shm_infection_type
            self.infection_severity, self.shm_infection_severity = agents_mp_to_clone.infection_severity, agents_mp_to_clone.shm_infection_severity
            self.vaccination_doses, self.shm_vaccination_doses = agents_mp_to_clone.vaccination_doses, agents_mp_to_clone.shm_vaccination_doses
        elif itinerary:
            self.age, self.shm_age = agents_mp_to_clone.age, agents_mp_to_clone.shm_age
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

            self.itinerary, self.shm_itinerary = agents_mp_to_clone.itinerary, agents_mp_to_clone.shm_itinerary
            self.itinerary_nextday, self.shm_itinerary_nextday = agents_mp_to_clone.itinerary_nextday, agents_mp_to_clone.shm_itinerary_nextday
            self.non_daily_activity_recurring, self.shm_non_daily_activity_recurring = agents_mp_to_clone.non_daily_activity_recurring, agents_mp_to_clone.shm_non_daily_activity_recurring
            self.prevday_non_daily_activity_recurring, self.shm_prevday_non_daily_activity_recurring = agents_mp_to_clone.prevday_non_daily_activity_recurring, agents_mp_to_clone.shm_prevday_non_daily_activity_recurring

            self.seir_state, self.shm_seir_state = agents_mp_to_clone.seir_state, agents_mp_to_clone.shm_seir_state
            self.seir_state_transition_for_day, self.shm_seir_state_transition_for_day = agents_mp_to_clone.seir_state_transition_for_day, agents_mp_to_clone.shm_seir_state_transition_for_day
            self.infection_type, self.shm_infection_type = agents_mp_to_clone.infection_type, agents_mp_to_clone.shm_infection_type
            self.infection_severity, self.shm_infection_severity = agents_mp_to_clone.infection_severity, agents_mp_to_clone.shm_infection_severity
            self.vaccination_doses, self.shm_vaccination_doses = agents_mp_to_clone.vaccination_doses, agents_mp_to_clone.shm_vaccination_doses

    def clone_shm(self, agents_mp_to_clone, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True

        self.n_total = agents_mp_to_clone.n_total
        self.n_locals = agents_mp_to_clone.n_locals
        self.n_tourists = agents_mp_to_clone.n_tourists

        if loadall:
            self.shm_age = agents_mp_to_clone.shm_age
            self.shm_sc_student = agents_mp_to_clone.shm_sc_student
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.shm_sc_type))
            # self.wpid = self.generate_ndarray_from_shared_memory_int(self.shm_wpid)
            self.shm_empstatus = agents_mp_to_clone.shm_empstatus
            self.shm_empind = agents_mp_to_clone.shm_empind
            self.shm_empftpt = agents_mp_to_clone.shm_empftpt
            self.shm_res_cellid = agents_mp_to_clone.shm_res_cellid
            self.shm_work_cellid = agents_mp_to_clone.shm_work_cellid
            self.shm_school_cellid = agents_mp_to_clone.shm_school_cellid
            self.shm_inst_cellid = agents_mp_to_clone.shm_inst_cellid
            self.shm_age_bracket_index = agents_mp_to_clone.shm_age_bracket_index
            self.shm_epi_age_bracket_index = agents_mp_to_clone.shm_epi_age_bracket_index
            self.shm_working_age_bracket_index = agents_mp_to_clone.shm_working_age_bracket_index
            self.shm_soc_rate = agents_mp_to_clone.shm_soc_rate
            self.shm_guardian_id = agents_mp_to_clone.shm_guardian_id
            self.shm_pub_transp_reg = agents_mp_to_clone.shm_pub_transp_reg
            self.shm_ent_activity = agents_mp_to_clone.shm_ent_activity
            self.shm_isshiftbased = agents_mp_to_clone.shm_isshiftbased
            self.shm_busdriver = agents_mp_to_clone.shm_busdriver
            self.shm_working_schedule = agents_mp_to_clone.shm_working_schedule

            self.shm_itinerary = agents_mp_to_clone.shm_itinerary
            self.shm_itinerary_nextday = agents_mp_to_clone.shm_itinerary_nextday
            self.shm_non_daily_activity_recurring = agents_mp_to_clone.shm_non_daily_activity_recurring
            self.shm_prevday_non_daily_activity_recurring = agents_mp_to_clone.shm_prevday_non_daily_activity_recurring
        
            self.shm_seir_state = agents_mp_to_clone.shm_seir_state
            self.shm_seir_state_transition_for_day = agents_mp_to_clone.shm_seir_state_transition_for_day
            self.shm_infection_type = agents_mp_to_clone.shm_infection_type
            self.shm_infection_severity = agents_mp_to_clone.shm_infection_severity
            self.shm_vaccination_doses = agents_mp_to_clone.shm_vaccination_doses
        elif contactnetwork:
            self.shm_age_bracket_index = agents_mp_to_clone.shm_age_bracket_index
            self.shm_soc_rate = agents_mp_to_clone.shm_soc_rate
            self.shm_epi_age_bracket_index = agents_mp_to_clone.shm_epi_age_bracket_index
            self.shm_res_cellid = agents_mp_to_clone.shm_res_cellid

            self.shm_seir_state = agents_mp_to_clone.shm_seir_state
            self.shm_seir_state_transition_for_day = agents_mp_to_clone.shm_seir_state_transition_for_day
            self.shm_infection_type = agents_mp_to_clone.shm_infection_type
            self.shm_infection_severity = agents_mp_to_clone.shm_infection_severity
            self.shm_vaccination_doses = agents_mp_to_clone.shm_vaccination_doses
        elif itinerary:
            self.shm_age = agents_mp_to_clone.shm_age
            self.shm_sc_student = agents_mp_to_clone.shm_sc_student
            self.shm_empstatus = agents_mp_to_clone.shm_empstatus
            self.shm_empind = agents_mp_to_clone.shm_empind
            self.shm_ent_activity = agents_mp_to_clone.shm_ent_activity
            self.shm_isshiftbased = agents_mp_to_clone.shm_isshiftbased
            self.shm_empftpt = agents_mp_to_clone.shm_empftpt
            self.shm_guardian_id = agents_mp_to_clone.shm_guardian_id
            self.shm_age_bracket_index = agents_mp_to_clone.shm_age_bracket_index
            self.shm_epi_age_bracket_index = agents_mp_to_clone.shm_epi_age_bracket_index
            self.shm_working_age_bracket_index = agents_mp_to_clone.shm_working_age_bracket_index
            self.shm_res_cellid = agents_mp_to_clone.shm_res_cellid
            self.shm_work_cellid = agents_mp_to_clone.shm_work_cellid
            self.shm_school_cellid = agents_mp_to_clone.shm_school_cellid
            self.shm_pub_transp_reg = agents_mp_to_clone.shm_pub_transp_reg
            self.shm_working_schedule = agents_mp_to_clone.shm_working_schedule

            self.shm_itinerary = agents_mp_to_clone.shm_itinerary
            self.shm_itinerary_nextday = agents_mp_to_clone.shm_itinerary_nextday
            self.shm_non_daily_activity_recurring = agents_mp_to_clone.shm_non_daily_activity_recurring
            self.shm_prevday_non_daily_activity_recurring = agents_mp_to_clone.shm_prevday_non_daily_activity_recurring

            self.shm_seir_state = agents_mp_to_clone.shm_seir_state
            self.shm_seir_state_transition_for_day = agents_mp_to_clone.shm_seir_state_transition_for_day
            self.shm_infection_type = agents_mp_to_clone.shm_infection_type
            self.shm_infection_severity = agents_mp_to_clone.shm_infection_severity
            self.shm_vaccination_doses = agents_mp_to_clone.shm_vaccination_doses

    # readonly memory does not have to be cleaned after closing shared memory for specific workers for specific day 
    # these can be retained throughout the whole simulation because they do not change
    # however, the option to clean them up in specific cases (i.e. itinerary, contactnetwork) is still provided
    def cleanup_shared_memory_readonly(self, closeall=False, itinerary=False, contactnetwork=False):
        if not closeall and not itinerary and not contactnetwork:
            closeall = True
    
        start = time.time()

        if closeall:
            util.close_shm(self.shm_age)
            util.close_shm(self.shm_sc_student)
            util.close_shm(self.shm_empstatus)
            util.close_shm(self.shm_empind)
            util.close_shm(self.shm_empftpt)
            util.close_shm(self.shm_work_cellid)
            util.close_shm(self.shm_school_cellid)
            util.close_shm(self.shm_inst_cellid)
            util.close_shm(self.shm_working_age_bracket_index)
            util.close_shm(self.shm_guardian_id)
            util.close_shm(self.shm_pub_transp_reg)
            util.close_shm(self.shm_ent_activity)
            util.close_shm(self.shm_isshiftbased)
            util.close_shm(self.shm_busdriver)
            util.close_shm(self.shm_inst_cellid)
            util.close_shm(self.shm_res_cellid)
            util.close_shm(self.shm_age_bracket_index)
            util.close_shm(self.shm_epi_age_bracket_index)
            util.close_shm(self.shm_soc_rate)
        elif contactnetwork:
            util.close_shm(self.shm_res_cellid)
            util.close_shm(self.shm_age_bracket_index)
            util.close_shm(self.shm_epi_age_bracket_index)
            util.close_shm(self.shm_soc_rate)
        elif itinerary:
            util.close_shm(self.shm_age)
            util.close_shm(self.shm_sc_student)
            util.close_shm(self.shm_empstatus)
            util.close_shm(self.shm_empind)
            util.close_shm(self.shm_empftpt)
            util.close_shm(self.shm_ent_activity)
            util.close_shm(self.shm_isshiftbased)
            util.close_shm(self.shm_guardian_id)
            util.close_shm(self.shm_age_bracket_index)
            util.close_shm(self.shm_epi_age_bracket_index)
            util.close_shm(self.shm_working_age_bracket_index)
            util.close_shm(self.shm_res_cellid)
            util.close_shm(self.shm_work_cellid)
            util.close_shm(self.shm_school_cellid)
            util.close_shm(self.shm_inst_cellid)
            util.close_shm(self.shm_pub_transp_reg)

        time_taken = time.time() - start
        print("agents_mp cleanup_shared_memory_readonly time taken: " + str(time_taken))

    # def convert_to_shared_memory_workingschedule(self):            
    #     self.shm_working_schedule = self.generate_shared_memory_threedim_varying(self.working_schedule)

    # def convert_to_shared_memory_isshiftbased(self):
    #     self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)

    def cleanup_shared_memory_dynamic(self, closeall=False, itinerary=False, contactnetwork=False):
        if not closeall and not itinerary and not contactnetwork:
            closeall = True

        if closeall or itinerary:
            util.close_shm(self.shm_itinerary)
            util.close_shm(self.shm_itinerary_nextday)
            util.close_shm(self.shm_non_daily_activity_recurring)
            util.close_shm(self.shm_prevday_non_daily_activity_recurring)
            util.close_shm(self.shm_state_transition_by_day)
            util.close_shm(self.shm_test_day)
            util.close_shm(self.shm_test_result_day)
            util.close_shm(self.shm_quarantine_days)
            util.close_shm(self.shm_hospitalisation_days)
            util.close_shm(self.shm_vaccination_days)
            util.close_shm(self.shm_seir_state)
            util.close_shm(self.shm_seir_state_transition_for_day)
            util.close_shm(self.shm_infection_type)
            util.close_shm(self.shm_infection_severity)
            util.close_shm(self.shm_vaccination_doses)
        elif contactnetwork:
            util.close_shm(self.shm_state_transition_by_day)
            util.close_shm(self.shm_test_day)
            util.close_shm(self.shm_test_result_day)
            util.close_shm(self.shm_quarantine_days)
            util.close_shm(self.shm_hospitalisation_days)
            util.close_shm(self.shm_vaccination_days)

    def clear_non_shared_memory(self):
        self.age = [] # int
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

        self.seir_state = [] # states -> 0: undefined, 1: susceptible, 2: exposed, 3: infectious, 4: recovered, 5: deceased
        self.seir_state_transition_for_day = []
        self.infection_type = [] # was dict {agentid: infectiontype}
        self.infection_severity = [] # was dict {agentid: infectionseverity}
        self.vaccination_doses = [] # number of doses per agent

    def clear_non_shared_memory_readonly(self):
        self.age = None
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

        self.seir_state = None
        self.seir_state_transition_for_day = None
        self.infection_type = None
        self.infection_severity = None
        self.vaccination_doses = None        

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
        # elif name == "gender": # from here onwards, these values are supposed to be readonly
        #     self.gender[index] = value
        # elif name == "hhid":
        #     self.hhid[index] = value
        # elif name == "scid":
        #     self.scid[index] = value
        elif name == "sc_student":
            self.sc_student[index] = value
        # elif name == "sc_type":
        #     self.sc_type[index] = value
        # elif name == "wpid":
        #     self.wpid[index] = value
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
        elif name == "seir_state":
            self.seir_state[index] = value
        elif name == "seir_state_transition_for_day":
            self.seir_state_transition_for_day[index] = value
        elif name == "infection_type":
            self.infection_type[index] = value
        elif name == "infection_severity":
            self.infection_severity[index] = value
        elif name == "vaccination_doses":
            self.vaccination_doses[index] = value

    def convert_to_shared_memory_readonly(self, loadall=False, itinerary=False, contactnetwork=False, clear_normal_memory=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            self.shm_age = util.generate_shared_memory_int(self.age)
            # self.shm_gender = self.generate_shared_memory_int(self.gender)
            # self.shm_hhid = self.generate_shared_memory_int(self.hhid)
            # self.shm_scid = self.generate_shared_memory_int(self.scid)
            self.shm_sc_student = util.generate_shared_memory_int(self.sc_student)
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.sc_type))
            # self.shm_wpid = self.generate_shared_memory_int(self.wpid)
            self.shm_empstatus = util.generate_shared_memory_int(self.empstatus)
            self.shm_empind = util.generate_shared_memory_int(self.empind)
            self.shm_empftpt = util.generate_shared_memory_int(self.empftpt)

            self.shm_work_cellid = util.generate_shared_memory_int(self.work_cellid)
            self.shm_school_cellid = util.generate_shared_memory_int(self.school_cellid)
            self.shm_inst_cellid = util.generate_shared_memory_int(self.inst_cellid)
            self.shm_working_age_bracket_index = util.generate_shared_memory_int(self.working_age_bracket_index)

            self.shm_guardian_id = util.generate_shared_memory_int(self.guardian_id)
            self.shm_pub_transp_reg = util.generate_shared_memory_int(self.pub_transp_reg)
            self.shm_ent_activity = util.generate_shared_memory_int(self.ent_activity)
            self.shm_isshiftbased = util.generate_shared_memory_int(self.isshiftbased)
            self.shm_busdriver = util.generate_shared_memory_int(self.busdriver)

            self.shm_res_cellid = util.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = util.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = util.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = util.generate_shared_memory_int(self.soc_rate, float)
        elif contactnetwork:
            self.shm_res_cellid = util.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = util.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = util.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = util.generate_shared_memory_int(self.soc_rate, float)
        elif itinerary:
            self.shm_age = util.generate_shared_memory_int(self.age)
            self.shm_sc_student = util.generate_shared_memory_int(self.sc_student)
            self.shm_empstatus = util.generate_shared_memory_int(self.empstatus)
            self.shm_empind = util.generate_shared_memory_int(self.empind)
            self.shm_empftpt = util.generate_shared_memory_int(self.empftpt)
            self.shm_ent_activity = util.generate_shared_memory_int(self.ent_activity)
            self.shm_isshiftbased = util.generate_shared_memory_int(self.isshiftbased)
            self.shm_guardian_id = util.generate_shared_memory_int(self.guardian_id)
            self.shm_age_bracket_index = util.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = util.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_working_age_bracket_index = util.generate_shared_memory_int(self.working_age_bracket_index)
            self.shm_res_cellid = util.generate_shared_memory_int(self.res_cellid)
            self.shm_work_cellid = util.generate_shared_memory_int(self.work_cellid)
            self.shm_school_cellid = util.generate_shared_memory_int(self.school_cellid)
            self.shm_inst_cellid = util.generate_shared_memory_int(self.inst_cellid)
            self.shm_pub_transp_reg = util.generate_shared_memory_int(self.pub_transp_reg)

        if clear_normal_memory:
            self.clear_non_shared_memory_readonly()

        time_taken = time.time() - start
        print("agents_mp convert_to_shared_memory_readonly time taken: " + str(time_taken))

    # def convert_to_shared_memory_workingschedule(self):            
    #     self.shm_working_schedule = self.generate_shared_memory_threedim_varying(self.working_schedule)

    # def convert_to_shared_memory_isshiftbased(self):
    #     self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)

    def convert_to_shared_memory_dynamic(self, loadall=False, itinerary=False, contactnetwork=False, clear_normal_memory=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True

        if loadall or itinerary:
            start = time.time()

            self.shm_working_schedule = util.generate_shared_memory_multidim_varying(self.working_schedule, 3)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (working_schedule) time taken: " + str(time_taken))

            start = time.time()

            self.shm_itinerary = util.generate_shared_memory_multidim_varying(self.itinerary, 3)
            self.shm_itinerary_nextday = util.generate_shared_memory_multidim_varying(self.itinerary_nextday, 3)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (itinerary) time taken: " + str(time_taken))

            start = time.time()

            self.shm_non_daily_activity_recurring = util.generate_shared_memory_multidim_single(self.non_daily_activity_recurring, 3)
            self.shm_prevday_non_daily_activity_recurring = util.generate_shared_memory_multidim_single(self.prevday_non_daily_activity_recurring, 3)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (non_daily_activity_recurring) time taken: " + str(time_taken))

            start = time.time()

            self.shm_state_transition_by_day = util.generate_shared_memory_multidim_varying(self.state_transition_by_day, 3)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.shm_test_day = util.generate_shared_memory_multidim_single(self.test_day, 3)
            self.shm_test_result_day = util.generate_shared_memory_multidim_single(self.test_result_day, 2)
            self.shm_quarantine_days = util.generate_shared_memory_multidim_varying(self.quarantine_days, 2)
            self.shm_hospitalisation_days = util.generate_shared_memory_multidim_single(self.hospitalisation_days, 3)
            self.shm_vaccination_days = util.generate_shared_memory_multidim_single(self.hospitalisation_days, 3)  

            self.shm_seir_state = util.generate_shared_memory_int(self.seir_state)
            self.shm_seir_state_transition_for_day = util.generate_shared_memory_multidim_single(self.seir_state_transition_for_day, 6)
            self.shm_infection_type = util.generate_shared_memory_int(self.infection_type)
            self.shm_infection_severity = util.generate_shared_memory_int(self.infection_severity)
            self.shm_vaccination_doses = util.generate_shared_memory_int(self.vaccination_doses)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (epi) time taken: " + str(time_taken))
        elif contactnetwork:
            start = time.time()

            self.shm_state_transition_by_day = util.generate_shared_memory_multidim_varying(self.state_transition_by_day, 3)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.shm_test_day = util.generate_shared_memory_multidim_single(self.test_day, 2)
            self.shm_test_result_day = util.generate_shared_memory_multidim_single(self.test_result_day, 2)
            self.shm_quarantine_days = util.generate_shared_memory_multidim_varying(self.quarantine_days, 2)
            self.shm_hospitalisation_days = util.generate_shared_memory_multidim_single(self.hospitalisation_days, 3)
            self.shm_vaccination_days = util.generate_shared_memory_multidim_single(self.vaccination_days, 3)  

            self.shm_seir_state = util.generate_shared_memory_int(self.seir_state)
            self.shm_seir_state_transition_for_day = util.generate_shared_memory_multidim_single(self.seir_state_transition_for_day, 6)
            self.shm_infection_type = util.generate_shared_memory_int(self.infection_type)
            self.shm_infection_severity = util.generate_shared_memory_int(self.infection_severity)
            self.shm_vaccination_doses = util.generate_shared_memory_int(self.vaccination_doses)

            time_taken = time.time() - start
            print("agents_mp convert_to_shared_memory_dynamic (epi) time taken: " + str(time_taken))

        if clear_normal_memory:
            self.clear_non_shared_memory_dynamic()

    def convert_from_shared_memory_readonly(self, loadall=False, itinerary=False, contactnetwork=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            self.age = util.generate_ndarray_from_shared_memory_int(self.shm_age, self.n_total)
            # self.gender = self.generate_ndarray_from_shared_memory_int(self.shm_gender)
            # self.hhid = self.generate_ndarray_from_shared_memory_int(self.shm_hhid)
            # self.scid = self.generate_ndarray_from_shared_memory_int(self.shm_scid)
            self.sc_student = util.generate_ndarray_from_shared_memory_int(self.shm_sc_student, self.n_total)
            # self.shared_memory_names.append(self.generate_shared_memory_str(self.shm_sc_type))
            # self.wpid = self.generate_ndarray_from_shared_memory_int(self.shm_wpid)
            self.empstatus = util.generate_ndarray_from_shared_memory_int(self.shm_empstatus, self.n_total)
            self.empind = util.generate_ndarray_from_shared_memory_int(self.shm_empind, self.n_total)
            self.empftpt = util.generate_ndarray_from_shared_memory_int(self.shm_empftpt, self.n_total)
            self.res_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_res_cellid, self.n_total)
            self.work_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_work_cellid, self.n_total)
            self.school_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_school_cellid, self.n_total)
            self.inst_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_inst_cellid, self.n_total)
            self.age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index, self.n_total)
            self.epi_age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index, self.n_total)
            self.working_age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_working_age_bracket_index, self.n_total)
            self.soc_rate = util.generate_ndarray_from_shared_memory_int(self.shm_soc_rate, self.n_total, float)
            self.guardian_id = util.generate_ndarray_from_shared_memory_int(self.shm_guardian_id, self.n_total)
            self.pub_transp_reg = util.generate_ndarray_from_shared_memory_int(self.shm_pub_transp_reg, self.n_total)
            self.ent_activity = util.generate_ndarray_from_shared_memory_int(self.shm_ent_activity, self.n_total)
            self.isshiftbased = util.generate_ndarray_from_shared_memory_int(self.shm_isshiftbased, self.n_total)  
            self.busdriver = util.generate_ndarray_from_shared_memory_int(self.shm_busdriver, self.n_total)
        elif contactnetwork:
            self.age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index, self.n_total)
            self.soc_rate = util.generate_ndarray_from_shared_memory_int(self.shm_soc_rate, self.n_total, float)
            self.epi_age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index, self.n_total)
            self.res_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_res_cellid, self.n_total)
        elif itinerary:
            self.age = util.generate_ndarray_from_shared_memory_int(self.shm_age, self.n_total)
            self.sc_student = util.generate_ndarray_from_shared_memory_int(self.shm_sc_student, self.n_total)
            self.empstatus = util.generate_ndarray_from_shared_memory_int(self.shm_empstatus, self.n_total)
            self.empind = util.generate_ndarray_from_shared_memory_int(self.shm_empind, self.n_total)
            self.ent_activity = util.generate_ndarray_from_shared_memory_int(self.shm_ent_activity, self.n_total)
            self.isshiftbased = util.generate_ndarray_from_shared_memory_int(self.shm_isshiftbased, self.n_total)
            self.empftpt = util.generate_ndarray_from_shared_memory_int(self.shm_empftpt, self.n_total)
            self.guardian_id = util.generate_ndarray_from_shared_memory_int(self.shm_guardian_id, self.n_total)
            self.age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_age_bracket_index, self.n_total)
            self.working_age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_working_age_bracket_index, self.n_total)
            self.res_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_res_cellid, self.n_total)
            self.epi_age_bracket_index = util.generate_ndarray_from_shared_memory_int(self.shm_epi_age_bracket_index, self.n_total)
            self.work_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_work_cellid, self.n_total)
            self.school_cellid = util.generate_ndarray_from_shared_memory_int(self.shm_school_cellid, self.n_total)
            self.pub_transp_reg = util.generate_ndarray_from_shared_memory_int(self.shm_pub_transp_reg, self.n_total)

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

            self.working_schedule = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_working_schedule, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (working_schedule) time taken: " + str(time_taken))

            start = time.time()

            self.itinerary = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_itinerary, self.n_total)
            self.itinerary_nextday = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_itinerary_nextday, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (itinerary) time taken: " + str(time_taken))

            start = time.time()

            self.non_daily_activity_recurring = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_non_daily_activity_recurring, self.n_total)
            self.prevday_non_daily_activity_recurring = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_prevday_non_daily_activity_recurring, self.n_total)
            
            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (non_daily_activity_recurring) time taken: " + str(time_taken))

            start = time.time()

            self.state_transition_by_day = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_state_transition_by_day, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()

            self.test_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_test_day, self.n_total)
            self.test_result_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_test_result_day, self.n_total)
            self.quarantine_days = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_quarantine_days, self.n_total)
            self.hospitalisation_days = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_hospitalisation_days, self.n_total)
            self.vaccination_days = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_vaccination_days, self.n_total)

            self.seir_state = util.generate_ndarray_from_shared_memory_int(self.shm_seir_state, self.n_total)
            self.seir_state_transition_for_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_seir_state_transition_for_day, self.n_total)
            self.infection_type = util.generate_ndarray_from_shared_memory_int(self.shm_infection_type, self.n_total)
            self.infection_severity = util.generate_ndarray_from_shared_memory_int(self.shm_infection_severity, self.n_total)
            self.vaccination_doses = util.generate_ndarray_from_shared_memory_int(self.shm_vaccination_doses, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (epi) time taken: " + str(time_taken))
        else:
            start = time.time()

            self.state_transition_by_day = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_state_transition_by_day, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (state_transition_by_day) time taken: " + str(time_taken))

            start = time.time()
            self.test_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_test_day, self.n_total)
            self.test_result_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_test_result_day, self.n_total)
            self.quarantine_days = util.generate_ndarray_from_shared_memory_multidim_varying(self.shm_quarantine_days, self.n_total)
            self.hospitalisation_days = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_hospitalisation_days, self.n_total)
            self.vaccination_days = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_vaccination_days, self.n_total)  

            self.seir_state = util.generate_ndarray_from_shared_memory_int(self.shm_seir_state, self.n_total)
            self.seir_state_transition_for_day = util.generate_ndarray_from_shared_memory_multidim_single(self.shm_seir_state_transition_for_day, self.n_total)
            self.infection_type = util.generate_ndarray_from_shared_memory_int(self.shm_infection_type, self.n_total)
            self.infection_severity = util.generate_ndarray_from_shared_memory_int(self.shm_infection_severity, self.n_total)
            self.vaccination_doses = util.generate_ndarray_from_shared_memory_int(self.shm_vaccination_doses, self.n_total)

            time_taken = time.time() - start
            print("agents_mp convert_from_shared_memory_dynamic (epi) time taken: " + str(time_taken))

    def calculate_memory_size(self, attr_name=None):
        total_size = sum([sys.getsizeof(getattr(self, attr)) for attr in dir(self) if attr_name is None or attr==attr_name])
        return total_size
    
    def log_memory_size(self, attr_name=None):
        log = [[attr, sys.getsizeof(getattr(self, attr))] for attr in dir(self) if attr_name is None or attr==attr_name]
        return log
    

