import numpy as np
import multiprocessing as mp
import sys
import time

# with this approach, agent properties are represented by indices within the different arrays
# this class supports both multiprocessing.RawArray and Numpy arrays
# the former is used with multiprocessing environments and the latter is used with Dask environments
# this approach enables the concept of static shared memory in different modes
# with multiprocessing it represents actual static shared memory that can be retrieved from multiple processes at the same time
# with dask it represents static memory that is loaded up in each worker at the beginning of the simulation and retained for the duration of the simulation
# since it is "mostly" static data there is no need for any additional synchronisation locking mechanisms
# this approach offers fast access of data, and it also ensures that the majority of the data is pre-loaded wherever it is needed
# in the case of tourism, data is passed as default values in the beginning e.g. -1, and then updated with the actual values every day as new tourists arrive
# however, as tourists depart, the data used up to represent them cannot be released, and if set as defaults, would still be using the same size in memory
# this limitation is due to how mp.RawArray and Numpy arrays work, in that, they utilise contiguous blocks of memory
# in this regard, this approach is extremely wasteful in terms of memory usage
# a better approach would be to use something like mp.manager.dict for multiprocessing environments and normal dict for dask environments
# however, mp.manager.dict includes fine-grained locking both for reads and writes and hence was not initially opted for
# a hybrid approach may be considered, whereby the local agents are maintained as static data, while the tourist information is maintained as mp.manager.dict
# this would introduce a small overhead in that for every read/write we would have to decide whether the agent is a tourist or local
# however, this will take virtually 0 seconds, due to us knowing outright how many locals are present in the simulation e.g. if agentid > 500k = tourist
# by using this approach we would be able to to reduce the memory size considerably (indeed there are more tourists than locals over the duration of the simulation)
# while maintaining the fast direct access and static nature of the data (apart from the overhead incurred by the fine-grained read/write locks of mp.manager.dict)
class Static:
    def __init__(self) -> None:
        self.n_total = None
        self.n_locals = None
        self.n_tourists = None

        self.age = [] # int
        self.sc_student = []
        self.empstatus = []
        self.empind = []
        self.empftpt = []
        self.res_cellid = []
        self.work_cellid = []
        self.school_cellid = []
        self.age_bracket_index = []
        self.epi_age_bracket_index = []
        self.working_age_bracket_index = []
        self.soc_rate = []
        self.guardian_id = []
        self.isshiftbased = []
        self.pub_transp_reg = []
        self.ent_activity = []        
        # self.busdriver = []

        self.shm_age = [] # int
        self.shm_sc_student = []
        self.shm_empstatus = []
        self.shm_empind = []
        self.shm_empftpt = []
        self.shm_res_cellid = []
        self.shm_work_cellid = []
        self.shm_school_cellid = []
        self.shm_age_bracket_index = []
        self.shm_epi_age_bracket_index = []
        self.shm_working_age_bracket_index = []
        self.shm_soc_rate = []
        self.shm_guardian_id = []
        self.shm_isshiftbased = []
        self.shm_pub_transp_reg = []
        self.shm_ent_activity = []
        # self.shm_busdriver = []

    def populate(self, data, n_locals, n_tourists, is_shm=True):
        start = time.time()

        n_total = n_locals + n_tourists

        self.n_total = n_total
        self.n_locals = n_locals
        self.n_tourists = n_tourists

        for _, properties in data.items():
            if len(properties) > 0:
                self.age.append(properties["age"] if "age" in properties else None)
                self.sc_student.append(properties["sc_student"] if "sc_student" in properties else None)
                self.empstatus.append(properties["empstatus"] if "empstatus" in properties else None)
                self.empind.append(properties["empind"] if "empind" in properties else None)
                self.empftpt.append(properties["empftpt"] if "empftpt" in properties else None)
                self.res_cellid.append(properties["res_cellid"])
                self.work_cellid.append(properties["work_cellid"])
                self.school_cellid.append(properties["school_cellid"])
                self.age_bracket_index.append(properties["age_bracket_index"] if "age_bracket_index" in properties else None)
                self.epi_age_bracket_index.append(properties["epi_age_bracket_index"] if "epi_age_bracket_index" in properties else None)
                self.working_age_bracket_index.append(properties["working_age_bracket_index"] if "working_age_bracket_index" in properties else None)
                self.soc_rate.append(properties["soc_rate"] if "soc_rate" in properties else None)
                self.guardian_id.append(properties["guardian_id"] if "guardian_id" in properties else None)
                self.isshiftbased.append(properties["isshiftbased"] if "isshiftbased" in properties else None) # this is calculated in working schedule generation on first day
                self.pub_transp_reg.append(properties["pub_transp_reg"])
                self.ent_activity.append(properties["ent_activity"] if "ent_activity" in properties else None)
                # self.busdriver.append(None)
            else:
                self.age.append(None)
                self.sc_student.append(None)
                self.empstatus.append(None)
                self.empind.append(None)
                self.empftpt.append(None)
                self.res_cellid.append(None)
                self.work_cellid.append(None)
                self.school_cellid.append(None)
                self.age_bracket_index.append(None)
                self.epi_age_bracket_index.append(None)
                self.working_age_bracket_index.append(None)
                self.soc_rate.append(None)
                self.guardian_id.append(None)
                self.isshiftbased.append(None) # this is calculated in working schedule generation on first day
                self.pub_transp_reg.append(None)
                self.ent_activity.append(None)
                # self.busdriver.append(None)

        if is_shm:
            self.convert_to_shared_memory_readonly(loadall=True, clear_normal_memory=True)
        else:
            self.convert_to_numpy_readonly(loadall=True, clear_normal_memory=True)

        time_taken = time.time() - start
        print("agents_mp populate time taken: " + str(time_taken))
        # self.convert_to_ndarray()

    def convert_to_shared_memory_readonly(self, loadall=False, itinerary=False, contactnetwork=False, clear_normal_memory=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            self.shm_age = self.generate_shared_memory_int(self.age)
            self.shm_sc_student = self.generate_shared_memory_int(self.sc_student)
            self.shm_empstatus = self.generate_shared_memory_int(self.empstatus)
            self.shm_empind = self.generate_shared_memory_int(self.empind)
            self.shm_empftpt = self.generate_shared_memory_int(self.empftpt)
            self.shm_work_cellid = self.generate_shared_memory_int(self.work_cellid)
            self.shm_school_cellid = self.generate_shared_memory_int(self.school_cellid)
            self.shm_working_age_bracket_index = self.generate_shared_memory_int(self.working_age_bracket_index)
            self.shm_guardian_id = self.generate_shared_memory_int(self.guardian_id)
            self.shm_pub_transp_reg = self.generate_shared_memory_int(self.pub_transp_reg)
            self.shm_ent_activity = self.generate_shared_memory_int(self.ent_activity)
            self.shm_isshiftbased = self.generate_shared_memory_int(self.isshiftbased)
            # self.shm_busdriver = self.generate_shared_memory_int(self.busdriver)
            self.shm_res_cellid = self.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = self.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_shared_memory_int(self.soc_rate, "f")
        elif contactnetwork:
            self.shm_res_cellid = self.generate_shared_memory_int(self.res_cellid)
            self.shm_age_bracket_index = self.generate_shared_memory_int(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_shared_memory_int(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_shared_memory_int(self.soc_rate, "f")
        elif itinerary:
            self.shm_age = self.generate_shared_memory_int(self.age)
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
            self.shm_pub_transp_reg = self.generate_shared_memory_int(self.pub_transp_reg)

        if clear_normal_memory:
            self.clear_non_shared_memory()

        time_taken = time.time() - start
        print("agents_mp convert_to_shared_memory_readonly time taken: " + str(time_taken))

    def clear_non_shared_memory(self):
        self.age = [] # int
        self.sc_student = []
        self.empstatus = []
        self.empind = []
        self.empftpt = []
        self.res_cellid = []
        self.work_cellid = []
        self.school_cellid = []
        self.age_bracket_index = []
        self.epi_age_bracket_index = []
        self.working_age_bracket_index = []
        self.soc_rate = []
        self.guardian_id = []
        self.isshiftbased = []
        self.pub_transp_reg = []
        self.ent_activity = []        
        # self.busdriver = []

    def generate_shared_memory_int(self, static_data, type="i"):
        default_value = -1 if type == "i" else 0.0

        converted_data = [default_value if item is None else item for item in static_data]

        return mp.RawArray(type, converted_data)
    
    def generate_numpy_array(self, static_data):
        return np.array(static_data)

    def convert_to_numpy_readonly(self, loadall=False, itinerary=False, contactnetwork=False, clear_normal_memory=False):
        if not loadall and not itinerary and not contactnetwork:
            loadall = True
    
        start = time.time()

        if loadall:
            self.shm_age = self.generate_numpy_array(self.age)
            self.shm_sc_student = self.generate_numpy_array(self.sc_student)
            self.shm_empstatus = self.generate_numpy_array(self.empstatus)
            self.shm_empind = self.generate_numpy_array(self.empind)
            self.shm_empftpt = self.generate_numpy_array(self.empftpt)
            self.shm_work_cellid = self.generate_numpy_array(self.work_cellid)
            self.shm_school_cellid = self.generate_numpy_array(self.school_cellid)
            self.shm_working_age_bracket_index = self.generate_numpy_array(self.working_age_bracket_index)
            self.shm_guardian_id = self.generate_numpy_array(self.guardian_id)
            self.shm_pub_transp_reg = self.generate_numpy_array(self.pub_transp_reg)
            self.shm_ent_activity = self.generate_numpy_array(self.ent_activity)
            self.shm_isshiftbased = self.generate_numpy_array(self.isshiftbased)
            # self.shm_busdriver = self.generate_shared_memory_int(self.busdriver)
            self.shm_res_cellid = self.generate_numpy_array(self.res_cellid)
            self.shm_age_bracket_index = self.generate_numpy_array(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_numpy_array(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_numpy_array(self.soc_rate)
        elif contactnetwork:
            self.shm_res_cellid = self.generate_numpy_array(self.res_cellid)
            self.shm_age_bracket_index = self.generate_numpy_array(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_numpy_array(self.epi_age_bracket_index)
            self.shm_soc_rate = self.generate_numpy_array(self.soc_rate)
        elif itinerary:
            self.shm_age = self.generate_numpy_array(self.age)
            self.shm_sc_student = self.generate_numpy_array(self.sc_student)
            self.shm_empstatus = self.generate_numpy_array(self.empstatus)
            self.shm_empind = self.generate_numpy_array(self.empind)
            self.shm_empftpt = self.generate_numpy_array(self.empftpt)
            self.shm_ent_activity = self.generate_numpy_array(self.ent_activity)
            self.shm_isshiftbased = self.generate_numpy_array(self.isshiftbased)
            self.shm_guardian_id = self.generate_numpy_array(self.guardian_id)
            self.shm_age_bracket_index = self.generate_numpy_array(self.age_bracket_index)
            self.shm_epi_age_bracket_index = self.generate_numpy_array(self.epi_age_bracket_index)
            self.shm_working_age_bracket_index = self.generate_numpy_array(self.working_age_bracket_index)
            self.shm_res_cellid = self.generate_numpy_array(self.res_cellid)
            self.shm_work_cellid = self.generate_numpy_array(self.work_cellid)
            self.shm_school_cellid = self.generate_numpy_array(self.school_cellid)
            self.shm_pub_transp_reg = self.generate_numpy_array(self.pub_transp_reg)

        if clear_normal_memory:
            self.clear_non_shared_memory()

        time_taken = time.time() - start
        print("static.py convert_to_numpy_readonly time taken: " + str(time_taken))
    
    def get(self, index, name):
        return getattr(self, "shm_" + name)[index]
    
    def set(self, index, name, value):
        if name == "age":
            self.shm_age[index] = value
        elif name == "sc_student":
            self.shm_sc_student[index] = value
        elif name == "empstatus":
            self.shm_empstatus[index] = value
        elif name == "empind":
            self.shm_empind[index] = value
        elif name == "empftpt":
            self.shm_empftpt[index] = value
        elif name == "res_cellid":
            self.shm_res_cellid[index] = value
        elif name == "work_cellid":
            self.shm_work_cellid[index] = value
        elif name == "school_cellid":
            self.shm_school_cellid[index] = value
        elif name == "age_bracket_index":
            self.shm_age_bracket_index[index] = value
        elif name == "epi_age_bracket_index":
            self.shm_epi_age_bracket_index[index] = value
        elif name == "working_age_bracket_index":
            self.shm_working_age_bracket_index[index] = value
        elif name == "soc_rate":
            self.shm_soc_rate[index] = value
        elif name == "guardian_id":
            self.shm_guardian_id[index] = value
        elif name == "isshiftbased":
            self.shm_isshiftbased[index] = value
        elif name == "pub_transp_reg":
            self.shm_pub_transp_reg[index] = value
        elif name == "ent_activity":
            self.shm_ent_activity[index] = value
    