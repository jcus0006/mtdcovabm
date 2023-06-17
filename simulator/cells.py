import numpy as np
import random
import math
import copy
import bisect
from classes.accomgroup import AccomGroup
from simulator import util
import multiprocessing as mp

class Cells:
    def __init__(self, manager, agents, cells, cellindex, seed=6):
        self.manager = manager
        self.agents = agents
        self.cells = cells
        self.cellindex = cellindex
        np.random.seed(seed)

    # return households which is a dict of dicts, where the key is the household id, and the values are:
    # member_uids representing the residents in the household, reference_uid representing the reference person, and reference_age the reference person age
    # cell splitting is not required in this case, the household is the cell
    # this method also returns the households to be assigned as employers; the only criteria being used is that the first "household_as_employer_count" where the education is tertiary will be assigned
    # had more fine-grained data been available in terms of net worth or income, this could have also been used
    def convert_households(self, households_original, workplaces, workplaces_params):
        households = self.manager.dict()
        householdscells = self.manager.dict()
        householdsworkplaces = self.manager.dict()
        householdsworkplacescells = self.manager.dict()

        households_employees_groups = []

        if len(workplaces) > 0:
            households_as_workplaces_employees_groups = [wp["member_uids"] for wp in workplaces if wp["indid"] == 20]
            households_as_workplaces_employees = [emp for grp in households_as_workplaces_employees_groups for emp in grp]
            households_as_workplace_params = workplaces_params[19] # index 0 based
            min_household_employees, max_household_employees = households_as_workplace_params[1], households_as_workplace_params[2]
            households_employees_group_sizes = util.random_group_partition(len(households_as_workplaces_employees), min_household_employees, max_household_employees)

            members_start = 0
            for index, group_size in enumerate(households_employees_group_sizes):
                members_count = group_size

                members_end = members_start + members_count

                temp_employees = households_as_workplaces_employees[members_start:members_end]

                members_start = members_end

                households_employees_groups.append(temp_employees)

        households_employees_groups_index = 0

        for household in households_original:
            member_uids = household["member_uids"]
            hhid = household["hhid"]
            ref_uid = household["reference_uid"]
            ref_age = household["reference_age"] 
            ref_agent = self.agents[ref_uid]

            this_hh_employees_uids = []
            if households_employees_groups_index < len(households_employees_groups) and ref_agent["edu"] == 5:
                this_hh_employees_uids = households_employees_groups[households_employees_groups_index]

                if len(self.agents) > 0:
                    for uid in this_hh_employees_uids:
                        agent = self.agents[uid]
                        agent["wpid"] = None
                        agent["hhwpid"] = hhid 
                        agent["work_cellid"] = self.cellindex
                        self.agents[uid] = agent

                households_employees_groups_index += 1

            households[hhid] = { "hhid": hhid, "resident_uids": np.array(member_uids), "staff_uids": np.array(this_hh_employees_uids), "reference_uid": ref_uid, "reference_age": ref_age, "visitor_uids": np.array([])}
            householdsworkplaces[hhid] = households[hhid]

            self.cells[self.cellindex] = { "type": "household", "place": households[hhid]}
            householdscells[self.cellindex] = self.cells[self.cellindex]
            householdsworkplacescells[self.cellindex] = self.cells[self.cellindex]

            if len(self.agents) > 0:
                for uid in member_uids:
                    agent = self.agents[uid]
                    agent["res_cellid"] = self.cellindex
                    agent["curr_cellid"] = self.cellindex
                    self.agents[uid] = agent

            self.cellindex += 1

        return households, householdsworkplaces, householdscells, householdsworkplacescells

    # return industries_by_indid which is a dict of dict of dict of dict with the below format:
    # industries_by_indid (indid) -> workplaces_by_wpid (wpid) -> cells_by_cellid (cellid) -> dict with member_uids key/value pair
    # workplaces are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
    # cells are split by an algorithm that ensures that cell sizes are balanced; at the same time as close to max_members as possible
    def split_workplaces_by_cellsize(self, workplaces, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype, workplaces_cells_params, hospital_cells_params, testing_hubs_cells_params, vaccination_hubs_cells_params, airport_cells_params, accom_cells_params, transport, entertainment_activity_dist):
        industries_by_indid = self.manager.dict()
        workplacescells = self.manager.dict()
        restaurantcells = self.manager.dict()
        hospitalcells = self.manager.dict()
        testinghubcells = self.manager.dict()
        vaccinationhubcells = self.manager.dict()
        accommodationcells = self.manager.dict()
        accommodationcells_by_accomid = self.manager.dict()
        breakfastcells_by_accomid = self.manager.dict()
        entertainmentcells = self.manager.dict()
        airportcells = self.manager.dict()

        airport_industries = [7, 8, 9, 19]
        airport_ensure_workplace_per_industry = [False, False, False, False] # [7, 8, 9, 19]
        # accomgroups = AccomGroup()

        bus_drivers = []

        if len(transport) > 0:       
            transportation_employees_groups = [wp["member_uids"] for wp in workplaces if wp["indid"] == 8]
            transportation_employees = [emp for grp in transportation_employees_groups for emp in grp]
            transportation_employeers_drivers_pool = [uid for uid in transportation_employees if self.agents[uid]["edu"] != 5]

            bus_count = len(transport)
            bus_drivers = np.random.choice(transportation_employeers_drivers_pool, bus_count)

        healthcare_workplaces_sizes = {}
        hospital_ids = []

        # pick the largest one as the main hospital
        # pick hosp_count - 1, at random, as the other hospitals

        healthcare_workplaces = [wp for wp in workplaces if wp["indid"] == 17]
        for wp in healthcare_workplaces:
            healthcare_workplaces_sizes[wp["wpid"]] = len(wp["member_uids"])

            healthcare_workplaces_sizes = dict(sorted(healthcare_workplaces_sizes.items(), key=lambda item: item[1]))

            healthcare_workplaces_keys = np.array(list(healthcare_workplaces_sizes.keys()))
        
        if len(hospital_cells_params) > 0:
            hosp_cell_size, hosp_cell_size_spare, hosp_count, hosp_beds_per_employee, hosp_avg_beds_per_rooms = hospital_cells_params[0], hospital_cells_params[1], hospital_cells_params[2], hospital_cells_params[3], hospital_cells_params[4]

            hospital_ids = healthcare_workplaces_keys[-1:]

            hospital_ids = np.append(hospital_ids, np.random.choice(healthcare_workplaces_keys, hosp_count-1))

        if len(testing_hubs_cells_params) > 0:
            test_hub_cell_size, test_hub_cell_size_spare, test_hub_healthcare_percent = testing_hubs_cells_params[0], testing_hubs_cells_params[1], testing_hubs_cells_params[2]
            
            testing_hubs_n = round(len(healthcare_workplaces) * test_hub_healthcare_percent)

            if testing_hubs_n == 0:
                testing_hubs_n = 1

            if testing_hubs_n >= len(healthcare_workplaces_keys):
                testing_hubs_cell_ids = healthcare_workplaces_keys
            else:
                testing_hubs_cell_ids = np.random.choice(healthcare_workplaces_keys, size=testing_hubs_n)

        if len(vaccination_hubs_cells_params) > 0:
            vacc_hub_cell_size, vacc_hub_cell_size_spare, vacc_hub_healthcare_percent = vaccination_hubs_cells_params[0], vaccination_hubs_cells_params[1], vaccination_hubs_cells_params[2]
            
            vacc_hubs_n = round(len(healthcare_workplaces) * vacc_hub_healthcare_percent)

            if vacc_hubs_n == 0:
                vacc_hubs_n = 1

            if vacc_hubs_n >= len(healthcare_workplaces_keys):
                vaccination_hubs_cell_ids = healthcare_workplaces_keys
            else:
                vaccination_hubs_cell_ids = np.random.choice(healthcare_workplaces_keys, size=vacc_hubs_n)

        activity_options = [activity_dist[0] for activity_dist in entertainment_activity_dist]
        activity_weights = [activity_dist[1] for activity_dist in entertainment_activity_dist]

        hospital_id = 0

        for workplace in workplaces:
            workplaces_by_wpid = self.manager.dict()
            cells_by_cellid = self.manager.dict()

            employees = workplace["member_uids"]
            wpid = workplace["wpid"]
            indid = workplace["indid"]

            is_entertainment = indid == 18
            is_household = indid == 20
            is_airport = False
            is_potentially_airport_external = indid in airport_industries # retail, transp, food, other service activities

            if not is_household:
                accomid = workplace["accomid"] if workplace["accomid"] is not None else -1 # -1 if not an accommodation
                accomtypeid = workplace["accomtypeid"] if workplace["accomtypeid"] is not None else -1 # -1 if not an accommodation
                
                is_accom = accomid > -1
                is_restaurant = indid == 9 and not is_accom
                is_hospital = wpid in hospital_ids
                is_testinghub = wpid in testing_hubs_cell_ids
                is_vaccinationhub = wpid in vaccination_hubs_cell_ids

                cellsize, cellsizespare = workplaces_cells_params[indid-1][1], workplaces_cells_params[indid-1][2]

                if is_hospital:
                    cellsize, cellsizespare = hosp_cell_size, hosp_cell_size_spare

                if is_testinghub:
                    if test_hub_cell_size > cellsize:
                        cellsize = test_hub_cell_size

                    if test_hub_cell_size_spare > cellsizespare:
                        cellsizespare = test_hub_cell_size_spare

                if is_vaccinationhub:
                    if vacc_hub_cell_size > cellsize:
                        cellsize = vacc_hub_cell_size

                    if vacc_hub_cell_size_spare > cellsizespare:
                        cellsizespare = vacc_hub_cell_size_spare

                if is_accom:
                    cellsize, cellsizespare = accom_cells_params[0], accom_cells_params[1]

                if is_potentially_airport_external and not is_accom and not is_restaurant and not is_hospital:
                    airport_industries_index = airport_industries.index(indid)

                    if airport_ensure_workplace_per_industry[airport_industries_index]: # if 1 wp for this ind has been set for airport, use cell param e.g. 3% 
                        is_airport_random = random.random()
                        is_airport_prob = airport_cells_params[2]

                        is_airport = is_airport_random < is_airport_prob
                    else:
                        is_airport = True # force as is_airport, ensure at least 1 wp from airport_industries
                        airport_ensure_workplace_per_industry[airport_industries_index] = True # mark as true, next wp from ind, will be based on cell param prob

                    cellsize, cellsizespare = airport_cells_params[0], airport_cells_params[1]

                # print(accomtypeid)

                num_employees = len(employees)

                max_members = int(cellsize * (1 - cellsizespare))

                # start_cell_index = self.cellindex

                # If the number of members is less than or equal to "max_members", no splitting needed
                if len(employees) <= max_members:
                    if not is_accom and not is_hospital and not is_entertainment and not is_airport:
                        cells_by_cellid[self.cellindex] = { "wpid": wpid, "indid": indid, "staff_uids": np.array(employees), "visitor_uids": np.array([])}

                        self.cells[self.cellindex] = { "type": "workplace", "place": cells_by_cellid[self.cellindex]}

                        if is_restaurant:
                            restaurantcells[self.cellindex] = self.cells[self.cellindex]

                        if is_testinghub:
                            testinghubcells[self.cellindex] = self.cells[self.cellindex]

                        if is_vaccinationhub:
                            vaccinationhubcells[self.cellindex] = self.cells[self.cellindex]

                    if is_accom:
                        cells_by_cellid[self.cellindex] = { "accomid": accomid, "accomtypeid": accomtypeid, "staff_uids": np.array(employees)}

                        self.cells[self.cellindex] = { "type": "accom", "place": cells_by_cellid[self.cellindex]}

                        accommodationcells[self.cellindex] = self.cells[self.cellindex]

                        if accomid not in accommodationcells_by_accomid:
                            accommodationcells_by_accomid[accomid] = []

                        accommodationcells_by_accomid[accomid].append(self.cells[self.cellindex])

                        if accomid not in breakfastcells_by_accomid:
                            breakfastcells_by_accomid[accomid] = self.manager.dict()

                        accomid_breakfastcells = breakfastcells_by_accomid[accomid] 
                                
                        accomid_breakfastcells[self.cellindex] = self.cells[self.cellindex]

                    if is_hospital:
                        cells_by_cellid[self.cellindex] = { "hospitalid": hospital_id, "staff_uids": np.array(employees)}

                        self.cells[self.cellindex] = { "type": "hospital", "place": cells_by_cellid[self.cellindex]}

                        hospitalcells[self.cellindex] = self.cells[self.cellindex]

                        if is_testinghub:
                            testinghubcells[self.cellindex] = self.cells[self.cellindex]

                        if is_vaccinationhub:
                            vaccinationhubcells[self.cellindex] = self.cells[self.cellindex]

                    sampled_activity = -1
                    if is_entertainment:
                        sampled_activity = np.random.choice(activity_options, 1, p=activity_weights)[0]

                        cells_by_cellid[self.cellindex] = { "wpid": wpid ,"indid": indid, "activityid": sampled_activity, "staff_uids": np.array(employees), "visitor_uids": np.array([])}

                        self.cells[self.cellindex] = { "type": "entertainment", "place": cells_by_cellid[self.cellindex]}

                        if sampled_activity not in entertainmentcells:
                            entertainmentcells[sampled_activity] = self.manager.dict()

                        entertainmentcells_by_activity = entertainmentcells[sampled_activity]

                        entertainmentcells_by_activity[self.cellindex] = self.cells[self.cellindex]

                    if is_airport:
                        cells_by_cellid[self.cellindex] = { "wpid": wpid ,"indid": indid,  "staff_uids": np.array(employees), "visitor_uids": np.array([])}

                        self.cells[self.cellindex] = { "type": "airport", "place": cells_by_cellid[self.cellindex]}

                        airportcells[self.cellindex] = self.cells[self.cellindex]

                    workplacescells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in employees:
                            agent = self.agents[uid]
                            agent["busdriver"] = uid in bus_drivers # bus drivers still go to the designated place of work, and then are randomly allocated into the first cell of a transport bus
                            agent["work_cellid"] = self.cellindex
                            agent["ent_activity"] = sampled_activity
                            self.agents[uid] = agent

                    self.cellindex += 1
                else:
                    # Calculate the number of groups needed, considering spare space
                    cell_sizes = self.split_balanced_cells_close_to_x(num_employees, max_members)

                    employees = np.array(employees)
                    # np.random.shuffle(employees)

                    accom_breakfast_cell_count = 0

                    if is_accom:
                        breakfast_cell_perc = accom_cells_params[2]
                        
                        accom_breakfast_cell_count = math.ceil(breakfast_cell_perc * len(cell_sizes))

                    members_start = 0
                    for index, cell_size in enumerate(cell_sizes):
                        members_count = cell_size

                        members_end = members_start + members_count

                        temp_members = employees[members_start:members_end]

                        members_start = members_end

                        if not is_accom and not is_hospital and not is_entertainment and not is_airport: # normal workplace
                            cells_by_cellid[self.cellindex] = { "wpid": wpid, "indid": indid, "staff_uids": np.array(temp_members), "visitor_uids": np.array([])}

                            self.cells[self.cellindex] = { "type": "workplace", "place": cells_by_cellid[self.cellindex]}

                            if is_restaurant:
                                restaurantcells[self.cellindex] = self.cells[self.cellindex]

                            if is_testinghub:
                                testinghubcells[self.cellindex] = self.cells[self.cellindex]

                            if is_vaccinationhub:
                                vaccinationhubcells[self.cellindex] = self.cells[self.cellindex]

                        if is_accom:
                            cells_by_cellid[self.cellindex] = { "accomid": accomid, "accomtypeid": accomtypeid, "staff_uids": np.array(temp_members)}

                            self.cells[self.cellindex] = { "type": "accom", "place": cells_by_cellid[self.cellindex]}

                            accommodationcells[self.cellindex] = self.cells[self.cellindex]

                            if accomid not in accommodationcells_by_accomid:
                                accommodationcells_by_accomid[accomid] = []

                            accommodationcells_by_accomid[accomid].append(self.cells[self.cellindex])

                            if accom_breakfast_cell_count > 0:
                                if accomid not in breakfastcells_by_accomid:
                                    breakfastcells_by_accomid[accomid] = self.manager.dict()

                                accomid_breakfastcells = breakfastcells_by_accomid[accomid] 
                                
                                accomid_breakfastcells[self.cellindex] = self.cells[self.cellindex]

                                accom_breakfast_cell_count -= 1

                        if is_hospital:
                            cells_by_cellid[self.cellindex] = { "hospitalid": hospital_id, "staff_uids": np.array(temp_members)}

                            self.cells[self.cellindex] = { "type": "hospital", "place": cells_by_cellid[self.cellindex]}

                            hospitalcells[self.cellindex] = self.cells[self.cellindex]

                            if is_testinghub:
                                testinghubcells[self.cellindex] = self.cells[self.cellindex]

                            if is_vaccinationhub:
                                vaccinationhubcells[self.cellindex] = self.cells[self.cellindex]

                        sampled_activity = -1
                        if is_entertainment:
                            sampled_activity = np.random.choice(activity_options, 1, p=activity_weights)[0]

                            cells_by_cellid[self.cellindex] = { "wpid": wpid, "indid": indid, "activityid": sampled_activity, "staff_uids": np.array(temp_members), "visitor_uids": np.array([])}

                            self.cells[self.cellindex] = { "type": "entertainment", "place": cells_by_cellid[self.cellindex]}

                            if sampled_activity not in entertainmentcells:
                                entertainmentcells[sampled_activity] = self.manager.dict()

                            entertainmentcells_by_activity = entertainmentcells[sampled_activity]

                            entertainmentcells_by_activity[self.cellindex] = self.cells[self.cellindex]

                        if is_airport:
                            cells_by_cellid[self.cellindex] = { "wpid": wpid ,"indid": indid,  "staff_uids": np.array(temp_members), "visitor_uids": np.array([])}

                            self.cells[self.cellindex] = { "type": "airport", "place": cells_by_cellid[self.cellindex]}

                            airportcells[self.cellindex] = self.cells[self.cellindex]
                                
                        workplacescells[self.cellindex] = self.cells[self.cellindex]

                        if len(self.agents) > 0:
                            for uid in temp_members:
                                agent = self.agents[uid]
                                agent["busdriver"] = uid in bus_drivers # bus drivers still go to the designated place of work, and then are randomly allocated into the first cell of a transport bus
                                agent["work_cellid"] = self.cellindex
                                agent["ent_activity"] = sampled_activity
                                self.agents[uid] = agent
                        
                        self.cellindex += 1

                if is_accom and len(roomsizes_by_accomid_by_accomtype) > 0:
                    cells_by_cellid, workplacescells, accommodationcells, accommodationcells_by_accomid = self.create_accom_rooms(accomid, accomtypeid, cells_by_cellid, workplacescells, accommodationcells, accommodationcells_by_accomid, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype)

                if is_hospital:
                    cells_by_cellid, workplacescells, hospitalcells = self.create_hospital_rooms(hospital_id, num_employees, hosp_beds_per_employee, hosp_avg_beds_per_rooms, cells_by_cellid, workplacescells, hospitalcells)

                    hospital_id += 1

                # assign cells into wpid
                workplaces_by_wpid[wpid] = cells_by_cellid

                # assign workplaces into indid
                if indid in industries_by_indid:
                    existing_workplaces_by_indid = industries_by_indid[indid]
                    for tempwpid, tempcells in workplaces_by_wpid.items():
                        existing_workplaces_by_indid[tempwpid] = tempcells
                else:
                    industries_by_indid[indid] = workplaces_by_wpid

        return industries_by_indid, workplacescells, restaurantcells, accommodationcells, accommodationcells_by_accomid, breakfastcells_by_accomid, rooms_by_accomid_by_accomtype, hospitalcells, testinghubcells, vaccinationhubcells, entertainmentcells, airportcells

    # return schools_by_type which is a dict of dict of dict of dict with the below format:
    # schools_by_type (indid) -> schools_by_scid (wpid) -> cells_by_cellid (cellid) -> cellinfodict (clid, student_uids, teacher_uids, non_teaching_staff_uids)
    # classrooms are assigned to cells as is
    # non-teaching staff are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
    # cells are split by an algorithm that ensures that cell sizes are as balanced and as close to max_members as possible
    def split_schools_by_cellsize(self, schools, cellsize, cellsizespare):
        schools_by_type = self.manager.dict()
        schoolscells = self.manager.dict()
        classroomscells = self.manager.dict()

        for school in schools:
            schools_by_scid = self.manager.dict()
            cells_by_cellid = self.manager.dict()

            scid = school["scid"]
            sctype = school["sc_type"]

            nonteachingstaff = school["non_teaching_staff_uids"] # consider separately from classrooms
            classrooms = school["classrooms"]

            num_nonteachingstaff = len(nonteachingstaff)

            max_members = int(cellsize * (1 - cellsizespare))

            # If the number of nonteachingstaff is less than or equal to "max_members", no splitting needed
            if len(nonteachingstaff) <= max_members:
                cells_by_cellid[self.cellindex] = { "scid": scid, "non_teaching_staff_uids" : np.array(nonteachingstaff) }
                
                self.cells[self.cellindex] = { "type": "school", "place": cells_by_cellid[self.cellindex]}
                schoolscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in nonteachingstaff:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex
                        self.agents[uid] = agent

                self.cellindex += 1
            else:
                # Calculate the number of groups needed, considering spare space

                cell_sizes = self.split_balanced_cells_close_to_x(num_nonteachingstaff, max_members)

                nonteachingstaff = np.array(nonteachingstaff)
                # np.random.shuffle(nonteachingstaff)

                members_start = 0
                for index, cell_size in enumerate(cell_sizes):
                    members_count = cell_size

                    members_end = members_start + members_count

                    temp_nonteachingstaff = nonteachingstaff[members_start:members_end]

                    members_start = members_end

                    cells_by_cellid[self.cellindex] = { "scid": scid, "non_teaching_staff_uids" : np.array(temp_nonteachingstaff) }

                    self.cells[self.cellindex] = { "type": "school", "place": cells_by_cellid[self.cellindex]}
                    schoolscells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in temp_nonteachingstaff:
                            agent = self.agents[uid]
                            agent["school_cellid"] = self.cellindex
                            agent["work_cellid"] = self.cellindex
                            self.agents[uid] = agent

                    self.cellindex += 1

            for classroom in classrooms:
                clid = classroom["clid"]
                students = classroom["student_uids"]
                teachers = classroom["teacher_uids"]

                cells_by_cellid[self.cellindex] = { "scid": scid, "clid":clid, "student_uids":np.array(students), "teacher_uids":np.array(teachers)}

                self.cells[self.cellindex] = { "type": "classroom", "place": cells_by_cellid[self.cellindex]}
                classroomscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in students:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex
                        self.agents[uid] = agent

                    for uid in teachers:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex
                        agent["work_cellid"] = self.cellindex
                        self.agents[uid] = agent

                self.cellindex += 1
                
            # assign cells into scid
            schools_by_scid[scid] = cells_by_cellid

            # assign schools into type
            if sctype in schools_by_type:
                existing_schools_by_type = schools_by_type[sctype]
                for tempscid, tempcells in schools_by_scid.items():
                    existing_schools_by_type[tempscid] = tempcells
            else:
                schools_by_type[sctype] = schools_by_scid

        return schools_by_type, schoolscells, classroomscells

    # return institutions_by_type which is a dict of dict of dict of dict with the below format:
    # institutions_by_type (indid) -> institutions_by_id (wpid) -> cells_by_cellid (cellid) -> cellinfodict (resident_uids, staff_uids)
    # residents and staff are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
    # cells are split by an algorithm that ensures that cell sizes are as balanced and as close to max_members as possible
    def split_institutions_by_cellsize(self, institutions, cellsize, cellsizespare):
        institutions_by_type = self.manager.dict()
        institutionscells = self.manager.dict()

        for institution in institutions:
            institutions_by_id = self.manager.dict()
            cells_by_cellid = self.manager.dict()

            residents = institution["resident_uids"]
            staff = institution["staff_uids"]
            members = residents + staff
            num_residents = len(residents)
            num_staff = len(staff)
            num_members = len(members)
            instid = institution["instid"]
            insttypeid = institution["insttypeid"]

            staff_resident_ratio = num_staff / num_members

            max_members = int(cellsize * (1 - cellsizespare))

            # If the number of members is less than or equal to "max_members", no splitting needed
            if num_members <= cellsize:
                cells_by_cellid[self.cellindex] = { "instid": instid, "resident_uids": np.array(residents), "staff_uids": np.array(staff)}

                self.cells[self.cellindex] = { "type": "institution", "place": cells_by_cellid[self.cellindex]}
                institutionscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in residents:
                        agent = self.agents[uid]
                        agent["inst_cellid"] = self.cellindex
                        agent["res_cellid"] = self.cellindex
                        agent["curr_cellid"] = self.cellindex
                        self.agents[uid] = agent

                    for uid in staff:
                        agent = self.agents[uid]
                        agent["inst_cellid"] = self.cellindex
                        self.agents[uid] = agent

                self.cellindex += 1
            else:
                # Calculate the number of groups needed, considering spare space

                cell_sizes, staff_cell_sizes, res_cell_sizes = self.split_balanced_cells_staff_residents_close_to_x(num_residents, num_staff, max_members, staff_resident_ratio)
                
                staff = np.array(staff)
                residents = np.array(residents)

                # np.random.shuffle(staff)
                # np.random.shuffle(residents)
                
                staff_start = 0
                resident_start = 0
                for index, this_cell_count in enumerate(cell_sizes):

                    # first assign staff
                    staff_count = staff_cell_sizes[index]

                    staff_end = staff_start + staff_count

                    temp_staff = staff[staff_start:staff_end]

                    staff_start = staff_end # for next iteration
                    
                    # then assign residents
                    resident_count = res_cell_sizes[index]

                    resident_end = resident_start + resident_count

                    temp_residents = residents[resident_start:resident_end]

                    resident_start = resident_end # for next iteration

                    cells_by_cellid[self.cellindex] = { "instid": instid, "resident_uids": temp_residents, "staff_uids": temp_staff }

                    self.cells[self.cellindex] = { "type": "institution", "place": cells_by_cellid[self.cellindex]}
                    institutionscells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in temp_residents:
                            agent = self.agents[uid]
                            agent["inst_cellid"] = self.cellindex
                            agent["res_cellid"] = self.cellindex
                            agent["curr_cellid"] = self.cellindex
                            self.agents[uid] = agent

                        for uid in temp_staff:
                            agent = self.agents[uid]
                            agent["inst_cellid"] = self.cellindex
                            self.agents[uid] = agent

                    self.cellindex += 1
            
            # assign cells into instid
            institutions_by_id[instid] = cells_by_cellid

            # assign workplaces into insttypeid
            if insttypeid in institutions_by_type:
                existing_institutions_by_id = institutions_by_type[insttypeid]
                for tempinstid, tempcells in institutions_by_id.items():
                    existing_institutions_by_id[tempinstid] = tempcells
            else:
                institutions_by_type[insttypeid] = institutions_by_id

        return institutions_by_type, institutionscells
    
    def create_airport_cell(self): # this is a single cell
        airport_cell = {"visitor_uids":[]}

        self.cells[self.cellindex] = {"type":"airport", "place": airport_cell}

        return self.cells[self.cellindex]

    def create_transport_cells(self, buscount, cellsize, cellsizespare, bus_capacities, bus_capacities_dist): 
        transport_by_id = self.manager.dict()
        cells_transport = self.manager.dict()

        max_members = int(cellsize * (1 - cellsizespare))

        bus_categories = [i for i in range(0, len(bus_capacities))]
        sampled_bus_categories = np.random.choice(bus_categories, buscount, p=bus_capacities_dist)

        for busid in range(0, buscount):
            cells_by_cellid = self.manager.dict()

            sampled_bus_capacity = bus_capacities[sampled_bus_categories[busid]]

            if sampled_bus_capacity <= max_members: 
                cells_by_cellid[self.cellindex] = {"busid": busid, "driver_uid": -1, "passenger_uids":[], "capacity": sampled_bus_capacity}

                self.cells[self.cellindex] = {"type":"transport", "place": cells_by_cellid[self.cellindex]}

                cells_transport[self.cellindex] = self.cells[self.cellindex]

                self.cellindex += 1
            else:
                cell_sizes = self.split_balanced_cells_close_to_x(sampled_bus_capacity, max_members)
                
                for index, cell_size in enumerate(cell_sizes):
                    if index == 0:
                        cells_by_cellid[self.cellindex] = { "busid": busid, "driver_uid": -1, "passenger_uids":[], "capacity": cell_size }
                    else:
                        cells_by_cellid[self.cellindex] = { "busid": busid, "passenger_uids":[], "capacity": cell_size }

                    self.cells[self.cellindex] = { "type": "transport", "place": cells_by_cellid[self.cellindex]}

                    cells_transport[self.cellindex] = self.cells[self.cellindex]

                    self.cellindex += 1

            transport_by_id[busid] = cells_by_cellid

        return transport_by_id, cells_transport
    
    def create_accom_rooms(self, accomid, accomtypeid, cells_by_cellid, workplacescells, accommodationcells, accommodationscells_by_accomid, roomsizes_by_accomid_by_accomtype, rooms_by_accomid_by_accomtype):
        roomsbysizes = roomsizes_by_accomid_by_accomtype[accomtypeid][accomid]

        for roomsize, roomids in roomsbysizes.items():
            for roomid in roomids:                        
                cells_by_cellid[self.cellindex] = { "accomid": accomid, "accomtypeid": accomtypeid, "roomid": roomid, "roomsize": roomsize, "guest_uids": np.array([])} # guest_uids here represents tourists that are not assigned yet

                self.cells[self.cellindex] = { "type": "accom", "place": cells_by_cellid[self.cellindex]}

                workplacescells[self.cellindex] = self.cells[self.cellindex]
                accommodationcells[self.cellindex] = self.cells[self.cellindex]
                accommodationscells_by_accomid[accomid].append(self.cells[self.cellindex])

                # accomgroups.append(accomtypeid, accomid, roomid, roomsize, self.cellindex)
                room_by_accomid_by_accomtype = rooms_by_accomid_by_accomtype[accomtypeid][accomid][roomid]
                room_by_accomid_by_accomtype["roomsize"] = roomsize
                room_by_accomid_by_accomtype["cellindex"] = self.cellindex

                self.cellindex += 1

        return cells_by_cellid, workplacescells, accommodationcells, accommodationscells_by_accomid

    def create_hospital_rooms(self, hospital_id, n_employees, hosp_beds_per_employee, hospitalaveragebedsperroom, cells_by_cellid, workplacescells, hospitalcells): # this is a dynamic cell
        total_no_beds = round(n_employees * hosp_beds_per_employee)
        total_no_rooms = round(total_no_beds / hospitalaveragebedsperroom)

        for i in range(0, total_no_rooms):
            cells_by_cellid[self.cellindex] = {"hospitalid": hospital_id, "staff_uids":np.array([]), "roomid": i, "patient_uids":np.array([])}

            self.cells[self.cellindex] = { "type": "hospital", "place": cells_by_cellid[self.cellindex]}

            workplacescells[self.cellindex] = self.cells[self.cellindex]
            hospitalcells[self.cellindex] = self.cells[self.cellindex]

            self.cellindex += 1
        
        return cells_by_cellid, workplacescells, hospitalcells

    def create_religious_cells(self, churchcount, cellsize, cellsizespare, mean_church_capacity, std_dev): # dynamic cell. data for religion as an industry was not available, so created as a form of "dynamic" contact hub       
        sampled_church_sizes = np.round(np.random.normal(mean_church_capacity, std_dev, churchcount)).astype(int)

        churchcells = self.manager.dict()
        churches = self.manager.dict()

        for church_id, sampled_size in enumerate(sampled_church_sizes):
            cells_by_cellid = self.manager.dict()

            max_members = int(cellsize * (1 - cellsizespare))

            if sampled_size <= max_members:
                cells_by_cellid[self.cellindex] = { "churchid": church_id, "visitor_uids" : [], "capacity": sampled_size }
                
                self.cells[self.cellindex] = { "type": "church", "place": cells_by_cellid[self.cellindex]}
                churchcells[self.cellindex] = self.cells[self.cellindex]

                self.cellindex += 1
            else:
                # Calculate the number of groups needed, considering spare space

                cell_sizes = self.split_balanced_cells_close_to_x(sampled_size, max_members)
                
                for index, cell_size in enumerate(cell_sizes):
                    cells_by_cellid[self.cellindex] = { "churchid": church_id, "visitor_uids" : [], "capacity": cell_size }

                    self.cells[self.cellindex] = { "type": "church", "place": cells_by_cellid[self.cellindex]}
                    churchcells[self.cellindex] = self.cells[self.cellindex]

                    self.cellindex += 1

            # assign cells into scid
            churches[church_id] = cells_by_cellid

        return churches, churchcells

    # takes "n" which is the number of agents to allocate, and "x" which is the max number of agents per cell
    # returns "cells": array of int, whereby its length represents the num of cells, and each value, the num of agents to be assigned
    def split_balanced_cells_close_to_x(self, n, x):
        # Calculate the maximum number of cells we can split n into
        max_cells = n // x + 1

        # Initialize a list to store the sizes of each cell
        cells = [0] * max_cells

        # Calculate the initial size of each cell
        init_size = n // max_cells

        # Initialize the remainder
        remainder = n % max_cells

        # Divide the remainder equally among the first few cells
        for i in range(remainder):
            cells[i] = init_size + 1

        # Divide the remaining n among the rest of the cells
        for i in range(remainder, max_cells):
            cells[i] = init_size

        return cells

    # takes "res_n" which is the number of residents to allocate, "staff_n" which is the number of staff to allocate, and "x" which is the max number of agents per cell
    # returns "cells": array of int, whereby its length represents the num of cells, and each value, the num of agents to be assigned
    def split_balanced_cells_staff_residents_close_to_x(self, res_n, staff_n, x, staff_resident_ratio):
        # Calculate the maximum number of cells we can split n into
        resident_staff_ratio = 1 - staff_resident_ratio

        n = res_n + staff_n

        max_cells = n // x + 1

        # Initialize a list to store the sizes of each cell
        cells = [0] * max_cells

        # Initialize a list to store the resident sizes of each cell
        res_cells = cells.copy()

        # Initialize a list to store the staff sizes of each cell
        staff_cells = cells.copy()

        # Calculate the initial size of each cell
        init_size = n // max_cells

        # Calculate the initial number of residents of each cell
        init_res_size = math.ceil(init_size * resident_staff_ratio)

        # Calculate the initial number of staff of cell
        init_staff_size = math.floor(init_size * staff_resident_ratio)

        # Initialize the remainder
        remainder = n % max_cells

        # Initialize the remainder of residents
        res_remainder = res_n - (init_res_size * max_cells)

        if res_remainder < 0:
            res_remainder = 0

        # Initialize the remainder of staff
        staff_remainder =  staff_n - (init_staff_size * max_cells)

        if staff_remainder < 0:
            staff_remainder = 0

        # Divide the remainder equally among the first few cells
        for i in range(remainder):
            cells[i] = init_size + 1

        for i in range(res_remainder):
            res_cells[i] = init_res_size + 1

        for i in range(staff_remainder):
            staff_cells[i] = init_staff_size + 1

        # Divide the remaining n among the rest of the cells
        for i in range(remainder, max_cells):
            cells[i] = init_size

        for i in range(res_remainder, max_cells):
            if sum(res_cells) + init_res_size < res_n:
                res_cells[i] = init_res_size
            else:
                res_cells[i] = res_n - sum(res_cells)

        for i in range(staff_remainder, max_cells):
            if sum(staff_cells) + init_staff_size < staff_n:
                staff_cells[i] = init_staff_size
            else:
                staff_cells[i] = staff_n - sum(staff_cells)

        return cells, staff_cells, res_cells

    def get_min_max_school_sizes(self, schools):
        max_classroom_size = 0
        min_classroom_size = 0
        max_nts_size = 0 # non teaching staff
        min_nts_size = 0

        for school in schools:
            nts = school["non_teaching_staff_uids"]

            if min_nts_size == 0:
                min_nts_size = len(nts)

            if max_nts_size == 0:
                max_nts_size = len(nts)

            if len(nts) < min_nts_size:
                min_nts_size = len(nts)

            if len(nts) > max_nts_size:
                max_nts_size = len(nts)

            classrooms = school["classrooms"]

            for classroom in classrooms:
                num_members = len(classroom["student_uids"]) + len(classroom["member_uids"])

                if min_classroom_size == 0:
                    min_classroom_size = num_members

                if max_classroom_size == 0:
                    max_classroom_size = num_members

                if num_members < min_classroom_size:
                    min_classroom_size = num_members

                if num_members > max_classroom_size:
                    max_classroom_size = num_members

        return min_nts_size, max_nts_size, min_classroom_size, max_classroom_size