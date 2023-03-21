import numpy as np
import math

class Cells:
    def __init__(self, agents, cells, cellindex):
        self.agents = agents
        self.cells = cells
        self.cellindex = cellindex

    # return households which is a dict of dicts, where the key is the household id, and the values are:
    # member_uids representing the residents in the household, reference_uid representing the reference person, and reference_age the reference person age
    # cell splitting is not required in this case, the household is the cell
    def convert_households(self, households_original):
        households = {}
        householdscells = {}

        for household in households_original:
            member_uids = household["member_uids"]
            hhid = household["hhid"]
            ref_uid = household["reference_uid"]
            ref_age = household["reference_age"]

            households[hhid] = { "member_uids": np.array(member_uids), "reference_uid": ref_uid, "reference_age": ref_age}

            self.cells[self.cellindex] = { "type": "household", "place": households[hhid]}
            householdscells[self.cellindex] = self.cells[self.cellindex]

            self.cellindex += 1

            if len(self.agents) > 0:
                for uid in member_uids:
                    agent = self.agents[uid]
                    agent["res_cellid"] = hhid
                    agent["curr_cellid"] = hhid

        return households, householdscells

    # return industries_by_indid which is a dict of dict of dict of dict with the below format:
    # industries_by_indid (indid) -> workplaces_by_wpid (wpid) -> cells_by_cellid (cellid) -> dict with member_uids key/value pair
    # workplaces are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
    # cells are split by an algorithm that ensures that cell sizes are balanced; at the same time as close to max_members as possible
    def split_workplaces_by_cellsize(self, workplaces, accommodations, cellsize, cellsizespare):
        industries_by_indid = {}
        workplacescells = {}
        accommodationcells = {}

        for workplace in workplaces:
            workplaces_by_wpid = {}
            cells_by_cellid = {}

            employees = workplace["member_uids"]
            wpid = workplace["wpid"]
            indid = workplace["indid"]
            accomid = workplace["accomid"] if workplace["accomid"] is not None else -1 # -1 if not an accommodation
            accomtypeid = workplace["accomtypeid"] if workplace["accomtypeid"] is not None else -1 # -1 if not an accommodation

            num_employees = len(employees)

            max_members = int(cellsize * (1 - cellsizespare))

            # If the number of members is less than or equal to "max_members", no splitting needed
            if len(employees) <= max_members:
                if accomid == -1:
                    cells_by_cellid[self.cellindex] = { "member_uids": np.array(employees)}
                else:
                    cells_by_cellid[self.cellindex] = { "member_uids": np.array(employees), "accomid": accomid, "accomtypeid": accomtypeid}

                self.cells[self.cellindex] = { "type": "workplace", "place": cells_by_cellid[self.cellindex]}
                workplacescells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in employees:
                        agent = self.agents[uid]
                        agent["work_cellid"] = self.cellindex   
            else:
                # Calculate the number of groups needed, considering spare space
                cell_counts = self.split_balanced_cells_close_to_x(num_employees, max_members)

                employees = np.array(employees)
                # np.random.shuffle(employees)
                
                for index, this_cell_count in enumerate(cell_counts):
                    members_count = this_cell_count
                    members_start = index * members_count
                    members_end = members_start + members_count

                    temp_members = employees[members_start:members_end]

                    if accomid == -1:
                        cells_by_cellid[self.cellindex] = { "member_uids": np.array(temp_members)}
                    else:
                        cells_by_cellid[self.cellindex] = { "member_uids": np.array(temp_members), "accomid": accomid, "accomtypeid": accomtypeid}

                    self.cells[self.cellindex] = { "type": "workplace", "place": cells_by_cellid[self.cellindex]}
                    workplacescells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in temp_members:
                            agent = self.agents[uid]
                            agent["work_cellid"] = self.cellindex
                    
            self.cellindex += 1

            if accomid > 0:
                roomsbysizes = accommodations[accomtypeid][accomid]
                roomsize = roomsbysizes["roomsize"]
                roomids = roomsbysizes["member_uids"]

                for roomid in roomids: 
                    cells_by_cellid[self.cellindex] = { "member_uids": [], "accomid": accomid, "accomtypeid": accomtypeid, "roomid": roomid, "roomsize": roomsize} # member_uids here represents tourists that are not assigned yet

                    self.cells[self.cellindex] = { "type": "room", "place": cells_by_cellid[self.cellindex]}

                    workplacescells[self.cellindex] = self.cells[self.cellindex]
                    accommodationcells[self.cellindex] = self.cells[self.cellindex]

                    self.cellindex += 1         
            
            # assign cells into wpid
            workplaces_by_wpid[wpid] = cells_by_cellid

            # assign workplaces into indid
            if indid in industries_by_indid:
                existing_workplaces_by_indid = industries_by_indid[indid]
                for tempwpid, tempcells in workplaces_by_wpid.items():
                    existing_workplaces_by_indid[tempwpid] = tempcells
            else:
                industries_by_indid[indid] = workplaces_by_wpid

        return industries_by_indid, workplacescells, accommodationcells

    # return schools_by_type which is a dict of dict of dict of dict with the below format:
    # schools_by_type (indid) -> schools_by_scid (wpid) -> cells_by_cellid (cellid) -> cellinfodict (clid, student_uids, teacher_uids, non_teaching_staff_uids)
    # classrooms are assigned to cells as is
    # non-teaching staff are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
    # cells are split by an algorithm that ensures that cell sizes are as balanced and as close to max_members as possible
    def split_schools_by_cellsize(self, schools, cellsize, cellsizespare):
        schools_by_type = {}
        schoolscells = {}
        classroomscells = {}

        for school in schools:
            schools_by_scid = {}
            cells_by_cellid = {}

            scid = school["scid"]
            sctype = school["sc_type"]

            nonteachingstaff = school["non_teaching_staff_uids"] # consider separately from classrooms
            classrooms = school["classrooms"]

            num_nonteachingstaff = len(nonteachingstaff)

            max_members = int(cellsize * (1 - cellsizespare))

            # If the number of nonteachingstaff is less than or equal to "max_members", no splitting needed
            if len(nonteachingstaff) <= max_members:
                cells_by_cellid[self.cellindex] = { "clid":-1, "non_teaching_staff_uids" : np.array(nonteachingstaff) }
                
                self.cells[self.cellindex] = { "type": "school", "place": cells_by_cellid[self.cellindex]}
                schoolscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in nonteachingstaff:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex

                self.cellindex += 1
            else:
                # Calculate the number of groups needed, considering spare space

                cell_counts = self.split_balanced_cells_close_to_x(num_nonteachingstaff, max_members)

                nonteachingstaff = np.array(nonteachingstaff)
                # np.random.shuffle(nonteachingstaff)
                
                for index, this_cell_count in enumerate(cell_counts):
                    members_count = this_cell_count
                    members_start = index * members_count
                    members_end = members_start + members_count

                    temp_nonteachingstaff = nonteachingstaff[members_start:members_end]

                    cells_by_cellid[self.cellindex] = { "clid":-1, "non_teaching_staff_uids" : temp_nonteachingstaff }

                    self.cells[self.cellindex] = { "type": "school", "place": cells_by_cellid[self.cellindex]}
                    schoolscells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in temp_nonteachingstaff:
                            agent = self.agents[uid]
                            agent["school_cellid"] = self.cellindex

                    self.cellindex += 1

            for classroom in classrooms:
                clid = classroom["clid"]
                students = classroom["student_uids"]
                teachers = classroom["teacher_uids"]

                cells_by_cellid[self.cellindex] = { "clid":clid, "student_uids":np.array(students), "teacher_uids":np.array(teachers)}

                self.cells[self.cellindex] = { "type": "classroom", "place": cells_by_cellid[self.cellindex]}
                classroomscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in students:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex

                    for uid in teachers:
                        agent = self.agents[uid]
                        agent["school_cellid"] = self.cellindex

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
        institutions_by_type = {}
        institutionscells = {}

        for institution in institutions:
            institutions_by_id = {}
            cells_by_cellid = {}

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
                cells_by_cellid[self.cellindex] = { "resident_uids": np.array(residents), "staff_uids": np.array(staff)}

                self.cells[self.cellindex] = { "type": "institution", "place": cells_by_cellid[self.cellindex]}
                institutionscells[self.cellindex] = self.cells[self.cellindex]

                if len(self.agents) > 0:
                    for uid in residents:
                        agent = self.agents[uid]
                        agent["inst_cellid"] = self.cellindex
                        agent["res_cellid"] = self.cellindex
                        agent["curr_cellid"] = self.cellindex

                    for uid in staff:
                        agent = self.agents[uid]
                        agent["inst_cellid"] = self.cellindex

                self.cellindex += 1
            else:
                # Calculate the number of groups needed, considering spare space

                cell_counts, staff_cell_counts, res_cell_counts = self.split_balanced_cells_staff_residents_close_to_x(num_residents, num_staff, max_members, staff_resident_ratio)
                
                staff = np.array(staff)
                residents = np.array(residents)

                # np.random.shuffle(staff)
                # np.random.shuffle(residents)
                
                for index, this_cell_count in enumerate(cell_counts):

                    # first assign staff
                    staff_count = staff_cell_counts[index]
                    staff_start = index * staff_count
                    staff_end = staff_start + staff_count

                    temp_staff = staff[staff_start:staff_end]

                    # then assign residents
                    resident_count = res_cell_counts[index]
                    resident_start = index * resident_count
                    resident_end = resident_start + resident_count

                    temp_residents = residents[resident_start:resident_end]

                    cells_by_cellid[self.cellindex] = { "resident_uids": temp_residents, "staff_uids": temp_staff }

                    self.cells[self.cellindex] = { "type": "institution", "place": cells_by_cellid[self.cellindex]}
                    institutionscells[self.cellindex] = self.cells[self.cellindex]

                    if len(self.agents) > 0:
                        for uid in temp_residents:
                            agent = self.agents[uid]
                            agent["inst_cellid"] = self.cellindex
                            agent["res_cellid"] = self.cellindex
                            agent["curr_cellid"] = self.cellindex

                        for uid in temp_staff:
                            agent = self.agents[uid]
                            agent["inst_cellid"] = self.cellindex

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