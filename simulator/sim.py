import json
import numpy as np
import math

# return households which is a dict of dicts, where the key is the household id, and the values are:
# member_uids representing the residents in the household, reference_uid representing the reference person, and reference_age the reference person age
# cell splitting is not required in this case, the household is the cell
def convert_households(households_original):
    global agents
    global cellindex
    global cells

    households = {}
    householdscells = {}

    for household in households_original:
        member_uids = household["member_uids"]
        hhid = household["hhid"]
        ref_uid = household["reference_uid"]
        ref_age = household["reference_age"]

        households[hhid] = { "member_uids": member_uids, "reference_uid": ref_uid, "reference_age": ref_age}

        cells[cellindex] = { "type": "household", "place": households[hhid]}
        householdscells[cellindex] = cells[cellindex]

        cellindex += 1

        for uid in member_uids:
            agent = agents[uid]
            agent["res_cellid"] = hhid
            agent["curr_cellid"] = hhid

    return households, householdscells

# return industries_by_indid which is a dict of dict of dict of dict with the below format:
# industries_by_indid (indid) -> workplaces_by_wpid (wpid) -> cells_by_cellid (cellid) -> dict with member_uids key/value pair
# workplaces are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
# cells are split by an algorithm that ensures that cell sizes are balanced; at the same time as close to max_members as possible
def split_workplaces_by_cellsize(workplaces, cellsize, cellsizespare):
    global cellindex
    global cells

    industries_by_indid = {}
    workplacescells = {}

    for workplace in workplaces:
        workplaces_by_wpid = {}
        cells_by_cellid = {}

        employees = workplace["member_uids"]
        wpid = workplace["wpid"]
        indid = workplace["indid"]

        num_employees = len(employees)

        # If the number of members is less than or equal to "cellsize", no splitting needed
        if len(employees) <= cellsize:
            cells_by_cellid[cellindex] = { "member_uids": employees}

            cells[cellindex] = { "type": "workplace", "place": cells_by_cellid[cellindex]}
            workplacescells[cellindex] = cells[cellindex]

            cellindex += 1
        else:
            max_members = int(cellsize * (1 - cellsizespare))

            # Calculate the number of groups needed, considering spare space
            cell_counts = split_balanced_cells_close_to_x(num_employees, max_members)
            
            for index, this_cell_count in enumerate(cell_counts):
                members_count = this_cell_count
                members_start = index * members_count
                members_end = members_start + members_count

                temp_members = employees[members_start:members_end]

                cells_by_cellid[cellindex] = { "member_uids" : temp_members }

                cells[cellindex] = { "type": "workplace", "place": cells_by_cellid[cellindex]}
                workplacescells[cellindex] = cells[cellindex]

                cellindex += 1
        
        # assign cells into wpid
        workplaces_by_wpid[wpid] = cells_by_cellid

        # assign workplaces into indid
        if indid in industries_by_indid:
            existing_workplaces_by_wpid = industries_by_indid[indid]
            for tempwpid, tempcells in workplaces_by_wpid.items():
                existing_workplaces_by_wpid[tempwpid] = tempcells
        else:
            industries_by_indid[indid] = workplaces_by_wpid

    return industries_by_indid, workplacescells

# return schools_by_type which is a dict of dict of dict of dict with the below format:
# schools_by_type (indid) -> schools_by_scid (wpid) -> cells_by_cellid (cellid) -> cellinfodict (clid, student_uids, teacher_uids, non_teaching_staff_uids)
# classrooms are assigned to cells as is
# non-teaching staff are split into cells of max size < max_members: cellsize * (1 - cellsizespare)
# cells are split by an algorithm that ensures that cell sizes are as balanced and as close to max_members as possible
def split_schools_by_cellsize(schools, cellsize, cellsizespare):
    global cellindex
    global cells

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

        # If the number of nonteachingstaff is less than or equal to "cellsize", no splitting needed
        if len(nonteachingstaff) <= cellsize:
            cells_by_cellid[cellindex] = { "clid":-1, "non_teaching_staff_uids" : nonteachingstaff }
            
            cells[cellindex] = { "type": "school", "place": cells_by_cellid[cellindex]}
            schoolscells[cellindex] = cells[cellindex]

            cellindex += 1
        else:
            max_members = int(cellsize * (1 - cellsizespare))

            # Calculate the number of groups needed, considering spare space

            cell_counts = split_balanced_cells_close_to_x(num_nonteachingstaff, max_members)
            
            for index, this_cell_count in enumerate(cell_counts):
                members_count = this_cell_count
                members_start = index * members_count
                members_end = members_start + members_count

                temp_nonteachingstaff = nonteachingstaff[members_start:members_end]

                cells_by_cellid[cellindex] = { "clid":-1, "non_teaching_staff_uids" : temp_nonteachingstaff }

                cells[cellindex] = { "type": "school", "place": cells_by_cellid[cellindex]}
                schoolscells[cellindex] = cells[cellindex]

                cellindex += 1

        for classroom in classrooms:
            clid = classroom["clid"]
            students = classroom["student_uids"]
            teachers = classroom["teacher_uids"]

            cells_by_cellid[cellindex] = { "clid":clid, "student_uids":students, "teacher_uids":teachers}

            cells[cellindex] = { "type": "classroom", "place": cells_by_cellid[cellindex]}
            classroomscells[cellindex] = cells[cellindex]

            cellindex += 1
            
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
def split_institutions_by_cellsize(institutions, cellsize, cellsizespare):
    global agents, cellindex, cells

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

        # If the number of members is less than or equal to "cellsize", no splitting needed
        if num_members <= cellsize:
            cells_by_cellid[cellindex] = { "resident_uids": residents, "staff_uids": staff}

            cells[cellindex] = { "type": "institution", "place": cells_by_cellid[cellindex]}
            institutionscells[cellindex] = cells[cellindex]

            cellindex += 1
        else:
            max_members = int(cellsize * (1 - cellsizespare))

            # Calculate the number of groups needed, considering spare space

            cell_counts, staff_cell_counts, res_cell_counts = split_balanced_cells_staff_residents_close_to_x(num_residents, num_staff, max_members, staff_resident_ratio)
            
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

                cells_by_cellid[cellindex] = { "resident_uids": temp_residents, "staff_uids": temp_staff }

                cells[cellindex] = { "type": "institution", "place": cells_by_cellid[cellindex]}
                institutionscells[cellindex] = cells[cellindex]

                cellindex += 1
        
        # assign cells into instid
        institutions_by_id[instid] = cells_by_cellid

        # assign workplaces into insttypeid
        if insttypeid in institutions_by_type:
            existing_institutions_by_id = institutions_by_type[insttypeid]
            for tempinstid, tempcells in institutions_by_id.items():
                existing_institutions_by_id[tempinstid] = tempcells
        else:
            institutions_by_type[insttypeid] = institutions_by_id

        for uid in residents:
            agent = agents[uid]
            agent["res_cellid"] = instid
            agent["curr_cellid"] = instid

    return institutions_by_type, institutionscells

# takes "n" which is the number of agents to allocate, and "x" which is the max number of agents per cell
# returns "cells": array of int, whereby its length represents the num of cells, and each value, the num of agents to be assigned
def split_balanced_cells_close_to_x(n, x):
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
def split_balanced_cells_staff_residents_close_to_x(res_n, staff_n, x, staff_resident_ratio):
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

def get_min_max_school_sizes(schools):
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

# to establish set of static parameters such as cellsize, but grouped in dict

params = { "cellsize": 100,
           "cellsizespare": 0.1,
           "loadagents": True,
           "loadhouseholds": True,
           "loadworkplaces": True,
           "loadschools": True,
           "loadinstitutions": True 
         }

cellindex = 0
cells = {}

# load agents and all relevant JSON files on each node

if params["loadagents"]:
    agentsfile = open("./population/agents.json")
    agents = json.load(agentsfile)

    temp_agents = {int(k): v for k, v in agents.items()}

    agents = temp_agents

    temp_agents = None

    n = len(agents)

    maleagents = {k:v for k, v in agents.items() if v["gender"] == 0}
    femaleagents = {k:v for k, v in agents.items() if v["gender"] == 1}

    agentsbyages = {}
    malesbyages = {}
    femalesbyages = {}

    for age in range(101):
        agentsbyage = {k:v for k, v in agents.items() if v["age"] == age}
        malesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 0}
        femalesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 1}

        agentsbyages[age] = agentsbyage
        malesbyages[age] = malesbyage
        femalesbyages[age] = femalesbyage

    agentsemployed = {k:v for k, v in agents.items() if v["empstatus"] == 0}
    malesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 0}
    femalesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 1}

    agentsbyindustries = {}
    malesbyindustries = {}
    femalesbyindustries = {}

    for industry in range(1,22):
        agentsbyindustry = {k:v for k, v in agentsemployed.items() if v["empind"] == industry}
        malesbyindustry = {k:v for k, v in malesemployed.items() if v["empind"] == industry}
        femalesbyindustry = {k:v for k, v in femalesemployed.items() if v["empind"] == industry}

        agentsbyindustries[industry] = agentsbyindustry
        malesbyindustries[industry] = malesbyindustry
        femalesbyindustries[industry] = femalesbyindustry

if params["loadhouseholds"]:
    householdsfile = open("./population/households.json")
    households_original = json.load(householdsfile)

    households, cells_households = convert_households(households_original)

if params["loadworkplaces"]:
    workplacesfile = open("./population/workplaces.json")
    workplaces = json.load(workplacesfile)

    # handle cell splitting (on workplaces, schools institutions)
    industries, cells_industries = split_workplaces_by_cellsize(workplaces, params["cellsize"], params["cellsizespare"])

if params["loadschools"]:
    schoolsfile = open("./population/schools.json")
    schools = json.load(schoolsfile)

    min_nts_size, max_nts_size, min_classroom_size, max_classroom_size = get_min_max_school_sizes(schools)

    print("Min classroom size: " + str(min_classroom_size) + ", Max classroom size: " + str(max_classroom_size))
    print("Min non-teaching staff size: " + str(min_nts_size) + ", Max classroom size: " + str(max_nts_size))

    schooltypes, cells_schools, cells_classrooms = split_schools_by_cellsize(schools, params["cellsize"], params["cellsizespare"])

if params["loadinstitutions"]:
    institutiontypesfile = open("./population/institutiontypes.json")
    institutiontypes_original = json.load(institutiontypesfile)

    institutionsfile = open("./population/institutions.json")
    institutions = json.load(institutionsfile)

    institutiontypes, cells_institutions = split_institutions_by_cellsize(institutions, params["cellsize"], params["cellsizespare"])

    print(len(institutiontypes))
