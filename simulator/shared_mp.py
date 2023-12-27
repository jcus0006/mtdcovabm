agents_static = None

# def init_pool_processes(the_agents_static):

#     '''Initialize each process with a global agents_static class instance included read-only in multiprocessing array format
#     '''
#     global agents_static
#     agents_static = the_agents_static

def init_pool_processes(params):
    '''Initialize each process with a global agents_static class instance included read-only in multiprocessing array format
    '''
    global agents_ids_by_ages
    global timestepmins
    global n_locals
    global n_tourists
    global locals_ratio_to_full_pop
    global itineraryparams
    global epidemiologyparams
    global cells_industries_by_indid_by_wpid
    global cells_restaurants
    global cells_hospital
    global cells_testinghub
    global cells_vaccinationhub
    global cells_entertainment_by_activityid
    global cells_religious
    global cells_households
    global cells_breakfast_by_accomid
    global cells_airport
    global cells_transport
    global cells_institutions
    global cells_accommodation
    global agents_static

    global contactnetworkparams
    global cells_type
    global indids_by_cellid

    a_static, a_ids_by_ages, t_mins, n_locs, n_tours, l_ratio, it_params, epi_params, c_ind_by_indid_by_wpid, c_restaurants, c_hospital, c_testinghub, c_vaccinationhub, c_ent_by_act_id, c_religious, c_households, c_breakfastfast_by_acc_id, c_airport, c_transport, c_institutions, c_accom, cn_params, c_type, i_ids_by_cellid = params

    agents_ids_by_ages = a_ids_by_ages
    timestepmins = t_mins
    n_locals = n_locs
    n_tourists = n_tours
    locals_ratio_to_full_pop = l_ratio
    itineraryparams = it_params
    epidemiologyparams = epi_params
    cells_industries_by_indid_by_wpid = c_ind_by_indid_by_wpid
    cells_restaurants = c_restaurants
    cells_hospital = c_hospital
    cells_testinghub = c_testinghub
    cells_vaccinationhub = c_vaccinationhub
    cells_entertainment_by_activityid = c_ent_by_act_id
    cells_religious = c_religious
    cells_households = c_households
    cells_breakfast_by_accomid = c_breakfastfast_by_acc_id
    cells_airport = c_airport
    cells_transport = c_transport
    cells_institutions = c_institutions
    cells_accommodation = c_accom
    agents_static = a_static
    contactnetworkparams = cn_params
    cells_type = c_type
    indids_by_cellid = i_ids_by_cellid

def init_pool_processes_dask_mp(worker):

    '''Initialize each process with shared read-only static data
    '''
    global agents_ids_by_ages
    global timestepmins
    global n_locals
    global n_tourists
    global locals_ratio_to_full_pop
    global itineraryparams
    global epidemiologyparams
    global cells_industries_by_indid_by_wpid
    global cells_restaurants
    global cells_hospital
    global cells_testinghub
    global cells_vaccinationhub
    global cells_entertainment_by_activityid
    global cells_religious
    global cells_households
    global cells_breakfast_by_accomid
    global cells_airport
    global cells_transport
    global cells_institutions
    global cells_accommodation
    global agents_static

    global contactnetworkparams
    global cells_type
    global indids_by_cellid

    agents_ids_by_ages = worker.data["agents_ids_by_ages"]
    timestepmins = worker.data["timestepmins"]
    n_locals = worker.data["n_locals"]
    n_tourists = worker.data["n_tourists"]
    locals_ratio_to_full_pop = worker.data["locals_ratio_to_full_pop"]

    itineraryparams = worker.data["itineraryparams"]
    epidemiologyparams = worker.data["epidemiologyparams"]
    cells_industries_by_indid_by_wpid = worker.data["cells_industries_by_indid_by_wpid"] 
    cells_restaurants = worker.data["cells_restaurants"] 
    cells_hospital = worker.data["cells_hospital"] 
    cells_testinghub = worker.data["cells_testinghub"] 
    cells_vaccinationhub = worker.data["cells_vaccinationhub"] 
    cells_entertainment_by_activityid = worker.data["cells_entertainment_by_activityid"] 
    cells_religious = worker.data["cells_religious"] 
    cells_households = worker.data["cells_households"] 
    cells_breakfast_by_accomid = worker.data["cells_breakfast_by_accomid"] 
    cells_airport = worker.data["cells_airport"] 
    cells_transport = worker.data["cells_transport"] 
    cells_institutions = worker.data["cells_institutions"] 
    cells_accommodation = worker.data["cells_accommodation"] 
    agents_static = worker.data["agents_static"]

    contactnetworkparams = worker.data["contactnetworkparams"]
    cells_type = worker.data["cells_type"]
    indids_by_cellid = worker.data["indids_by_cellid"]