import numpy as np
from simulator import util, seirstateutil
from simulator.epidemiology import SEIRState
import multiprocessing as mp

def initialize_agents(agents, agents_ids_by_ages, agents_ids_by_agebrackets, tourists, params, itineraryparams, powerlaw_distribution_parameters, sociability_rate_min, sociability_rate_max, initial_seir_state_distribution, figure_count, n_locals, age_brackets, age_brackets_workingages):
    temp_agents = {int(k): {"age": v["age"], 
                            "edu": v["edu"],
                            "sc_student": v["sc_student"], 
                            "empstatus": v["empstatus"], 
                            "empind": v["empind"], 
                            "empftpt": v["empftpt"],
                            "hhid": v["hhid"],
                            "scid": v["scid"]} for k, v in agents.items()}

    agents_vaccination_doses = np.array([0 for i in range(n_locals)])

    locals_ratio_to_full_pop = n_locals / params["fullpop"]

    if params["loadtourism"]:
        largest_agent_id = sorted(list(temp_agents.keys()), reverse=True)[0]

        for i in range(len(tourists)):
            temp_agents[largest_agent_id+1] = {}
            largest_agent_id += 1

    agents_seir_state = np.array([SEIRState(0) for i in range(len(temp_agents))])

    # contactnetwork_sum_time_taken = 0
    # contactnetwork_util = contactnetwork.ContactNetwork(n_locals, n_tourists, locals_ratio_to_full_pop, agents, agents_seir_state, agents_seir_state_transition_for_day, agents_infection_type, agents_infection_severity, agents_vaccination_doses, cells, cells_agents_timesteps, contactnetworkparams, epidemiologyparams, contactnetwork_sum_time_taken, False, False, params["numprocesses"])
    # epi_util = contactnetwork_util.epi_util

    for index, (agent_uid, agent) in enumerate(temp_agents.items()):
        if index < n_locals: # ignore tourists for now
            # agent["curr_cellid"] = -1
            agent["res_cellid"] = -1
            agent["work_cellid"] = -1
            agent["school_cellid"] = -1
            agent["inst_cellid"] = -1
            # agent["symptomatic"] = False
            agent["tourist_id"] = None 
            agent["state_transition_by_day"] = []
            # intervention_events_by_day
            agent["test_day"] = [] # [day, timestep]
            agent["test_result_day"] = [] # [day, timestep]
            agent["quarantine_days"] = [] # [[[startday, timestep], [endday, timestep]]] -> [startday, timestep, endday]
            agent["vaccination_days"] = [] # [[day, timestep]]
            agent["hospitalisation_days"] = [] # [[startday, timestep], [endday, timestep]] -> [startday, timestep, endday]

            if agent["empstatus"] == 0:
                agent["working_schedule"] = {}

            agent, age, agents_ids_by_ages, agents_ids_by_agebrackets = util.set_age_brackets(agent, agents_ids_by_ages, agent_uid, age_brackets, age_brackets_workingages, agents_ids_by_agebrackets)

            agent["epi_age_bracket_index"] = util.get_sus_mort_prog_age_bracket_index(age)

            agent = util.set_public_transport_regular(agent, itineraryparams["public_transport_usage_probability"][0])
        else:
            break
    
        # agent["soc_rate"] = np.random.choice(sociability_rate_options, size=1, p=sociability_rate_distribution)[0]

    temp_agents = util.generate_sociability_rate_powerlaw_dist(temp_agents, agents_ids_by_agebrackets, powerlaw_distribution_parameters, params, sociability_rate_min, sociability_rate_max, figure_count)

    agents_seir_state = seirstateutil.initialize_agent_states(n_locals, initial_seir_state_distribution, agents_seir_state)

    agents = temp_agents

    temp_agents = None

    return agents, agents_seir_state, agents_vaccination_doses, locals_ratio_to_full_pop, figure_count

def initialize_shared_agents_dict_ct(manager, agents):
    shared_agents = manager.dict()

    for agentid, props in agents.items():
        shared_agents[agentid] = manager.dict()
        shared_agents[agentid]["propname"] = props["propname"]

    return shared_agents

def initialize_agents_dict_ct(manager, agents):
    temp_agents = {}

    # for agentid, props in agents.items():
    #     temp_agents[agentid] = {}
        
    #     temp_agents[agentid]["res_cellid"] = props["res_cellid"]
    #     temp_agents[agentid]["test_day"] = props["test_day"]
    #     temp_agents[agentid]["test_result_day"] = props["test_result_day"]
    #     temp_agents[agentid]["quarantine_days"] = props["quarantine_days"]        

    temp_agents = {agentid: {"res_cellid": props["res_cellid"],
                            "test_day": props["test_day"],
                            "test_result_day": props["test_result_day"],
                            "quarantine_days": props["quarantine_days"]} for agentid, props in agents.items() if len(props) > 0}

    return temp_agents