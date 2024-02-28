from epidemiologyclasses import SEIRState
from vars import Vars
import time
import numpy as np

class DynamicStatistics:
    def __init__(self, n_locals, n_tourists, n_tourists_initial):
        self.n_locals = n_locals # ok
        self.n_tourists = n_tourists # ok
        self.n_tourists_initial = n_tourists_initial # ok
        self.total_active_tourists = 0 # ok
        self.total_arriving_tourists = 0 # ok
        self.total_arriving_nextday_tourists = 0 # ok
        self.total_departing_tourists = 0 # ok
        self.total_active_population = self.n_locals + n_tourists_initial
        self.total_exposed = 0 # ok
        self.total_susceptible = 0 # ok
        self.total_infectious = 0 # ok
        self.total_recovered = 0 # ok
        self.total_deceased = 0 # ok
        self.new_exposed = 0 # ok
        self.new_susceptible = 0 # ok
        self.new_infectious = 0 # ok
        self.new_recovered = 0 # ok
        self.new_deaths = 0 # ok
        self.num_direct_contacts = 0 # ok
        self.avg_direct_contacts_per_agent = 0 # ok
        self.infectious_rate = 0 # ok
        self.recovery_rate = 0 # ok
        self.mortality_rate = 0 # ok
        self.basic_reproduction_number = 1
        self.effective_reproduction_number = 1
        self.total_vaccinations = 0
        self.total_tests = 0
        self.total_contacttraced = 0
        self.total_quarantined = 0
        self.total_hospitalised = 0
        self.new_vaccinations = 0
        self.new_tests = 0
        self.new_contacttraced = 0
        self.new_quarantined = 0
        self.new_hospitalised = 0

        self.num_locals_infected_once = 0
        self.num_locals_infected_at_least_once = 0
        self.num_locals_infected_more_than_once = 0
        self.num_locals_infected_never = 0
        self.num_locals_max_infected = 0
        self.avg_locals_infections = 0
        self.med_locals_infections = 0
        self.std_locals_infections = 0

        self.locals_infected_counts = {}
        self.current_locals_infected = set()
        self.prev_locals_infected = set()

        self.init_stage = False

    def refresh_rates(self, day, act_tourists, arr_tourists, arr_nextday_tourists, dep_tourists, vars_util, seir_states=None): # optimised
        start = time.time()
        
        prev_deceased = self.total_deceased
        prev_exposed = self.total_exposed
        prev_susceptible = self.total_susceptible
        prev_infectious = self.total_infectious
        prev_recovered = self.total_recovered

        if seir_states is None:
            seir_states = self.calculate_seir_states_counts(vars_util)
        else:
            tourists_seir_states = self.calculate_seir_states_counts(vars_util, False, True)
            seir_states = self.combine_local_and_tourists_seir_state(seir_states, tourists_seir_states)

        n_deceased, n_exposed, n_susceptible, n_infectious, n_recovered = seir_states
        
        if day == 1:
            self.new_deaths = n_deceased
        else:
            self.new_deaths = n_deceased - prev_deceased 
        
        self.total_deceased = n_deceased
        
        # exposed

        if day == 1:
            self.new_exposed = n_exposed
        else:
            self.new_exposed = n_exposed - prev_exposed
        
        self.total_exposed = n_exposed

        # susceptible
    
        if day == 1:
            self.new_susceptible = n_susceptible
        else:
            self.new_susceptible = n_susceptible - prev_susceptible
        
        self.total_susceptible = n_susceptible

        # infectious

        if day == 1:
            self.new_infectious = n_infectious
        else:
            self.new_infectious = n_infectious - prev_infectious
        
        self.total_infectious = n_infectious

        # recovered

        if day == 1:
            self.new_recovered = n_recovered
        else:
            self.new_recovered = n_recovered - prev_recovered
        
        self.total_recovered = n_recovered

        time_taken = time.time() - start
        print("refresh infectious rate (compr) time taken: " + str(time_taken))

        self.total_arriving_tourists = arr_tourists 
        self.total_arriving_nextday_tourists = arr_nextday_tourists
        self.total_departing_tourists = dep_tourists
        self.total_active_tourists = act_tourists
        
        self.total_active_population = self.n_locals + self.total_active_tourists - n_deceased

        if self.num_direct_contacts > 0:
            self.avg_direct_contacts_per_agent = round(self.num_direct_contacts / self.total_active_population, 2)

        # self.new_tests, self.new_vaccinations, self.new_quarantined, self.new_hospitalised = intervention_totals

        self.total_tests += self.new_tests
        self.total_vaccinations += self.new_vaccinations
        self.total_quarantined += self.new_quarantined
        self.total_hospitalised += self.new_hospitalised
        self.total_contacttraced += self.new_contacttraced

        self.infectious_rate = n_infectious / self.total_active_population
        self.recovery_rate = n_recovered / self.total_active_population
        self.mortality_rate = n_deceased / self.total_active_population

        # local infected counts
        if not self.init_stage:
            new_infected = self.current_locals_infected - self.prev_locals_infected # set difference
            for id in new_infected:
                if id not in self.locals_infected_counts:
                    self.locals_infected_counts[id] = 1
                else:
                    self.locals_infected_counts[id] += 1
            
            self.num_locals_infected_at_least_once = len(self.locals_infected_counts)
            self.num_locals_infected_never = self.n_locals - self.num_locals_infected_at_least_once 

            locals_infected_counts = np.array(list(self.locals_infected_counts.values()) + [0 for _ in range(self.num_locals_infected_never)]) # represent zeros too
            self.num_locals_infected_once = np.sum(locals_infected_counts == 1)
            self.num_locals_infected_more_than_once = np.sum(locals_infected_counts > 1)
            self.num_locals_max_infected = np.max(locals_infected_counts)
            self.avg_locals_infections = np.mean(locals_infected_counts)
            self.med_locals_infections = np.median(locals_infected_counts)
            self.std_locals_infections = np.std(locals_infected_counts)

            self.prev_locals_infected = self.current_locals_infected.copy()

        if self.init_stage: # this can only be once, so reset to False once it runs once
            self.init_stage = False

    def calculate_seir_states_counts(self, vars_util, ignore_tourists=False, tourists_only=False):
        n_deceased, n_exposed, n_susceptible, n_infectious, n_recovered = 0, 0, 0, 0, 0
        for id, state in vars_util.agents_seir_state.items():
            if (not tourists_only and (not ignore_tourists or id < self.n_locals)) or (tourists_only and id >= self.n_locals):
                match state:
                    case SEIRState.Deceased:
                        n_deceased += 1
                    case SEIRState.Exposed:
                        n_exposed += 1
                    case SEIRState.Susceptible:
                        n_susceptible += 1
                    case SEIRState.Infectious:
                        n_infectious += 1
                        if not self.init_stage and id < self.n_locals:
                            self.current_locals_infected.add(id)
                    case SEIRState.Recovered:
                        n_recovered += 1
            else:
                pass

        seir_states = n_deceased, n_exposed, n_susceptible, n_infectious, n_recovered
        return seir_states
    
    def combine_local_and_tourists_seir_state(self, local_seir_states, tourists_seir_states):
        n_deceased, n_exposed, n_susceptible, n_infectious, n_recovered = local_seir_states

        n_deceased += tourists_seir_states[0]
        n_exposed += tourists_seir_states[1]
        n_susceptible += tourists_seir_states[2]
        n_infectious += tourists_seir_states[3]
        n_recovered += tourists_seir_states[4]

        local_seir_states = n_deceased, n_exposed, n_susceptible, n_infectious, n_recovered

        return local_seir_states

    def update_statistics_df(self, day, df):
        df.loc[day, "total_active_population"] = self.total_active_population
        df.loc[day, "total_locals"] = self.n_locals
        df.loc[day, "total_active_tourists"] = self.total_active_tourists
        df.loc[day, "total_arriving_tourists"] = self.total_arriving_tourists
        df.loc[day, "total_arriving_nextday_tourists"] = self.total_arriving_nextday_tourists
        df.loc[day, "total_departing_tourists"] = self.total_departing_tourists
        df.loc[day, "total_exposed"] = self.total_exposed
        df.loc[day, "total_susceptible"] = self.total_susceptible
        df.loc[day, "total_infectious"] = self.total_infectious
        df.loc[day, "total_recovered"] = self.total_recovered
        df.loc[day, "total_deceased"] = self.total_deceased
        df.loc[day, "new_exposed"] = self.new_exposed
        df.loc[day, "new_susceptible"] = self.new_susceptible
        df.loc[day, "new_infectious"] = self.new_infectious
        df.loc[day, "new_recovered"] = self.new_recovered
        df.loc[day, "new_deaths"] = self.new_deaths
        df.loc[day, "num_direct_contacts"] = self.num_direct_contacts
        df.loc[day, "avg_direct_contacts_per_agent"] = self.avg_direct_contacts_per_agent
        df.loc[day, "infectious_rate"] = round(self.infectious_rate, 2)
        df.loc[day, "recovery_rate"] = round(self.recovery_rate, 2)
        df.loc[day, "mortality_rate"] = round(self.mortality_rate, 2)
        df.loc[day, "locals_inf_once"] = self.num_locals_infected_once
        df.loc[day, "locals_inf_at_least_once"] = self.num_locals_infected_at_least_once
        df.loc[day, "locals_inf_more_than_once"] = self.num_locals_infected_more_than_once
        df.loc[day, "locals_inf_never"] = self.num_locals_infected_never
        df.loc[day, "locals_max_inf"] = self.num_locals_max_infected
        df.loc[day, "locals_avg_inf_counts"] = round(self.avg_locals_infections, 2)
        df.loc[day, "locals_med_inf_counts"] = round(self.med_locals_infections, 2)
        df.loc[day, "locals_std_inf_counts"] = round(self.std_locals_infections, 2)
        df.loc[day, "basic_reproduction_number"] = round(self.basic_reproduction_number, 2)
        df.loc[day, "effective_reproduction_number"] = round(self.effective_reproduction_number, 2)
        df.loc[day, "total_vaccinations"] = self.total_vaccinations
        df.loc[day, "total_tests"] = self.total_tests
        df.loc[day, "total_contacttraced"] = self.total_contacttraced
        df.loc[day, "total_quarantined"] = self.total_quarantined
        df.loc[day, "total_hospitalised"] = self.total_hospitalised
        df.loc[day, "new_vaccinations"] = self.new_vaccinations
        df.loc[day, "new_tests"] = self.new_tests
        df.loc[day, "new_contacttraced"] = self.new_contacttraced
        df.loc[day, "new_quarantined"] = self.new_quarantined
        df.loc[day, "new_hospitalised"] = self.new_hospitalised

        return df