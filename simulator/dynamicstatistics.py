from epidemiologyclasses import SEIRState
from vars import Vars
import time

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
        self.average_contacts_per_person = 0

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

        # self.new_tests, self.new_vaccinations, self.new_quarantined, self.new_hospitalised = intervention_totals

        self.total_tests += self.new_tests
        self.total_vaccinations += self.new_vaccinations
        self.total_quarantined += self.new_quarantined
        self.total_hospitalised += self.new_hospitalised
        self.total_contacttraced += self.new_contacttraced

        self.infectious_rate = n_infectious / self.total_active_population
        self.recovery_rate = n_recovered / self.total_active_population
        self.mortality_rate = n_deceased / self.total_active_population

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
        df.loc[day, "infectious_rate"] = round(self.infectious_rate, 2)
        df.loc[day, "recovery_rate"] = round(self.recovery_rate, 2)
        df.loc[day, "mortality_rate"] = round(self.mortality_rate, 2)
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
        df.loc[day, "average_contacts_per_person"] = round(self.average_contacts_per_person, 2)

        return df