from epidemiologyclasses import SEIRState
from vars import Vars
import time

class DynamicStatistics:
    def __init__(self, n_locals, n_tourists, n_tourists_initial, vars_util: Vars):
        self.n_locals = n_locals # ok
        self.n_tourists = n_tourists # ok
        self.n_tourists_initial = n_tourists_initial
        self.vars_util = vars_util # ok
        self.tourists_active_ids = [] # ok
        self.total_active_tourists = 0 # ok
        self.total_arriving_tourists = 0 # ok
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
        self.total_hospitalized = 0
        self.total_to_be_vaccinated = 0
        self.new_vaccinations = 0
        self.new_tests = 0
        self.new_contacttraced = 0
        self.new_quarantined = 0
        self.new_hospitalized = 0
        self.average_contacts_per_person = 0

    def refresh_rates(self, day, arr_tourists, dep_tourists, tourists_active_ids): # optimised
        start = time.time()

        self.tourists_active_ids = tourists_active_ids

        # deceased
        prev_deceased = self.total_deceased

        n_deceased = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Deceased])
        n_deceased += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Deceased])
        
        if day == 1:
            self.new_deaths = n_deceased
        else:
            self.new_deaths = n_deceased - prev_deceased 
        
        self.total_deceased = n_deceased
        
        # exposed
        prev_exposed = self.total_exposed

        n_exposed = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Exposed])
        n_exposed += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Exposed])

        if day == 1:
            self.new_exposed = n_exposed
        else:
            self.new_exposed = n_exposed - prev_exposed
        
        self.total_exposed = n_exposed

        # susceptible
        prev_susceptible = self.total_susceptible

        n_susceptible = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Susceptible])
        n_susceptible += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Susceptible])

        if day == 1:
            self.new_susceptible = n_susceptible
        else:
            self.new_susceptible = n_susceptible - prev_susceptible
        
        self.total_susceptible = n_susceptible

        # infectious
        prev_infectious = self.total_infectious

        n_infectious = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Infectious])
        n_infectious += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Infectious])
        
        if day == 1:
            self.new_infectious = n_infectious
        else:
            self.new_infectious = n_infectious - prev_infectious
        
        self.total_infectious = n_infectious

        prev_recovered = self.total_recovered

        n_recovered = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Recovered])
        n_recovered += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Recovered])

        if day == 1:
            self.new_recovered = n_recovered
        else:
            self.new_recovered = n_recovered - prev_recovered
        
        self.total_recovered = n_recovered

        time_taken = time.time() - start
        print("refresh infectious rate (compr) time taken: " + str(time_taken))

        self.total_arriving_tourists = arr_tourists 
        self.total_departing_tourists = dep_tourists
        self.total_active_tourists = len(self.tourists_active_ids)
        
        self.total_active_population = self.n_locals + self.total_active_tourists - n_deceased

        self.infectious_rate = n_infectious / self.total_active_population
        self.recovery_rate = n_recovered / self.total_active_population
        self.mortality_rate = n_deceased / self.total_active_population

    def update_statistics_df(self, day, df):
        df.loc[day, "total_active_population"] = self.total_active_population
        df.loc[day, "total_locals"] = self.n_locals
        df.loc[day, "total_active_tourists"] = self.total_active_tourists
        df.loc[day, "total_arriving_tourists"] = self.total_arriving_tourists
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
        df.loc[day, "total_hospitalized"] = self.total_hospitalized
        df.loc[day, "total_to_be_vaccinated"] = self.total_to_be_vaccinated
        df.loc[day, "new_vaccinations"] = self.new_vaccinations
        df.loc[day, "new_tests"] = self.new_tests
        df.loc[day, "new_contacttraced"] = self.new_contacttraced
        df.loc[day, "new_quarantined"] = self.new_quarantined
        df.loc[day, "new_hospitalized"] = self.new_hospitalized
        df.loc[day, "average_contacts_per_person"] = round(self.average_contacts_per_person, 2)

        return df