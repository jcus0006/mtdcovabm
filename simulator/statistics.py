from epidemiologyclasses import SEIRState
from vars import Vars
import time

class Statistics:
    def __init__(self, n_locals, n_tourists, vars_util: Vars):
        self.n_locals = n_locals # ok
        self.n_tourists = n_tourists # ok
        self.vars_util = vars_util # ok
        self.tourists_active_ids = [] # ok
        self.total_active_tourists = 0 # ok
        self.total_arriving_tourists = 0 # ok
        self.total_departing_tourists = 0 # ok
        self.total_active_population = self.n_locals + self.total_active_tourists # ok - assume 0 tourists at the beginning
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
        self.total_contactracted = 0
        self.total_quarantined = 0
        self.total_hospitalized = 0
        self.total_to_be_vaccinated = 0
        self.new_vaccinations = 0
        self.new_tests = 0
        self.new_contacttraced = 0
        self.new_quarantined = 0
        self.new_hospitalized = 0

    def refresh_rates(self, arr_tourists, dep_tourists, tourists_active_ids): # optimised
        start = time.time()

        self.total_arriving_tourists = arr_tourists
        self.total_departing_tourists = dep_tourists
        self.tourists_active_ids = tourists_active_ids

        # deceased
        prev_deceased = self.total_deceased

        n_deceased = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Deceased])
        n_deceased += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Deceased])
        
        self.new_deaths = n_deceased - prev_deceased 
        self.total_deceased = n_deceased
        
        # exposed
        prev_exposed = self.total_exposed

        n_exposed = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Exposed])
        n_exposed += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Exposed])

        self.new_exposed = n_exposed - prev_exposed
        self.total_exposed = n_exposed

        # susceptible
        prev_susceptible = self.total_susceptible

        n_susceptible = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Susceptible])
        n_susceptible += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Susceptible])

        self.new_susceptible = n_susceptible - prev_susceptible
        self.total_susceptible = n_susceptible

        # infectious
        prev_infectious = self.total_infectious

        n_infectious = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Infectious])
        n_infectious += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Infectious])
        
        self.new_infectious = n_infectious - prev_infectious
        self.total_infectious = n_infectious

        prev_recovered = self.total_recovered

        n_recovered = sum([1 for index, state in enumerate(self.vars_util.agents_seir_state) if index < self.n_locals and state == SEIRState.Recovered])
        n_recovered += sum([1 for tourist_id in self.tourists_active_ids if self.vars_util.agents_seir_state[self.n_locals + tourist_id] == SEIRState.Recovered])

        self.new_recovered = n_recovered - prev_recovered
        self.total_recovered = n_recovered

        time_taken = time.time() - start
        print("refresh infectious rate (compr) time taken: " + str(time_taken))

        n_total = self.n_locals + len(self.tourists_active_ids)

        self.total_active_population = n_total - n_deceased

        self.infectious_rate = n_infectious / self.total_active_population
        self.recovery_rate = n_recovered / self.total_active_population
        self.mortality_rate = n_deceased / self.total_active_population