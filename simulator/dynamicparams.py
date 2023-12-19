from simulator.statistics import Statistics

class DynamicParams:
    def __init__(self, n_locals, n_tourists, epidemiologyparams, vars_util):
        self.n_locals = n_locals
        self.n_tourists = n_tourists
        self.masks_hygiene_distancing_day_thresholds = epidemiologyparams["masks_hygiene_distancing_day_thresholds"]
        self.masks_hygiene_distancing_infectiousrate_thresholds = epidemiologyparams["masks_hygiene_distancing_infectiousrate_thresholds"]
        self.vaccination_day_thresholds = epidemiologyparams["vaccination_day_thresholds"]
        self.vaccination_infectiousrate_thresholds = epidemiologyparams["vaccination_infectiousrate_thresholds"]
        self.intervention_day_thresholds = epidemiologyparams["intervention_day_thresholds"]
        self.intervention_infectiousrate_thresholds = epidemiologyparams["intervention_infectiousrate_thresholds"]
        self.lockdown_infectiousrate_thresholds = epidemiologyparams["lockdown_infectiousrate_thresholds"]
        self.lockdown_day_thresholds = epidemiologyparams["lockdown_day_thresholds"]

        self.quarantine_enabled = False
        self.testing_enabled = False
        self.contact_tracing_enabled = False
        self.workplaces_lockdown = False
        self.schools_lockdown = False
        self.entertainment_lockdown = False
        self.masks_hygiene_distancing_multiplier = 0
        self.vaccination_propensity = 0
        self.last_vaccination_propensity = self.vaccination_propensity
        self.num_agents_to_be_vaccinated = 0
        self.num_vaccinations_today = 0

        # daily refreshed statistics here
        self.statistics = Statistics(n_locals, n_tourists, vars_util)

    def to_dict(self):
        return {"n_locals": self.n_locals, 
                "n_tourists": self.n_tourists, 
                "masks_hygience_distancing_day_thresholds": self.masks_hygiene_distancing_day_thresholds, 
                "masks_hygiene_distancing_infectiousrate_thresholds": self.masks_hygiene_distancing_infectiousrate_thresholds,
                "vaccination_day_thresholds": self.vaccination_day_thresholds,
                "vaccination_infectiousrate_thresholds": self.vaccination_infectiousrate_thresholds,
                "intervention_day_thresholds": self.intervention_day_thresholds,
                "intervention_infectiousrate_thresholds": self.intervention_infectiousrate_thresholds,
                "quarantine_enabled": self.quarantine_enabled,
                "testing_enabled": self.testing_enabled,
                "contact_tracing_enabled": self.contact_tracing_enabled,
                "masks_hygiene_distancing_multiplier": self.masks_hygiene_distancing_multiplier,
                "vaccination_propensity": self.vaccination_propensity,
                "last_vaccination_propensity": self.last_vaccination_propensity,
                "num_agents_to_be_vaccinated": self.num_agents_to_be_vaccinated,
                "infectious_rate": self.infectious_rate}
    
    def refresh_dynamic_parameters(self, day, agents_seir_state, tourists_active_ids, force_infectious_rate_refresh=True):
        infectious_rate_refreshed = False

        if force_infectious_rate_refresh:
            self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
            infectious_rate_refreshed = True

        if len(self.masks_hygiene_distancing_infectiousrate_thresholds) == 0: # if both are present, default to infectiousrate thresholds
            self.masks_hygiene_distancing_multiplier = self.get_value_by_rate_in_threshold(day, self.masks_hygiene_distancing_day_thresholds)
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True

            self.masks_hygiene_distancing_multiplier = self.get_value_by_rate_in_threshold(self.infectious_rate, self.masks_hygiene_distancing_infectiousrate_thresholds)

        if len(self.vaccination_infectiousrate_thresholds) == 0:
            self.vaccination_propensity = self.get_value_by_rate_in_threshold(day, self.vaccination_day_thresholds)
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True

            self.vaccination_propensity = self.get_value_by_rate_in_threshold(self.infectious_rate, self.vaccination_infectiousrate_thresholds)        

        if len(self.intervention_infectiousrate_thresholds) == 0:
            quarantine_threshold, testing_threshold, contacttracing_threshold = self.intervention_day_thresholds[0], self.intervention_day_thresholds[1], self.intervention_day_thresholds[2]

            self.quarantine_enabled = day >= quarantine_threshold
            self.testing_enabled = day >= testing_threshold
            self.contact_tracing_enabled = day >= contacttracing_threshold
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True
            
            quarantine_threshold, testing_threshold, contacttracing_threshold = self.intervention_infectiousrate_thresholds[0], self.intervention_infectiousrate_thresholds[1], self.intervention_infectiousrate_thresholds[2]

            self.quarantine_enabled = self.infectious_rate >= quarantine_threshold
            self.testing_enabled = self.infectious_rate >= testing_threshold
            self.contact_tracing_enabled = self.infectious_rate >= contacttracing_threshold

        if len(self.lockdown_infectiousrate_thresholds) == 0:
            workplaces_threshold, schools_threshold, entertainment_threshold = self.lockdown_day_thresholds[0], self.lockdown_day_thresholds[1], self.lockdown_day_thresholds[2]

            self.workplaces_lockdown = day >= workplaces_threshold
            self.schools_lockdown = day >= schools_threshold
            self.entertainment_lockdown = day >= entertainment_threshold
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True
            
            workplaces_threshold, schools_threshold, entertainment_threshold = self.lockdown_infectiousrate_thresholds[0], self.lockdown_infectiousrate_thresholds[1], self.lockdown_infectiousrate_thresholds[2]

            self.workplaces_lockdown = self.infectious_rate >= workplaces_threshold
            self.schools_lockdown = self.infectious_rate >= schools_threshold
            self.entertainment_lockdown = self.infectious_rate >= entertainment_threshold

    def log_intervention_info():
        pass

    def get_value_by_rate_in_threshold(self, rate, params, none_val=0):
        val = None
        for index, param in enumerate(params):
            if rate > param[0]:
                if index + 1 < len(params):
                    next_param = params[index+1]

                    if rate < next_param[0]:
                        val = param[1] # smaller than next
                        break
                else:
                    val = param[1] # end of array
                    break

        if val is None:
            val=none_val

        return val
    
# class DynamicParamsEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, DynamicParams):
#             return {"data": obj.data}
#         return super().default(obj)