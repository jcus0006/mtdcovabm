from epidemiologyclasses import SEIRState
import time
# import json

class DynamicParams:
    def __init__(self, n_locals, n_tourists, epidemiologyparams):
        self.n_locals = n_locals
        self.n_tourists = n_tourists
        self.masks_hygiene_distancing_day_thresholds = epidemiologyparams["masks_hygiene_distancing_day_thresholds"]
        self.masks_hygiene_distancing_infectiousrate_thresholds = epidemiologyparams["masks_hygiene_distancing_infectiousrate_thresholds"]
        self.vaccination_day_thresholds = epidemiologyparams["vaccination_day_thresholds"]
        self.vaccination_infectiousrate_thresholds = epidemiologyparams["vaccination_infectiousrate_thresholds"]
        self.intervention_day_thresholds = epidemiologyparams["intervention_day_thresholds"]
        self.intervention_infectiousrate_thresholds = epidemiologyparams["intervention_infectiousrate_thresholds"]

        self.quarantine_enabled = False
        self.testing_enabled = False
        self.contact_tracing_enabled = False
        self.masks_hygiene_distancing_multiplier = 0
        self.vaccination_propensity = 0
        self.last_vaccination_propensity = self.vaccination_propensity
        self.num_agents_to_be_vaccinated = 0

        # daily refreshed statistics here
        self.infectious_rate = 0

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

    def refresh_infectious_rate(self, agents_seir_state, tourists_active_ids): # to optimise
        n_infectious = 0
        n_inactive = 0

        start = time.time()

        n_infectious = sum([1 for index, state in enumerate(agents_seir_state) if index < self.n_locals and state == SEIRState.Infectious])
        n_inactive = sum([1 for index, state in enumerate(agents_seir_state) if index < self.n_locals and state == SEIRState.Deceased])
        
        n_infectious += sum([1 for tourist_id in tourists_active_ids if agents_seir_state[self.n_locals + tourist_id] == SEIRState.Infectious])
        n_inactive += sum([1 for tourist_id in tourists_active_ids if agents_seir_state[self.n_locals + tourist_id] == SEIRState.Deceased])

        time_taken = time.time() - start
        print("refresh infectious rate (compr) time taken: " + str(time_taken))

        # start = time.time()

        # for index, state in enumerate(agents_seir_state):
        #     if index < self.n_locals or index in tourists_active_ids:
        #         if state == SEIRState.Infectious:
        #             n_infectious += 1
        #         elif state == SEIRState.Deceased:
        #             n_inactive += 1
        #     else:
        #         n_inactive += 1

        # time_taken = time.time() - start
        # print("refresh infectious rate (loop) time taken: " + str(time_taken))

        n_total = self.n_locals + len(tourists_active_ids)

        n_active = n_total - n_inactive

        self.infectious_rate = n_infectious / n_active
    
    def refresh_dynamic_parameters(self, day, agents_seir_state, tourists_active_ids, force_infectious_rate_refresh=True):
        infectious_rate_refreshed = False

        if force_infectious_rate_refresh:
            self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
            infectious_rate_refreshed = True

        if len(self.masks_hygiene_distancing_day_thresholds) > 0:
            self.masks_hygiene_distancing_multiplier = self.get_value_by_rate_in_threshold(day, self.masks_hygiene_distancing_day_thresholds)

            # for threshold, propensity in self.masks_hygiene_distancing_day_thresholds:
            #     if day >= threshold:
            #         self.masks_hygiene_distancing_propensity = propensity
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True

            self.masks_hygiene_distancing_multiplier = self.get_value_by_rate_in_threshold(self.infectious_rate, self.masks_hygiene_distancing_infectiousrate_thresholds)
            # for threshold, propensity in self.masks_hygiene_distancing_infectiousrate_thresholds:
            #     if self.infectious_rate >= threshold:
            #         self.masks_hygiene_distancing_propensity = propensity
            #         break

        if len(self.vaccination_day_thresholds) > 0:
            self.vaccination_propensity = self.get_value_by_rate_in_threshold(day, self.vaccination_day_thresholds)

            # for threshold, propensity in self.vaccination_day_thresholds:
            #     if day >= threshold:
            #         self.vaccination_propensity = propensity
            #         break
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True

            self.vaccination_propensity = self.get_value_by_rate_in_threshold(self.infectious_rate, self.vaccination_infectiousrate_thresholds)        
            # for threshold, propensity in self.vaccination_infectiousrate_thresholds:
            #     if self.infectious_rate >= threshold:
            #         self.vaccination_propensity = propensity
            #         break

        if len(self.intervention_day_thresholds) > 0:
            quarantine, testing, contacttracing = self.intervention_day_thresholds[0], self.intervention_day_thresholds[1], self.intervention_day_thresholds[2]

            if day >= quarantine:
                self.quarantine_enabled = True
            else:
                self.quarantine_enabled = False

            if day >= testing:
                self.testing_enabled = True
            else:
                self.testing_enabled = False

            if day >= contacttracing:
                self.contact_tracing_enabled = True
            else:
                self.contact_tracing_enabled = False
        else:
            if not infectious_rate_refreshed:
                self.refresh_infectious_rate(agents_seir_state, tourists_active_ids)
                infectious_rate_refreshed = True
            
            quarantine, testing, contacttracing = self.intervention_infectiousrate_thresholds[0], self.intervention_infectiousrate_thresholds[1], self.intervention_infectiousrate_thresholds[2]

            if self.infectious_rate >= quarantine:
                self.quarantine_enabled = True
            else:
                self.quarantine_enabled = False

            if self.infectious_rate >= testing:
                self.testing_enabled = True
            else:
                self.testing_enabled = False

            if self.infectious_rate >= contacttracing:
                self.contact_tracing_enabled = True
            else:
                self.contact_tracing_enabled = False

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