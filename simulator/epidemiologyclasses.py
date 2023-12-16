from enum import IntEnum

class SEIRState(IntEnum):
    Undefined = 0
    Susceptible = 1 # not infected, susceptible to become exposed
    Exposed = 2 # infected, not yet infectious
    Infectious = 3 # infected and infectious
    Recovered = 4 # recovered, immune for an immunity period
    Deceased = 5 # deceased

class InfectionType(IntEnum):
    Undefined = 0
    PreAsymptomatic = 1 # infected, no symptoms yet, will not have symptoms (not infectious yet but will be)
    PreSymptomatic = 2 # infected, no symptoms yet, will have symptoms (not infectious yet but will be)
    Asymptomatic = 3 # infected, no symptoms, infectious
    Symptomatic = 4 # infected, with symptoms, infectious

class Severity(IntEnum):
    Undefined = 0
    Mild = 1 # mild symptoms
    Severe = 2 # severe symptoms
    Critical = 3 # critical symptoms

class EpidemiologyProbabilities(IntEnum):
    SusceptibilityMultiplier = 0
    SymptomaticProbability = 1
    SevereProbability = 2
    CriticalProbability = 3
    DeceasedProbability = 4

# state_transition_by_day - handled per agent: { day : (state_transition, timestep)}
# in itinerary, if current day exists in state_transition_by_day, switch the state in the agents_seir_state (with the new SEIRState enum), and update the agents_seir_state_transition (with the timestep)
# when an infected agent meets a susceptible agent, the virus transmission model below is activated for the pair
# handle InfectionType and Severity in agents_infection_type, and agents_infection_severity dicts (which only contains the keys of currently infected agents)
# to do - include the gene + the LTI (as multipliers within the infection probability)
# to handle - mask-wearing + quarantine + testing + vaccination + contact tracing
class SEIRStateTransition(IntEnum):
    ExposedToInfectious = 0, # in the case of a direct contact, (base_prob * susc_multiplier) chance of being exposed: if exposed (infected, not yet infectious), sample ExpToInfDays
    InfectiousToSymptomatic = 1, # if exposed/infected, compute symptomatic_probability: if symptomatic, assign "Presymptomatic", sample InfToSymp, assign "Mild" after InfToSymp, else, assign "Asymptomatic"
    SymptomaticToSevere = 2, # if symptomatic, compute severe_probability: if severe, sample SympToSev, assign "Severe" after InfToSymp + SympToSev
    SevereToCritical = 3, # if severe, compute critical_probability: if critical, sample SevToCri, assign "Critical" after InfToSymp + SympToSev + SevToCri
    CriticalToDeath = 4, # if critical, compute death_probability: if dead, sample CriToDea, send to "dead cell" after InfToSymp + SympToSev + SevToCri + CriToDea
    AsymptomaticToRecovery = 5, # if asymptomatic (not symptomatic), sample AsympToRec, assign "Recovered" after AsympToRec
    MildToRecovery = 6, # if mild, sample MildToRec, assign "Recovered" after MildToRec
    SevereToRecovery = 7, # if severe, sample SevToRec, assign "Recovered" after SevToRec
    CriticalToRecovery = 8 # if critical, sample CriToRec, assign "Recovered" after CriToRec
    RecoveredToExposed = 9 # if recovered, sample RecToExp (uniform from range e.g. 30-90 days), assign "Exposed" after RecToExp

class QuarantineType(IntEnum):
    Symptomatic = 0, # there's a probabiliy that someone feels symptoms and quarantines immediately without having received a test result
    Positive = 1, # represents positive test result case (whether true or false positive)
    PositiveContact = 2, # contact tracing positive (primary) contact
    SecondaryContact = 3 # contact tracing secondary contact

# class InterventionAgentEvents(IntEnum):
#     Test = 0,
#     TestResult = 1,
#     Quarantine = 2,
#     ContactTracing = 3,
#     Vaccine = 4

class InterventionSimulationEvents(IntEnum):
    MasksHygeneDistancing = 0,
    ContactTracing = 1,
    PartialLockDown = 2,
    LockDown = 3