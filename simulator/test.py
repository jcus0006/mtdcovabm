import random
import numpy as np
import matplotlib.pyplot as plt
from simulator import util
import powerlaw
import networkx as nx
import collections

# Set the parameters for the power law distribution
exponent = 2.64
xmin = 10

# Generate a power law distribution with the given parameters
pl_dist = powerlaw.Power_Law(xmin=xmin, parameters=[exponent])

# Generate a single sample from the distribution
sample = np.random.choice(pl_dist.generate_random(1))

# Scale the sample by the number of people in the contact network
num_people = 100 # Replace with the actual number of people in the contact network
scaled_sample = sample * num_people

print("Sample:", sample)
print("Scaled Sample:", scaled_sample)

# n = 1  # number of nodes in the graph
# m = 2  # number of edges each new node connects to
# exponent = 2.5  # power law exponent
# xmin = 4  # minimum degree

# # generate degree sequence using power law distribution
# pldist = powerlaw.Power_Law(xmin=xmin, parameters=[exponent])
# degrees = pldist.generate_random(n)

# # assign degrees to nodes in initial graph
# G = nx.empty_graph(m)
# degrees = np.array(degrees, dtype=int)
# G.add_nodes_from(range(m), degree=degrees[:m])
# for i in range(m, n):
#     # preferentially connect new node to nodes with high degree
#     probs = list(nx.get_node_attributes(G, 'degree').values())
#     probs = np.power(probs, exponent)
#     probs = probs / sum(probs)
#     targets = np.random.choice(G.nodes, size=m, replace=False, p=probs)
#     G.add_node(i, degree=degrees[i])
#     G.add_edges_from(zip([i]*m, targets))
#     for t in targets:
#         G.nodes[t]['degree'] += 1

# print(G.degree())

# Set the parameters
# exponent = 2.64
# multipliers = [0.2, 1, 1.8]

# # Calculate the scaling parameter (xmin)
# # xmin = np.power(np.prod(multipliers), -1/(exponent-1))
# xmin = 4

# # Generate a sample of 100 values
# data = xmin * powerlaw.Power_Law(xmin=xmin, parameters=[exponent]).generate_random(100)

# # Apply the multipliers to the sample
# sample = np.array(data) * np.random.choice(multipliers, size=len(data))

# # Plot the sample
# powerlaw.plot_pdf(sample, color='b')

# Define the base probability of taking leave for each age group
age_groups = 16
leave_probs = np.random.rand(age_groups)

# Generate the binary outcomes for each day for each person
num_people = 1000
num_days = 365
leave_days = np.zeros((num_people, num_days))
for i in range(num_people):
    for j in range(num_days):
        age_group = np.random.randint(age_groups)
        leave_prob = leave_probs[age_group]
        take_leave = np.random.binomial(1, leave_prob)
        if take_leave:
            num_leave_days = np.random.poisson(5)  # example mean of 5 leave days
            # Distribute the leave days into groups of varying sizes
            leave_days[i, j] = num_leave_days

# generate a quick look up for agents that join and leave the cell in every timestep, in a range of 144 timesteps
# assuming cell_agent_dict is the original dictionary, it is being filled in with random values
# start_time_dict and end_time_dict will be created in the context of every cell

cell_agent_dict = {}

for cellid in range(1, 351):
    for agentid in range(1, 1001):
        x = random.randint(1, 144)
        y = random.randint(1, 144)

        if x > y:
            starttimestep = y
            endtimestep = x
        else:
            starttimestep = x
            endtimestep = y

        if cellid not in cell_agent_dict:
            cell_agent_dict[cellid] = []

        cell_agent_dict[cellid].append((agentid, starttimestep, endtimestep)) # (agentid, starttimestep, endtimestep)

start_time_dict = {i:[] for i in range(1, 145)}
end_time_dict = {i:[] for i in range(1, 145)}

cell_ids = list(cell_agent_dict.keys())

for cellid in cell_ids:
    start_time_dict = {i:[] for i in range(1, 145)}
    end_time_dict = {i:[] for i in range(1, 145)}

    agent_ranges_by_cellid = cell_agent_dict[cellid]

    for agentid, starttime, endtime in agent_ranges_by_cellid:
        start_time_dict[starttime].append(agentid)
        end_time_dict[endtime].append(agentid)

# Visualize the leave days for a sample individual
import matplotlib.pyplot as plt
plt.plot(leave_days[0])
plt.xlabel('Day')
plt.ylabel('Number of leave days')
plt.show()

try:
    # Set the number of samples to generate
    n_samples = 10000

    range_from_1 = 7

    # Generate samples favoring the lower end of the range
    alpha_low = 1
    beta_low = 8
    low_values = np.random.beta(alpha_low, beta_low, n_samples) * range_from_1 + 1

    # Generate samples favoring the higher end of the range
    # alpha_high = 4
    # beta_high = 1
    # high_values = np.random.beta(alpha_high, beta_high, n_samples) * range_from_1 + 1

    # Compute the histograms
    low_hist, low_edges = np.histogram(low_values, bins=range_from_1, range=(1, range_from_1+1))
    # high_hist, high_edges = np.histogram(high_values, bins=range_from_1, range=(1, range_from_1+1))

    # Plot the histograms
    width = 0.4
    x_low = low_edges[:-1] + width / 2
    # x_high = high_edges[:-1] + width / 2
    fig, ax = plt.subplots()
    ax.bar(x_low, low_hist, width, label='Favors Lower End')
    # ax.bar(x_high, high_hist, width, label='Favors Higher End')
    ax.legend()
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title('Comparison of Two Beta Distributions')
    plt.show()

except Exception as e:
    print(e.message)


# Define primary frequency distribution for activities
activity_probabilities = {'activity1': 0.1, 'activity2': 0.3, 'activity3': 0.2, 'activity4': 0.15, 'activity5': 0.25}

# Define age bracket frequency distributions for activities
activity_age_probabilities = {
    'activity1': {'age1': 0.2, 'age2': 0.3, 'age3': 0.5},
    'activity2': {'age1': 0.4, 'age2': 0.3, 'age3': 0.3},
    'activity3': {'age1': 0.3, 'age2': 0.4, 'age3': 0.3},
    'activity4': {'age1': 0.4, 'age2': 0.2, 'age3': 0.4},
    'activity5': {'age1': 0.2, 'age2': 0.3, 'age3': 0.5}
}

# Define agent's age bracket
agent_age = 'age2'

# Calculate probability of each activity for agent
activity_probabilities_for_agent = {}
for activity in activity_probabilities:
    primary_probability = activity_probabilities[activity]
    age_probability = activity_age_probabilities[activity][agent_age]
    activity_probabilities_for_agent[activity] = primary_probability * age_probability

# Normalize probabilities so they sum to 1
total_probability = sum(activity_probabilities_for_agent.values())
for activity in activity_probabilities_for_agent:
    activity_probabilities_for_agent[activity] /= total_probability

# Sample an activity for the agent based on their probability distribution
chosen_activity = random.choices(list(activity_probabilities_for_agent.keys()), list(activity_probabilities_for_agent.values()))[0]
print(f"The agent's chosen activity is {chosen_activity}")