import random
import numpy as np
import matplotlib.pyplot as plt
from simulator import util
import powerlaw
import networkx as nx
import collections

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# # Define parameters
# mean = 15    # Mean number of contacts per unit time
# timesteps = 144    # Number of timesteps
# min_contacts = 1    # Minimum number of contacts per timestep
# max_contacts = 30   # Maximum number of contacts per timestep
# n = 1000

# # Sample from Poisson distribution for each timestep
# contacts_per_timestep = stats.poisson(mu=mean).rvs(size=n)

# # Clip values to fit within the minimum and maximum bounds
# contacts_per_timestep = np.clip(contacts_per_timestep, min_contacts, max_contacts)

# # Plot histogram of the truncated distribution
# plt.hist(contacts_per_timestep, bins=max_contacts-min_contacts+1, range=[min_contacts-0.5, max_contacts+0.5], density=True)

# # Add vertical line for mean
# plt.axvline(x=mean, color='r', linestyle='--')

# # Set plot title and labels
# plt.title("Poisson Distribution with Mean = {}".format(mean))
# plt.xlabel("Number of Contacts")
# plt.ylabel("Frequency")

# # Show plot
# plt.show()

# # Print the resulting contacts per timestep
# print(contacts_per_timestep)

# # Parameters
# mean = 15
# min_val = 3
# max_val = 30
# num_samples = 1000

# # Generate Poisson distribution
# samples = np.random.poisson(mean, size=num_samples)

# # Truncate distribution between min and max values
# samples = np.clip(samples, min_val, max_val)

# # Plot histogram of the truncated distribution
# plt.hist(samples, bins=max_val-min_val+1, range=[min_val-0.5, max_val+0.5], density=True)

# # Add vertical line for mean
# plt.axvline(x=mean, color='r', linestyle='--')

# # Set plot title and labels
# plt.title("Poisson Distribution with Mean = {}".format(mean))
# plt.xlabel("Number of Contacts")
# plt.ylabel("Frequency")

# # Show plot
# plt.show()


# Define the number of nodes in the graph
# num_nodes = 50

# # Define the power law distribution parameters for max number of contacts
# alpha = 2.0
# xmin = 0
# xmax = 1

# # Define the time range limits
# min_time = 1
# max_time = 144

# # Create the graph object
# G = nx.Graph()

# # Add nodes to the graph with max number of contacts sampled from power law distribution
# for i in range(num_nodes):
#     max_contacts = int(nx.utils.powerlaw_sequence(1, alpha, seed=i)[0])
#     G.add_node(i, max_contacts=max_contacts)

# # normalize to sum to 1
# max_contacts_arr = [G.nodes[i]['max_contacts'] for i in range(len(G.nodes))]
# sum_max_contacts = sum(max_contacts_arr)

# for i in range(num_nodes):
#     G.nodes[i]['max_contacts'] /= sum_max_contacts

# # Add edges to the graph with potential direct contacts and timerange labels
# for i in range(num_nodes):
#     for j in range(i+1, num_nodes):
#         # Calculate the probability of an edge between nodes i and j based on the power law distribution
#         prob = 1.0 / ((i+1)**alpha + (j+1)**alpha - 2)
#         if random.random() < prob:
#             # Add the edge with a timerange label between min_time and max_time
#             timerange = random.randint(min_time, max_time)
#             G.add_edge(i, j, timerange=timerange, direct_contact=False)

# # Set a maximum number of direct contact edges per node for each time step
# for time_step in range(min_time, max_time+1):
#     for i in range(num_nodes):
#         max_contacts = G.nodes[i]['max_contacts']
#         potential_contacts = [j for j in G.neighbors(i) if G[i][j]['timerange'] == time_step and not G[i][j]['direct_contact']]
#         num_potential_contacts = len(potential_contacts)
#         if num_potential_contacts > 0:
#             # Compute the actual number of direct contacts for the node at the current time step
#             sampled_value = random.random()
#             num_direct_contacts = int(sampled_value * num_potential_contacts * max_contacts)
#             num_direct_contacts = min(num_direct_contacts, max_contacts)  # Enforce the maximum number of contacts
#             # Set the direct contact edges for the node at the current time step
#             for j in random.sample(potential_contacts, num_direct_contacts):
#                 G[i][j]['direct_contact'] = True

# # Print the graph information
# print(nx.info(G))

# Define the parameters of the distribution
alpha = 2.64
xmin = 4

# Create the power law distribution
dist = powerlaw.Power_Law(xmin=xmin, parameters=[alpha])

# Generate 1000 samples from the distribution
samples = dist.generate_random(1000)

# Create a figure with two subplots
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Plot the probability density function (PDF) of the data
ax[0].set_title('PDF')
powerlaw.plot_pdf(samples, ax=ax[0], color='b')
ax[0].set_xlabel('Value')
ax[0].set_ylabel('Probability Density')

# Plot the complementary cumulative distribution function (CCDF) of the data
ax[1].set_title('CCDF')
powerlaw.plot_ccdf(samples, ax=ax[1], color='r')
ax[1].set_xlabel('Value')
ax[1].set_ylabel('Complementary CDF')

# Display the plot
plt.show()

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