import random
import numpy as np
import matplotlib.pyplot as plt
from simulator import util

# values = []

# for i in range(1000):
#     values.append(util.sample_gamma_reject_out_of_range(5, 1, 7, 2, True, True))

# Set parameters for beta distribution
# alpha = 5
# beta = 1
# low = 1
# high = 9

# # Generate 1000 random numbers from beta distribution
# values = np.random.beta(alpha, beta, size=1000)

# # Rescale to desired range
# values = values * (high - low) + low

# # Plot histogram of values
# plt.hist(values, bins=20)
# plt.show()

try:
    # Set the number of samples to generate
    n_samples = 10000

    range_from_1 = 7

    # Generate samples favoring the lower end of the range
    alpha_low = 2
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