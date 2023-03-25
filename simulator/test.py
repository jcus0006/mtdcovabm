import numpy as np
import matplotlib.pyplot as plt

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


plt.plot(leave_days[0])
plt.xlabel('Day')
plt.ylabel('Number of leave days')
plt.show()
