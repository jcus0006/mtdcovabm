import time
import json
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..")) # add root to sys.path
import numpy as np
import matplotlib.pyplot as plt
from utility.npencoder import NpEncoder

class fdvalidation:

    def __init__(self, freqdistname, freqdist, expectedvals, actualvals, diffvals, femexpectedvals, femactualvals, femdiffvals, femaledist):
        self.frequencydistributionname = freqdistname
        self.frequencydistribution = freqdist
        self.expected_values = expectedvals
        self.actual_values = actualvals
        self.difference_values = diffvals
        self.fem_expected_values = femexpectedvals
        self.fem_actual_values = femactualvals
        self.fem_difference_values = femdiffvals
        self.femalefrequencydistribution = femaledist

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

fdlist = [ "population_age_distributions", 
            "enrollment_rates_by_age", 
            "sex_byage_distributions", 
            "longtermillness_byage_distributions",
            "bmi_byage_distributions",
            "employment_status_byage_distributions",
            "employment_byindustry_distributions",
            "employment_byindustry_ftpt_distributions",
            "institutions_rates_by_age",
            "institutions_type_distribution" ]

keyparams = [ "age",
              "scid",
              "gender",
              "lti",
              "bmi",
              "empstatus",
              "empind",
              "empftpt",
              "inst_res",
              "inst_type"]

figureindex = 1
def display_male_female_expected_vs_actual(propName, labelName, expected_values, actual_values, fem_expected_values, fem_actual_values):
    global figureindex

    figureindex += 1
    # plt.figure(figureindex)

    prop_exp = [v for v in expected_values.values()]
    prop_act = [v for v in actual_values.values()]

    fem_prop_exp = [v for v in fem_expected_values.values()]
    fem_prop_act = [v for v in fem_actual_values.values()]

    # show male
    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Set the width of the bars
    bar_width = 0.35

    # Set the positions of the bars on the x-axis
    ages = np.arange(101)

    # Create the expected counts bar chart
    rects1 = ax.bar(ages - bar_width/2, prop_exp, bar_width, label='Expected Counts')

    # Create the actual counts bar chart
    rects2 = ax.bar(ages + bar_width/2, prop_act, bar_width, label='Actual Counts')

    # Add axis labels and a title
    ax.set_xlabel(labelName)
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of ' + labelName + ' by Age (Male)')

    # Add a legend
    ax.legend()

    # Show the plot
    # plt.show()

    figureindex += 1
    # plt.figure(figureindex)

    # Show female
    # Create a figure and axes object
    fig, ax = plt.subplots()

    # Create the expected counts bar chart
    rects1 = ax.bar(ages - bar_width/2, fem_prop_exp, bar_width, label='Expected Counts')

    # Create the actual counts bar chart
    rects2 = ax.bar(ages + bar_width/2, fem_prop_act, bar_width, label='Actual Counts')

    # Add axis labels and a title
    ax.set_xlabel(labelName)
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of ' + labelName + ' by Age (Female)')

    # Add a legend
    ax.legend()

    # Show the plot
    # plt.show()

def compare_category_ageranges_expected_vs_actual(exp_hist, act_hist, x_label, y_label, title, fig_size, bar_width, age_ranges=np.arange(0, 100, 10)):
    global figureindex

    figureindex += 1
    # plt.figure(figureindex)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=fig_size)

    # Define the age ranges for the bars

    # Plot the expected counts as a blue bar chart
    ax.bar(age_ranges, exp_hist, width=bar_width, label='Expected Counts')

    # Plot the actual counts as an orange bar chart
    ax.bar(age_ranges + bar_width, act_hist, width=-bar_width, label='Actual Counts')

    # Set the labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if len(age_ranges) == 10:
        ax.set_xticks(age_ranges + bar_width / 2)
        ax.set_xticklabels(['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '95-100', '90-100'], rotation=45, ha="right")
    elif len(age_ranges) == 9:
        ax.set_xticks(age_ranges + bar_width / 2)
        ax.set_xticklabels(['15-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100'], rotation=45, ha="right")
    elif len(age_ranges) == 7:
        ax.set_xticks(age_ranges + bar_width / 2)
        ax.set_xticklabels(['0-14', '15-24', '25-34', '35-44', '45-54', '55-64', '65-100'], rotation=45, ha="right")

    ax.legend()
    plt.show()

def compare_category_industries_expected_vs_actual(exp_hist, act_hist, x_label, y_label, title, fig_size, bar_width):
    global figureindex

    figureindex += 1
    # plt.figure(figureindex)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=fig_size)

    # Define the age ranges for the bars
    industries = np.arange(1, 22)

    # Plot the expected counts as a blue bar chart
    ax.bar(industries, exp_hist, width=bar_width, label='Expected Counts')

    # Plot the actual counts as an orange bar chart
    ax.bar(industries + bar_width, act_hist, width=-bar_width, label='Actual Counts')

    # Set the labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    ax.set_xticks(industries + bar_width / 2)
    ax.set_xticklabels(industries)

    ax.legend()
    plt.show()

maltafile  = open('./population/Malta.json')
maltafds = json.load(maltafile)

agentsfile = open("./population/agents.json")
agents = json.load(agentsfile)

n = len(agents)

maleagents = {k:v for k, v in agents.items() if v["gender"] == 0}
femaleagents = {k:v for k, v in agents.items() if v["gender"] == 1}

agentsbyages = {}
malesbyages = {}
femalesbyages = {}

for age in range(101):
    agentsbyage = {k:v for k, v in agents.items() if v["age"] == age}
    malesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 0}
    femalesbyage = {k:v for k, v in agentsbyage.items() if v["gender"] == 1}

    agentsbyages[age] = agentsbyage
    malesbyages[age] = malesbyage
    femalesbyages[age] = femalesbyage

agentsemployed = {k:v for k, v in agents.items() if v["empstatus"] == 0}
malesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 0}
femalesemployed = {k:v for k, v in agentsemployed.items() if v["gender"] == 1}

agentsbyindustries = {}
malesbyindustries = {}
femalesbyindustries = {}

for industry in range(1,22):
    agentsbyindustry = {k:v for k, v in agentsemployed.items() if v["empind"] == industry}
    malesbyindustry = {k:v for k, v in malesemployed.items() if v["empind"] == industry}
    femalesbyindustry = {k:v for k, v in femalesemployed.items() if v["empind"] == industry}

    agentsbyindustries[industry] = agentsbyindustry
    malesbyindustries[industry] = malesbyindustry
    femalesbyindustries[industry] = femalesbyindustry

householdsfile = open("./population/households.json")
households = json.load(householdsfile)

workplacesfile = open("./population/workplaces.json")
workplaces = json.load(workplacesfile)

schoolsfile = open("./population/schools.json")
schools = json.load(schoolsfile)

institutiontypesfile = open("./population/institutiontypes.json")
institutiontypes = json.load(institutiontypesfile)

institutionsfile = open("./population/institutions.json")
institutions = json.load(institutionsfile)

smoothedagedistfile = open("./population/smoothedagedist.json")
smoothedagedist = json.load(smoothedagedistfile)

fdvalidations = []
forceload = True

if not forceload and os.path.exists('./population/fdvalidations.json'):
    fdvalidationsfile = open('./population/fdvalidations.json')
    fdvalidations = json.load(fdvalidationsfile)
else:

    for index, fd in enumerate(fdlist):
        mtfd = maltafds[fd]
        keyparam = keyparams[index]
        freqdist = None
        femalefreqdist = None

        expectedvals = {}
        actualvals = {}
        differencevals = {}

        fem_expectedvals = {}
        fem_actualvals = {}
        fem_differencevals = {}

        if fd == "population_age_distributions": # straight ages 1 to 100
            # popagedist = mtfd[1]["distribution"] # 91 bracket

            freqdist = smoothedagedist

            expectedvals = {i:0 for i in range(101)}

            for age in expectedvals:
                # popagedist_age = [agedist for agedist in popagedist if age >= agedist[0] and age <= agedist[1]][0]
                # minage, maxage, percentage = popagedist_age[0], popagedist_age[1], popagedist_age[2]

                percentage = freqdist[str(age)] # by key
                    
                expectedvals[age] = n * percentage

                agentsbyage = agentsbyages[age]

                actualvals[age] = len(agentsbyage)

                differencevals[age] = expectedvals[age] - actualvals[age]
        elif fd == "enrollment_rates_by_age":
            freqdist = mtfd

            expectedvals = {i:0 for i in range(101)}

            for age in expectedvals:
                percentage = freqdist[age][1] # by index

                agentsbyage = agentsbyages[age]

                expectedvals[age] = len(agentsbyage) * percentage

                actualvals[age] = sum([1 for agent in agentsbyage.values() if agent["scid"] is not None])

                differencevals[age] = expectedvals[age] - actualvals[age]
        elif fd == "sex_byage_distributions":
            freqdist = mtfd[0]["distribution"]

            expectedvals = {i:0 for i in range(101)}
            fem_expectedvals = {i:0 for i in range(101)}

            for age in expectedvals:
                age_bracket = [agedist for agedist in freqdist if age >= agedist[0] and age <= agedist[1]][0]
                minage, maxage, malepercentage, femalepercentage = age_bracket[0], age_bracket[1], age_bracket[2], age_bracket[3]

                agentsbyage = agentsbyages[age]

                malesbyage = malesbyages[age]
                femalesbyage = femalesbyages[age]

                expectedvals[age] = len(agentsbyage) * malepercentage
                fem_expectedvals[age] = len(agentsbyage) * femalepercentage

                actualvals[age] = len(malesbyage)
                fem_actualvals[age] = len(femalesbyage)

                differencevals[age] = expectedvals[age] - actualvals[age]
                fem_differencevals[age] = fem_expectedvals[age] - fem_actualvals[age]
        elif fd == "longtermillness_byage_distributions":
            freqdist = mtfd[0]["maledistribution"]
            femalefreqdist = mtfd[0]["femaledistribution"]

            expectedvals = {i:0 for i in range(101) if i >= 15}
            fem_expectedvals = {i:0 for i in range(101) if i >= 15}

            for age in expectedvals:
                if age >= 15:
                    male_age_bracket = [agedist for agedist in freqdist if age >= agedist[0] and age <= agedist[1]][0]
                    male_minage, male_maxage, male_withillness_percent, male_withoutillness_percent = male_age_bracket[0], male_age_bracket[1], male_age_bracket[2], male_age_bracket[3]

                    female_age_bracket = [agedist for agedist in femalefreqdist if age >= agedist[0] and age <= agedist[1]][0]
                    female_minage, female_maxage, female_withillness_percent, female_withoutillness_percent = female_age_bracket[0], female_age_bracket[1], female_age_bracket[2], female_age_bracket[3]

                    malesbyage = malesbyages[age]
                    femalesbyage = femalesbyages[age]

                    expectedvals[age] = [len(malesbyage) * male_withillness_percent, len(malesbyage) * male_withoutillness_percent]
                    fem_expectedvals[age] = [len(femalesbyage) * female_withillness_percent, len(femalesbyage) * female_withoutillness_percent]

                    actualvals[age] = [sum([1 for agent in malesbyage.values() if agent["lti"] == 0]), sum([1 for agent in malesbyage.values() if agent["lti"] == 1])]
                    fem_actualvals[age] = [sum([1 for agent in femalesbyage.values() if agent["lti"] == 0]), sum([1 for agent in femalesbyage.values() if agent["lti"] == 1])]
                    
                    differencevals[age] = [expectedvals[age][0] - actualvals[age][0], expectedvals[age][1] - actualvals[age][1]]
                    fem_differencevals[age] = [fem_expectedvals[age][0] - fem_actualvals[age][0], fem_expectedvals[age][1] - fem_actualvals[age][1]]
        elif fd == "bmi_byage_distributions":
            freqdist = mtfd[0]["maledistribution"]
            femalefreqdist = mtfd[0]["femaledistribution"]

            expectedvals = {i:0 for i in range(101)}
            fem_expectedvals = {i:0 for i in range(101)}

            for age in expectedvals:
                male_age_bracket = [agedist for agedist in freqdist if age >= agedist[0] and age <= agedist[1]][0]
                male_minage, male_maxage, male_normal_percent, male_overweight_percent, male_obese_percent = male_age_bracket[0], male_age_bracket[1], male_age_bracket[2], male_age_bracket[3], male_age_bracket[4]

                female_age_bracket = [agedist for agedist in femalefreqdist if age >= agedist[0] and age <= agedist[1]][0]
                female_minage, female_maxage, female_normal_percent, female_overweight_percent, female_obese_percent = female_age_bracket[0], female_age_bracket[1], female_age_bracket[2], female_age_bracket[3], female_age_bracket[4]

                malesbyage = malesbyages[age]
                femalesbyage = femalesbyages[age]

                expectedvals[age] = [len(malesbyage) * male_normal_percent, 
                                    len(malesbyage) * male_overweight_percent, 
                                    len(malesbyage) * male_obese_percent]

                fem_expectedvals[age] = [len(femalesbyage) * female_normal_percent, 
                                        len(femalesbyage) * female_overweight_percent, 
                                        len(femalesbyage) * female_obese_percent]

                actualvals[age] = [sum([1 for agent in malesbyage.values() if agent["bmi"] == 0]), 
                                    sum([1 for agent in malesbyage.values() if agent["bmi"] == 1]),
                                    sum([1 for agent in malesbyage.values() if agent["bmi"] == 2])]

                fem_actualvals[age] = [sum([1 for agent in femalesbyage.values() if agent["bmi"] == 0]), 
                                        sum([1 for agent in femalesbyage.values() if agent["bmi"] == 1]),
                                        sum([1 for agent in femalesbyage.values() if agent["bmi"] == 2])]
                
                differencevals[age] = [expectedvals[age][0] - actualvals[age][0], 
                                        expectedvals[age][1] - actualvals[age][1],
                                        expectedvals[age][2] - actualvals[age][2]]

                fem_differencevals[age] = [fem_expectedvals[age][0] - fem_actualvals[age][0], 
                                            fem_expectedvals[age][1] - fem_actualvals[age][1],
                                            fem_expectedvals[age][2] - fem_actualvals[age][2]]
        elif fd == "employment_status_byage_distributions":
            freqdist = mtfd[0]["maledistribution"]
            femalefreqdist = mtfd[0]["femaledistribution"]

            expectedvals = {i:0 for i in range(101) if i >= 15}
            fem_expectedvals = {i:0 for i in range(101) if i >= 15}

            for age in expectedvals:
                if age >= 15:
                    male_age_bracket = [agedist for agedist in freqdist if age >= agedist[0] and age <= agedist[1]][0]
                    male_minage, male_maxage, male_emp_percent, male_unemp_percent, male_inactive_percent = male_age_bracket[0], male_age_bracket[1], male_age_bracket[2], male_age_bracket[3], male_age_bracket[4]

                    female_age_bracket = [agedist for agedist in femalefreqdist if age >= agedist[0] and age <= agedist[1]][0]
                    female_minage, female_maxage, female_emp_percent, female_unemp_percent, female_inactive_percent = female_age_bracket[0], female_age_bracket[1], female_age_bracket[2], female_age_bracket[3], female_age_bracket[4]

                    malesbyage = malesbyages[age]
                    femalesbyage = femalesbyages[age]

                    expectedvals[age] = [len(malesbyage) * male_emp_percent, 
                                        len(malesbyage) * male_unemp_percent, 
                                        len(malesbyage) * male_inactive_percent]

                    fem_expectedvals[age] = [len(femalesbyage) * female_emp_percent, 
                                            len(femalesbyage) * female_unemp_percent, 
                                            len(femalesbyage) * female_inactive_percent]

                    actualvals[age] = [sum([1 for agent in malesbyage.values() if agent["empstatus"] == 0]), 
                                        sum([1 for agent in malesbyage.values() if agent["empstatus"] == 1]),
                                        sum([1 for agent in malesbyage.values() if agent["empstatus"] == 2])]

                    fem_actualvals[age] = [sum([1 for agent in femalesbyage.values() if agent["empstatus"] == 0]), 
                                            sum([1 for agent in femalesbyage.values() if agent["empstatus"] == 1]),
                                            sum([1 for agent in femalesbyage.values() if agent["empstatus"] == 2])]
                    
                    differencevals[age] = [expectedvals[age][0] - actualvals[age][0], 
                                            expectedvals[age][1] - actualvals[age][1],
                                            expectedvals[age][2] - actualvals[age][2]]

                    fem_differencevals[age] = [fem_expectedvals[age][0] - fem_actualvals[age][0], 
                                                fem_expectedvals[age][1] - fem_actualvals[age][1],
                                                fem_expectedvals[age][2] - fem_actualvals[age][2]]
        elif fd == "employment_byindustry_distributions":
            freqdist = mtfd[0]["maledistribution"]
            femalefreqdist = mtfd[0]["femaledistribution"]

            industries = {i:0 for i in range(1,22)} # 1 to 21 industries

            for industry in industries:
                male_bracket = [ind for ind in freqdist if ind[0] == industry][0]
                male_percent = male_bracket[1]
                female_bracket = [ind for ind in femalefreqdist if ind[0] == industry][0]
                female_percent = female_bracket[1]

                expectedvals[industry] = len(malesemployed) * male_percent
                fem_expectedvals[industry] = len(femalesemployed) * female_percent

                actualvals[industry] = sum([1 for agent in maleagents.values() if agent["empind"] == industry])
                fem_actualvals[industry] = sum([1 for agent in femaleagents.values() if agent["empind"] == industry])

                differencevals[industry] = expectedvals[industry] - actualvals[industry]
                fem_differencevals[industry] = fem_expectedvals[industry] - fem_actualvals[industry]
        elif fd == "employment_byindustry_ftpt_distributions":
            freqdist = mtfd[0]["maledistribution"]
            femalefreqdist = mtfd[0]["femaledistribution"]

            industries = {i:0 for i in range(1,22)} # 1 to 21 industries

            for industry in industries:
                male_bracket = [ind for ind in freqdist if ind[0] == industry][0]
                ind, male_ft_percent, male_ftpt_percent, male_ftnopt_percent = male_bracket[0], male_bracket[1], male_bracket[2], male_bracket[3]

                female_bracket = [ind for ind in femalefreqdist if ind[0] == industry][0]
                ind, female_ft_percent, female_ftpt_percent, female_ftnopt_percent = female_bracket[0], female_bracket[1], female_bracket[2], female_bracket[3]

                malesbyindustry = malesbyindustries[industry]
                femalesbyindustry = femalesbyindustries[industry]

                expectedvals[ind] = [len(malesbyindustry) * male_ft_percent, 
                                    len(malesbyindustry) * male_ftpt_percent, 
                                    len(malesbyindustry) * male_ftnopt_percent]

                fem_expectedvals[ind] = [len(femalesbyindustry) * female_ft_percent, 
                                        len(femalesbyindustry) * female_ftpt_percent, 
                                        len(femalesbyindustry) * female_ftnopt_percent]

                actualvals[ind] = [sum([1 for agent in malesbyindustry.values() if agent["empftpt"] == 0]), 
                                    sum([1 for agent in malesbyindustry.values() if agent["empftpt"] == 1]),
                                    sum([1 for agent in malesbyindustry.values() if agent["empftpt"] == 2])]

                fem_actualvals[ind] = [sum([1 for agent in femalesbyindustry.values() if agent["empftpt"] == 0]), 
                                        sum([1 for agent in femalesbyindustry.values() if agent["empftpt"] == 1]),
                                        sum([1 for agent in femalesbyindustry.values() if agent["empftpt"] == 2])]
                
                differencevals[ind] = [expectedvals[ind][0] - actualvals[ind][0], 
                                        expectedvals[ind][1] - actualvals[ind][1],
                                        expectedvals[ind][2] - actualvals[ind][2]]

                fem_differencevals[ind] = [fem_expectedvals[ind][0] - fem_actualvals[ind][0], 
                                            fem_expectedvals[ind][1] - fem_actualvals[ind][1],
                                            fem_expectedvals[ind][2] - fem_actualvals[ind][2]]
        elif fd == "institutions_rates_by_age":
            freqdist = mtfd

            expectedvals = {i:0 for i in range(101)}
            fem_expectedvals = {i:0 for i in range(101)}

            for age in expectedvals:
                age_bracket = [agedist for agedist in freqdist if age >= agedist[0] and age <= agedist[1]][0]

                percentage = age_bracket[2] # by index

                agentsbyage = agentsbyages[age]

                expectedvals[age] = len(agentsbyage) * percentage

                actualvals[age] = sum([1 for agent in agentsbyage.values() if agent["inst_res"] == 1])

                differencevals[age] = expectedvals[age] - actualvals[age]
        elif fd == "institutions_type_distribution":
            freqdist = mtfd

            institutiontypeids = {i:0 for i in range(1,12)} # 1 to 11 institutiontypes

            for institutiontype in institutiontypeids:
                institutiontype_bracket = [insttype for insttype in freqdist if insttype[0] == institutiontype][0]

                percent = institutiontype_bracket[1]

                expectedvals[institutiontype] = len(institutions) * percent

                actualvals[institutiontype] = sum([1 for institution in institutions if institution["insttypeid"] == institutiontype])

                differencevals[institutiontype] = expectedvals[institutiontype] - actualvals[institutiontype]

        fdval = fdvalidation(fd, freqdist, expectedvals, actualvals, differencevals, fem_expectedvals, fem_actualvals, fem_differencevals, femalefreqdist)
        fdvalidations.append(fdval)

    timestr = str(int(time.time()))
    newpath = "output\\" + timestr

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    fdvalidations_filename = "fdvalidations.json"

    def obj_dict(obj):
        return obj.__dict__

    # first make class serializable
    with open(os.path.join(newpath, fdvalidations_filename), 'w', encoding='utf-8') as f:
        json.dump(fdvalidations, f, ensure_ascii=False, indent=4, cls=NpEncoder, default=obj_dict)

    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':100})

for fdval in fdvalidations:
    frequencydistributionname = fdval["frequencydistributionname"] if type(fdval) is dict else fdval.frequencydistributionname 
    expected_values = fdval["expected_values"] if type(fdval) is dict else fdval.expected_values
    actual_values = fdval["actual_values"] if type(fdval) is dict else fdval.actual_values
    fem_expected_values = fdval["fem_expected_values"] if type(fdval) is dict else fdval.fem_expected_values
    fem_actual_values = fdval["fem_actual_values"] if type(fdval) is dict else fdval.fem_actual_values

    if frequencydistributionname == "population_age_distributions":
        agents_ages = [v["age"] for v in agents.values()]

        plt.figure(figureindex)

        plt.hist(agents_ages, bins=101)
        
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.title("Distribution of Ages")
        plt.show()

        ages_exp = [v for v in expected_values.values()]
        ages_act = [v for v in actual_values.values()]

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        ages = np.arange(101)

        figureindex += 1
        # plt.figure(figureindex)

        # Create the expected counts bar chart
        rects1 = ax.bar(ages - bar_width/2, ages_exp, bar_width, label='Expected Counts')

        # Create the actual counts bar chart
        rects2 = ax.bar(ages + bar_width/2, ages_act, bar_width, label='Actual Counts')

        # Add axis labels and a title
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Ages')

        # Add a legend
        ax.legend()

        # Show the plot
        # plt.show()
    elif frequencydistributionname == "enrollment_rates_by_age":
        agents_by_enrolled = [1 if v["scid"] is not None else 0 for v in agents.values()]

        num_enrolled = sum([enrolled for enrolled in agents_by_enrolled])
        num_not_enrolled = len(agents_by_enrolled) - num_enrolled

        prop_enrolled = num_enrolled / len(agents_by_enrolled)
        prop_notenrolled = num_not_enrolled / len(agents_by_enrolled)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['Enrolled', 'Not Enrolled']
        proportions = [prop_enrolled, prop_notenrolled]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('Enrollment')
        plt.ylabel('Proportion')
        plt.title('Enrolled-NotEnrolled Split')
        plt.show()

        # compare expected with actual

        figureindex += 1
        # plt.figure(figureindex)

        enrolled_exp = [v for v in expected_values.values()]
        enrolled_act = [v for v in actual_values.values()]

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        ages = np.arange(101)

        # Create the expected counts bar chart
        rects1 = ax.bar(ages - bar_width/2, enrolled_exp, bar_width, label='Expected Counts')

        # Create the actual counts bar chart
        rects2 = ax.bar(ages + bar_width/2, enrolled_act, bar_width, label='Actual Counts')

        # Add axis labels and a title
        ax.set_xlabel('Age')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Enrollment')

        # Add a legend
        ax.legend()

        # Show the plot
        # plt.show()
    elif frequencydistributionname == "sex_byage_distributions":
        num_males = len(maleagents)
        num_females = len(femaleagents)

        prop_males = num_males / len(agents)
        prop_females = num_females / len(agents)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['Males', 'Females']
        proportions = [prop_males, prop_females]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('Gender')
        plt.ylabel('Proportion')
        plt.title('Male-Female Split')
        plt.show()

        display_male_female_expected_vs_actual("gender", "Gender", expected_values, actual_values, fem_expected_values, fem_actual_values)
    elif frequencydistributionname == "longtermillness_byage_distributions":
        agents_by_lti = [v["lti"] for v in agents.values()]

        num_with_lti = sum([1 for lti in agents_by_lti if lti == 0])
        num_without_lti = sum([1 for lti in agents_by_lti if lti == 1])

        prop_with_lti = num_with_lti / len(agents_by_lti)
        prop_without_lti = num_without_lti / len(agents_by_lti)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['With Long-Term Illness', 'Without Long-Term Illness']
        proportions = [prop_with_lti, prop_without_lti]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('Long-Term Illness')
        plt.ylabel('Proportion')
        plt.title('With LTI-Without LTI Split')
        plt.show()

        figureindex += 1
        # plt.figure(figureindex)

        lti_exp = [v for v in expected_values.values()]
        lti_act = [v for v in actual_values.values()]

        fem_lti_exp = [v for v in fem_expected_values.values()]
        fem_lti_act = [v for v in fem_actual_values.values()]

        # show male
        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Set the positions of the bars on the x-axis
        ages = np.arange(15, 101)

        # Create the expected counts bar chart
        rects1 = ax.bar(ages - bar_width/2, np.array(lti_exp)[:, 0], bar_width, label='Expected Counts (with LTI)')
        rects2 = ax.bar(ages + bar_width/2, np.array(lti_exp)[:, 1], bar_width, label='Expected Counts (without LTI)')

        # Create the actual counts bar chart
        rects3 = ax.bar(ages - bar_width/2, np.array(lti_act)[:, 0], bar_width/2, label='Actual Counts (with LTI)')
        rects3 = ax.bar(ages + bar_width/2, np.array(lti_act)[:, 1], bar_width/2, label='Actual Counts (without LTI)')

        # Add axis labels and a title
        ax.set_xlabel("Long-Term Illness")
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Long-Term Illness by Age (Male)')

        # Add a legend
        ax.legend()

        # Show the plot
        # plt.show()

        figureindex += 1
        # plt.figure(figureindex)

        # Show female
        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Create the expected counts bar chart
        rects1 = ax.bar(ages - bar_width/2, np.array(fem_lti_exp)[:, 0], bar_width, label='Expected Counts (with LTI)')
        rects2 = ax.bar(ages + bar_width/2, np.array(fem_lti_exp)[:, 1], bar_width, label='Expected Counts (without LTI)')

        # Create the actual counts bar chart
        rects3 = ax.bar(ages - bar_width/2, np.array(fem_lti_act)[:, 0], bar_width/2, label='Actual Counts (with LTI)')
        rects3 = ax.bar(ages + bar_width/2, np.array(fem_lti_act)[:, 1], bar_width/2, label='Actual Counts (without LTI)')

        # Add axis labels and a title
        ax.set_xlabel("Long-Term Illness")
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Long-Term Illness by Age (Female)')

        # Add a legend
        ax.legend()

        # Show the plot
        # plt.show()
    elif frequencydistributionname == "bmi_byage_distributions":
        agents_by_bmi = [v["bmi"] for v in agents.values()]

        num_normal = sum([1 for bmi in agents_by_bmi if bmi == 0])
        num_overweight = sum([1 for bmi in agents_by_bmi if bmi == 1])
        num_obese = sum([1 for bmi in agents_by_bmi if bmi == 2])

        prop_normal = num_normal / len(agents_by_bmi)
        prop_overweight = num_overweight / len(agents_by_bmi)
        prop_obese = num_obese / len(agents_by_bmi)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['Normal', 'Overweight', 'Obese']
        proportions = [prop_normal, prop_overweight, prop_obese]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('BMI')
        plt.ylabel('Proportion')
        plt.title('Normal-Overweight-Obese Split')
        plt.show()

        bmi_exp = [v for v in expected_values.values()]
        bmi_act = [v for v in actual_values.values()]

        fem_bmi_exp = [v for v in fem_expected_values.values()]
        fem_bmi_act = [v for v in fem_actual_values.values()]

        # Convert data to numpy arrays for easier manipulation
        expected_counts = np.array(bmi_exp)
        actual_counts = np.array(bmi_act)

        fem_expected_counts = np.array(fem_bmi_exp)
        fem_actual_counts = np.array(fem_bmi_act)

        normal_exp_counts = expected_counts[:, 0]
        overweight_exp_counts = expected_counts[:, 1]
        obese_exp_counts = expected_counts[:, 2]

        normal_act_counts = actual_counts[:, 0]
        overweight_act_counts = actual_counts[:, 1]
        obese_act_counts = actual_counts[:, 2]

        fem_normal_exp_counts = fem_expected_counts[:, 0]
        fem_overweight_exp_counts = fem_expected_counts[:, 1]
        fem_obese_exp_counts = fem_expected_counts[:, 2]

        fem_normal_act_counts = fem_actual_counts[:, 0]
        fem_overweight_act_counts = fem_actual_counts[:, 1]
        fem_obese_act_counts = fem_actual_counts[:, 2]

        # Calculate histogram with bin edges at multiples of 10
        bin_edges = np.arange(0, 101, 10)

        normal_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=normal_exp_counts)
        overweight_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=overweight_exp_counts)
        obese_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=obese_exp_counts)

        normal_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=normal_act_counts)
        overweight_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=overweight_act_counts)
        obese_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=obese_act_counts)

        fem_normal_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_normal_exp_counts)
        fem_overweight_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_overweight_exp_counts)
        fem_obese_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_obese_exp_counts)

        fem_normal_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_normal_act_counts)
        fem_overweight_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_overweight_act_counts)
        fem_obese_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=fem_obese_act_counts)

        compare_category_ageranges_expected_vs_actual(normal_exp_hist, normal_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Normal Category'", (10,7), 4)
        compare_category_ageranges_expected_vs_actual(overweight_exp_hist, overweight_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Overweight Category'", (10,7), 4)
        compare_category_ageranges_expected_vs_actual(obese_exp_hist, obese_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Obese Category'", (10,7), 4)

        compare_category_ageranges_expected_vs_actual(fem_normal_exp_hist, fem_normal_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Normal Category'", (10,7), 4)
        compare_category_ageranges_expected_vs_actual(fem_overweight_exp_hist, fem_overweight_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Overweight Category'", (10,7), 4)
        compare_category_ageranges_expected_vs_actual(fem_obese_exp_hist, fem_obese_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Obese Category'", (10,7), 4)
    elif frequencydistributionname == "employment_status_byage_distributions":
        agents_by_empstatus = [v["empstatus"] for v in agents.values()]

        num_employed = sum([1 for empstatus in agents_by_empstatus if empstatus == 0])
        num_unemployed = sum([1 for empstatus in agents_by_empstatus if empstatus == 1])
        num_inactive = sum([1 for empstatus in agents_by_empstatus if empstatus == 2])

        prop_employed = num_employed / len(agents_by_empstatus)
        prop_unemployed = num_unemployed / len(agents_by_empstatus)
        prop_inactive = num_inactive / len(agents_by_empstatus)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['Employed', 'Unemployed', 'Inactive']
        proportions = [prop_employed, prop_unemployed, prop_inactive]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('Employment Status')
        plt.ylabel('Proportion')
        plt.title('Employed-Unemployed-Inactive Split')
        plt.show()

        empstatus_exp = [v for v in expected_values.values()]
        empstatus_act = [v for v in actual_values.values()]

        fem_empstatus_exp = [v for v in fem_expected_values.values()]
        fem_empstatus_act = [v for v in fem_actual_values.values()]

        # Convert data to numpy arrays for easier manipulation
        expected_counts = np.array(empstatus_exp)
        actual_counts = np.array(empstatus_act)

        fem_expected_counts = np.array(fem_empstatus_exp)
        fem_actual_counts = np.array(fem_empstatus_act)

        emp_exp_counts = expected_counts[:, 0]
        unemp_exp_counts = expected_counts[:, 1]
        inact_exp_counts = expected_counts[:, 2]

        emp_act_counts = actual_counts[:, 0]
        unemp_act_counts = actual_counts[:, 1]
        inact_act_counts = actual_counts[:, 2]

        fem_emp_exp_counts = fem_expected_counts[:, 0]
        fem_unemp_exp_counts = fem_expected_counts[:, 1]
        fem_inact_exp_counts = fem_expected_counts[:, 2]

        fem_emp_act_counts = fem_actual_counts[:, 0]
        fem_unemp_act_counts = fem_actual_counts[:, 1]
        fem_inact_act_counts = fem_actual_counts[:, 2]

        # Calculate histogram with bin edges at multiples of 10
        bin_edges = np.array([15, 20, 30, 40, 50, 60, 70, 80, 90, 100])

        emp_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=emp_exp_counts)
        unemp_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=unemp_exp_counts)
        inact_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=inact_exp_counts)

        emp_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=emp_act_counts)
        unemp_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=unemp_act_counts)
        inact_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=inact_act_counts)

        fem_emp_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_emp_exp_counts)
        fem_unemp_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_unemp_exp_counts)
        fem_inact_exp_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_inact_exp_counts)

        fem_emp_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_emp_act_counts)
        fem_unemp_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_unemp_act_counts)
        fem_inact_act_hist, _ = np.histogram(np.arange(86), bins=bin_edges, weights=fem_inact_act_counts)

        compare_category_ageranges_expected_vs_actual(emp_exp_hist, emp_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Employed Category'", (10,7), 2, bin_edges[:-1])
        compare_category_ageranges_expected_vs_actual(unemp_exp_hist, unemp_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Unemployed Category'", (10,7), 2, bin_edges[:-1])
        compare_category_ageranges_expected_vs_actual(inact_exp_hist, inact_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Male Inactive Category'", (10,7), 2, bin_edges[:-1])

        compare_category_ageranges_expected_vs_actual(fem_emp_exp_hist, fem_emp_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Employed Category'", (10,7), 2, bin_edges[:-1])
        compare_category_ageranges_expected_vs_actual(fem_unemp_exp_hist, fem_unemp_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Unemployed Category'", (10,7), 2, bin_edges[:-1])
        compare_category_ageranges_expected_vs_actual(fem_inact_exp_hist, fem_inact_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for 'Female Inactive Category'", (10,7), 2, bin_edges[:-1])
    elif frequencydistributionname == "employment_byindustry_distributions":
        agents_industries = [v["empind"] for v in agentsemployed.values()]

        figureindex += 1

        # Set the positions of the bars on the x-axis
        industries = np.arange(1,22)

        plt.figure(figureindex, figsize=(10, 6))

        plt.hist(agents_industries, bins=21)
        
        plt.xlabel("Industry")
        plt.ylabel("Frequency")
        plt.title("Distribution of Industries")
        plt.xticks(industries)
        plt.show()

        figureindex += 1
        # plt.figure(figureindex)

        inds_exp = [v for v in expected_values.values()]
        inds_act = [v for v in actual_values.values()]

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Create the expected counts bar chart
        rects1 = ax.bar(industries - bar_width/2, inds_exp, bar_width, label='Expected Counts')

        # Create the actual counts bar chart
        rects2 = ax.bar(industries + bar_width/2, inds_act, bar_width, label='Actual Counts')

        # Add axis labels and a title
        ax.set_xlabel('Industry')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Industries')

        ax.set_xticks(industries)
        # ax.set_xticklabels(['15-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-100'], rotation=45, ha="right")

        # Add a legend
        ax.legend()

        # Show the plot
        # plt.show()
    elif fdval.frequencydistributionname == "employment_byindustry_ftpt_distributions":
        agents_by_empftpt = [v["empftpt"] for v in agentsemployed.values()]

        num_ft = sum([1 for empftpt in agents_by_empftpt if empftpt == 0])
        num_pt_ft = sum([1 for empftpt in agents_by_empftpt if empftpt == 1])
        num_pt_no_ft = sum([1 for empftpt in agents_by_empftpt if empftpt == 2])

        prop_ft = num_ft / len(agents_by_empftpt)
        prop_pt_ft = num_pt_ft / len(agents_by_empftpt)
        prop_pt_no_ft = num_pt_no_ft / len(agents_by_empftpt)

        figureindex += 1
        plt.figure(figureindex)
            
        # Create a bar chart
        labels = ['FT', 'PTwithFT', 'PTwithoutFT']
        proportions = [prop_ft, prop_pt_ft, prop_pt_no_ft]
        plt.bar(labels, proportions)

        # Add axis labels and a title
        plt.xlabel('Employment Type')
        plt.ylabel('Proportion')
        plt.title('FT-PTwithFT-PTwithoutFT Split')
        plt.show()

        # extra validation might be barchart comparison for expected and actual, for the 3 different categories, for males/females
        # consider creating equivalent to compare_category_ageranges_expected_vs_actual, for industries

        empftpt_exp = [v for v in expected_values.values()]
        empftpt_act = [v for v in actual_values.values()]

        fem_empftpt_exp = [v for v in fem_expected_values.values()]
        fem_empftpt_act = [v for v in fem_actual_values.values()]

        # Convert data to numpy arrays for easier manipulation
        expected_counts = np.array(empftpt_exp)
        actual_counts = np.array(empftpt_act)

        fem_expected_counts = np.array(fem_empftpt_exp)
        fem_actual_counts = np.array(fem_empftpt_act)

        ft_exp_counts = expected_counts[:, 0]
        ptft_exp_counts = expected_counts[:, 1]
        ptnoft_exp_counts = expected_counts[:, 2]

        ft_act_counts = actual_counts[:, 0]
        ptft_act_counts = actual_counts[:, 1]
        ptnoft_act_counts = actual_counts[:, 2]

        fem_ft_exp_counts = fem_expected_counts[:, 0]
        fem_ptft_exp_counts = fem_expected_counts[:, 1]
        fem_ptnoft_exp_counts = fem_expected_counts[:, 2]

        fem_ft_act_counts = fem_actual_counts[:, 0]
        fem_ptft_act_counts = fem_actual_counts[:, 1]
        fem_ptnoft_act_counts = fem_actual_counts[:, 2]

        # Calculate histogram with bin edges at multiples of 10
        bin_edges = np.arange(1,22)

        # ft_exp_hist, _ = np.histogram(ft_exp_counts, bins=bin_edges)
        # ptft_exp_hist, _ = np.histogram(ptft_exp_counts, bins=bin_edges)
        # ptnoft_exp_hist, _ = np.histogram(ptnoft_exp_counts, bins=bin_edges)

        # ft_act_hist, _ = np.histogram(ft_act_counts, bins=bin_edges)
        # ptft_act_hist, _ = np.histogram(ptft_act_counts, bins=bin_edges)
        # ptnoft_act_hist, _ = np.histogram(ptnoft_act_counts, bins=bin_edges)

        # fem_ft_exp_hist, _ = np.histogram(fem_ft_exp_counts, bins=bin_edges)
        # fem_ptft_exp_hist, _ = np.histogram(fem_ptft_exp_counts, bins=bin_edges)
        # fem_ptnoft_exp_hist, _ = np.histogram(fem_ptnoft_exp_counts, bins=bin_edges)

        # fem_ft_act_hist, _ = np.histogram(fem_ft_act_counts, bins=bin_edges)
        # fem_ptft_act_hist, _ = np.histogram(fem_ptft_act_counts, bins=bin_edges)
        # fem_ptnoft_act_hist, _ = np.histogram(fem_ptnoft_act_counts, bins=bin_edges)

        compare_category_industries_expected_vs_actual(ft_exp_counts, ft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Male FullTime Category'", (10,7), 0.35)
        compare_category_industries_expected_vs_actual(ptft_exp_counts, ptft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Male PartTime (FullTime) Category'", (10,7), 0.35)
        compare_category_industries_expected_vs_actual(ptnoft_exp_counts, ptnoft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Male PartTime (no FullTime) Category'", (10,7), 0.35)

        compare_category_industries_expected_vs_actual(fem_ft_exp_counts, fem_ft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Female FullTime Category'", (10,7), 0.35)
        compare_category_industries_expected_vs_actual(fem_ptft_exp_counts, fem_ptft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Female PartTime (FullTime) Category'", (10,7), 0.35)
        compare_category_industries_expected_vs_actual(fem_ptnoft_exp_counts, fem_ptnoft_act_counts, "Industries", "Counts", "Expected vs Actual Counts for 'Female PartTime (no FullTime) Category'", (10,7), 0.35)
    elif fdval.frequencydistributionname == "institutions_rates_by_age":
        # simple expected vs actual comparison (age based) should suffice
        # consider extending compare_category_ageranges_expected_vs_actual
        instrates_exp = [v for v in expected_values.values()]
        instrates_act = [v for v in actual_values.values()]

        # Convert data to numpy arrays for easier manipulation
        expected_counts = np.array(instrates_exp)
        actual_counts = np.array(instrates_act)

        # Calculate histogram with bin edges at multiples of 10
        bin_edges = np.array([0, 15, 25, 35, 45, 55, 65, 100])

        emp_exp_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=expected_counts)

        emp_act_hist, _ = np.histogram(np.arange(101), bins=bin_edges, weights=actual_counts)

        compare_category_ageranges_expected_vs_actual(emp_exp_hist, emp_act_hist, "Age Ranges", "Counts", "Expected vs Actual Counts for Institution Rates by Age", (10,7), 4, bin_edges[:-1])
    elif fdval.frequencydistributionname == "institutions_type_distribution":
        # simple expected vs actual comparison (institution type based) should suffice
        # consider copying employment_byindustry_distributions case
        institutions_by_type = [v["insttypeid"] for v in institutions]

        figureindex += 1

        # Set the positions of the bars on the x-axis
        institutiontypes = np.arange(1,12) # 11 institution types

        plt.figure(figureindex, figsize=(10, 6))

        plt.hist(institutions_by_type, bins=11)
        
        plt.xlabel("Institution Types")
        plt.ylabel("Frequency")
        plt.title("Distribution of Institution Types")
        plt.xticks(institutiontypes)
        plt.show()

        figureindex += 1
        # plt.figure(figureindex)

        insttypes_exp = [v for v in expected_values.values()]
        insttypes_act = [v for v in actual_values.values()]

        # Create a figure and axes object
        fig, ax = plt.subplots()

        # Set the width of the bars
        bar_width = 0.35

        # Create the expected counts bar chart
        rects1 = ax.bar(institutiontypes - bar_width/2, insttypes_exp, bar_width, label='Expected Counts')

        # Create the actual counts bar chart
        rects2 = ax.bar(institutiontypes + bar_width/2, insttypes_act, bar_width, label='Actual Counts')

        # Add axis labels and a title
        ax.set_xlabel('Institution Types')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Institution Types')

        ax.set_xticks(institutiontypes)

        # Add a legend
        ax.legend()
