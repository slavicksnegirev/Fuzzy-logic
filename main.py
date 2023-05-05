import math

import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import matplotlib.pyplot as plt

x0 = 0.4
y0 = 0.7

def get_degrees_of_truth(i, key, input):
    x1 = math.floor(input)
    x2 = math.ceil(input)

    for index in range(len(i.universe)-1):
        y1 = i[key].mf[index]
        y2 = i[key].mf[index+1]
        if x1 == i.universe[index] and x2 == i.universe[index+1]:
            return (input-x1)/(x2-x1)*(y2-y1)+y1


def sub_plot(i1, i2, x0_key_list, y0_key_list):
    fig, ax = plt.subplots(nrows=4, ncols=3)

    for i in range(4):
        for j in range(3):
            ax[i, j].set_ylim(0, 1.01)
            ax[i, j].set_xlim(i1.universe.min(), i1.universe.max())

            # Turn off top/right axes
            ax[i, j].spines['top'].set_visible(False)
            ax[i, j].spines['right'].set_visible(False)
            ax[i, j].get_xaxis().tick_bottom()
            ax[i, j].get_yaxis().tick_left()

            # Ticks outside the axes
            ax[i, j].tick_params(direction='out')

    # Make the plots
    for i in range(len(x0_key_list)):
        ax[i, 0].plot(i1.universe, i1[x0_key_list[i]].mf)
        ax[i, 1].plot(i2.universe, i2[y0_key_list[i]].mf)

        ax[i, 0].plot(x0, get_degrees_of_truth(i1, x0_key_list[i], x0), '.')
        ax[i, 1].plot(y0, get_degrees_of_truth(i2, y0_key_list[i], y0), '.')

        # Label the axes
        ax[i, 0].set_title(x0_key_list[i])
        ax[i, 1].set_title(y0_key_list[i])

# Set variable range
i1_range = np.arange(-3, 4, 1, np.float32)  # Input 1
i2_range = np.arange(-2, 3, 1, np.float32)  # Input 2
o_range = np.arange(-6, 7, 1, np.float32)   # Output

# Create fuzzy control variables
i1 = ctrl.Antecedent(i1_range, 'I1')      # Input 1
i2 = ctrl.Antecedent(i2_range, 'I2')      # Input 2
o = ctrl.Consequent(o_range, 'O')         # Output

# Fuzzy sets and their membership functions are defined
i1['I1NH'] = fuzz.zmf(i1_range, -2, -1)
i1['I1NL'] = fuzz.trimf(i1_range, [-2, -1, 0])
i1['I1Z'] = fuzz.trimf(i1_range, [-1, 0, 1])
i1['I1PL'] = fuzz.trimf(i1_range, [0, 1, 2])
i1['I1PH'] = fuzz.smf(i1_range, 1, 2)

i2['I2N'] = fuzz.zmf(i2_range, -1, 0)
i2['I2Z'] = fuzz.trimf(i2_range, [-1, 0, 1])
i2['I2P'] = fuzz.smf(i2_range, 0, 1)

o['ONH'] = fuzz.zmf(o_range, -4, -2)
o['ONL'] = fuzz.trimf(o_range, [-4, -2, 0])
o['OZ'] = fuzz.trimf(o_range, [-2, 0, 2])
o['OPL'] = fuzz.trimf(o_range, [0, 2, 4])
o['OPH'] = fuzz.smf(o_range, 2, 4)

o.defuzzify_method = 'mom'

# Reference value setting visualization
# i1.view(), i2.view(), o.view()

x0_key_list = []
y0_key_list = []

for key, term in i1.terms.items():
    if 1.0 == term.mf[list(i1.universe).index(math.floor(x0))]:
        x0_key_list.append(key)
    elif 1.0 == term.mf[list(i1.universe).index(math.ceil(x0))]:
        x0_key_list.append(key)

for key, term in i2.terms.items():
    if 1.0 == term.mf[list(i2.universe).index(math.floor(y0))]:
        y0_key_list.append(key)
    elif 1.0 == term.mf[list(i2.universe).index(math.ceil(y0))]:
        y0_key_list.append(key)

x0_key_list.insert(1, x0_key_list[1])
x0_key_list.insert(0, x0_key_list[0])
y0_key_list.insert(0, y0_key_list[1])
y0_key_list.insert(0, y0_key_list[1])

# print(i1.universe)
# for key, term in i1.terms.items():
#     print(term.mf)



# plt.plot(i1.universe, i1[x0_mf_list[0]].mf)
sub_plot(i1, i2, x0_key_list, y0_key_list)

rule1 = ctrl.Rule(antecedent= (i1['I1NH'] & i2['I2N']) |
                              (i1['I1NL'] & i2['I2N']),
                  consequent=o['ONH'], label='ONH')

rule2 = ctrl.Rule(antecedent= (i1['I1NH'] & i2['I2Z']) |
                              (i1['I1NL'] & i2['I2Z']) |
                              (i1['I1Z'] & i2['I2N']),
                  consequent=o['ONL'], label='ONL')

rule3 = ctrl.Rule(antecedent= (i1['I1NH'] & i2['I2P']) |
                              (i1['I1NL'] & i2['I2P']) |
                              (i1['I1Z'] & i2['I2Z']) |
                              (i1['I1PL'] & i2['I2N']) |
                              (i1['I1PH'] & i2['I2N']),
                  consequent=o['OZ'], label='OZ')

rule4 = ctrl.Rule(antecedent= (i1['I1Z'] & i2['I2P']) |
                              (i1['I1PL'] & i2['I2Z']) |
                              (i1['I1PH'] & i2['I2Z']),
                  consequent=o['OPL'], label='OPL')

rule5 = ctrl.Rule(antecedent= (i1['I1PL'] & i2['I2P']) |
                              (i1['I1PH'] & i2['I2P']),
                  consequent=o['OPH'], label='OPH')

# System and running environment initialization
system = ctrl.ControlSystem(rules=[rule1, rule2, rule3, rule4, rule5])
sim = ctrl.ControlSystemSimulation(system)

# Operating system
sim.input['I1'] = x0
sim.input['I2'] = y0
sim.compute()
output = sim.output['O']

# Printout results
print(output)

# Draw O
o.view(sim=sim)
plt.subplots_adjust(wspace=0.5, hspace=1.0)
plt.show()
