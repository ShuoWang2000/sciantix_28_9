import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


params_at_max_prob = np.genfromtxt('params_at_max_prob.txt')
params_optimized = np.genfromtxt('params_optimized.txt')
calibration_data = np.genfromtxt('calibration_data.txt')
optim_exp = np.genfromtxt('optim_data.txt')

# params = params_optimized.shape[1]
params = params_at_max_prob.ndim
if params == 1:
    plt.plot(optim_exp[:,0], params_at_max_prob, color = 'b', label = 'bayesian_calibration')
    plt.scatter(optim_exp[:,0], params_at_max_prob, color = 'b', marker='o')
    plt.plot(optim_exp[:,0], params_optimized, color = 'g', label = 'optimization')
    plt.scatter(optim_exp[:,0], params_optimized, color = 'g', marker='o')
    plt.xlabel('time / h')
    plt.ylabel('ln(sf) / -')
    plt.legend()
    plt.title('Logarithm scaling factor')
    plt.show()
else:
    for i in range(params):
        plt.plot(optim_exp[:,0], params_at_max_prob[:,i], color = 'b', label = 'bayesian_calibration')
        plt.scatter(optim_exp[:,0], params_at_max_prob[:,i], color = 'b', marker='o')
        plt.plot(optim_exp[:,0], params_optimized[:,i], color = 'g', label = 'optimization')
        plt.scatter(optim_exp[:,0], params_optimized[:,i], color = 'g', marker='o')
        plt.xlabel('time / h')
        plt.ylabel('ln(sf) / -')
        plt.legend()
        plt.title('Logarithm scaling factor')
        plt.show()


# plt.plot(optim_exp[:,0], optim_exp[:,1], color = 'g', label = 'optimization')
plt.scatter(optim_exp[:,0], optim_exp[:,1], color = 'g', marker='x', label = 'optimization')
# plt.plot(optim_exp[:,0], optim_exp[:,2], color = 'r', label = 'experimental')
plt.scatter(optim_exp[:,0], optim_exp[:,2], facecolor = 'none', edgecolors='r',marker='o', label = 'experimental')
plt.scatter(optim_exp[1:,0], calibration_data[:,2], color = 'black', marker='x', label = 'calibration')
plt.xlabel('time / h')
plt.ylabel('fraction release / -')
plt.legend()
plt.title('Fraction release')
plt.show()


priors_over_time = []
with open('priors_over_time.txt', 'r') as file:
    for line in file:
        # Banish the brackets to the shadow realm
        cleaned_line = line.replace('[', '').replace(']', '')
        # Split the numbers, now free from their bracketed confines
        number_strings = cleaned_line.split()

        # Convert the string representations into the true form of floating-point numbers
        numbers = [float(num) for num in number_strings]
        priors_over_time.append(numbers)

points_over_time = []
with open('points_over_time.txt', 'r') as file:
    for line in file:
        # Banish the brackets to the shadow realm
        cleaned_line = line.replace('[', '').replace(']', '')
        # Split the numbers, now free from their bracketed confines
        number_strings = cleaned_line.split()

        # Convert the string representations into the true form of floating-point numbers
        numbers = [float(num) for num in number_strings]
        points_over_time.append(numbers)

i = 0
while i < 10:
    plt.scatter(points_over_time[i], priors_over_time[i]/np.max(priors_over_time[i]))
    i = i + 1
# plt.scatter(points_over_time[11], priors_over_time[11]/np.max(priors_over_time[0]))
# # plt.show()

# plt.scatter(points_over_time[21], priors_over_time[21]/np.max(priors_over_time[0]))
# # plt.show()

# plt.scatter(points_over_time[31], priors_over_time[31]/np.max(priors_over_time[0]))
# # plt.show()

# plt.scatter(points_over_time[71], priors_over_time[71]/np.max(priors_over_time[0]))
# # plt.show()

# plt.scatter(points_over_time[99], priors_over_time[99]/np.max(priors_over_time[0]))
plt.show()