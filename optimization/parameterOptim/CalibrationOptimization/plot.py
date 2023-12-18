import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


params_at_max_prob = np.genfromtxt('params_at_max_prob.txt')
params_optimized = np.genfromtxt('params_optimized.txt')

optim_exp = np.genfromtxt('optim_data.txt')

# params = params_optimized.shape[1]
params = 1
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
plt.xlabel('time / h')
plt.ylabel('fraction release / -')
plt.legend()
plt.title('Fraction release')
plt.show()