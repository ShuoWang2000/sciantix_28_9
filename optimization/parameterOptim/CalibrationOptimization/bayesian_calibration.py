import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from user_model import UserModel
import os
from itertools import product
import shutil
import copy


class BayesianCalibration:
    def __init__(
        self,
        keys:np.ndarray = None,
        mean_values:np.ndarray = None,
        stds:np.ndarray = None,
        sampling_number:int = 101,
        time_point:np.ndarray = None
    ) -> None:
        self.time_point = time_point
        self.sampling_number = sampling_number

        params_info = {
            keys[i]: {'range': np.linspace(mean_values[i]-3*stds[i], mean_values[i]+3*stds[i],sampling_number),
                      'mu': mean_values[i],
                      'sigma':stds[i]
                      } for i in range(len(keys))
        }
        self.params_info = {key : params_info[key] for key in sorted(params_info)}
        params_grid = np.meshgrid(*[info['range'] for info in self.params_info.values()], indexing = 'ij')
        self.params_grid = {key : grid for key, grid in zip(self.params_info.keys(), params_grid)}
        priors = [norm.pdf(grid, loc = info['mu'], scale = info['sigma']) for grid, info in zip(self.params_grid.values(), self.params_info.values())]
        joint_prior = np.ones(priors[0].shape)
        for prior in priors:
            joint_prior *= prior
        self.joint_prior = joint_prior/np.sum(joint_prior)

        self.params_combination = product(*[info['range'] for info in self.params_info.values()])

    def bayesian_calibration(self, model:UserModel):
        posteriors = [self.joint_prior.flatten()]
        max_params_over_time = [[info['mu'] for info in self.params_info.values()]]
        destination_name = 'Bayesian_calibration'
        if not os.path.exists(destination_name):
            os.makedirs('Bayesian_calibration')
        else:
            shutil.rmtree(destination_name)
            os.makedirs(destination_name)
        for i in range(1,len(self.time_point)):
            observed = model._exp(time_point=self.time_point[i])
            model_values = []
            params_combination = copy.deepcopy(self.params_combination)
            for combination in params_combination:

                params = {key:value for key, value in zip(self.params_grid.keys(),combination)}
                model_value = model._sciantix(destination_name,0,self.time_point[i],params)[2]

                model_values.append(model_value)

            likelihood = norm.pdf(observed[1], loc = model_values, scale = observed[2])
            posterior = self.bayesian_update(posteriors[-1], likelihood)
            posteriors.append(posterior)

            reshaped_posterior = posterior.reshape(*[len(self.params_info[key]['range']) for key in self.params_info.keys()])
            max_index = np.unravel_index(np.argmax(reshaped_posterior), reshaped_posterior.shape)
            max_params = [self.params_info[key]['range'][max_index[i]] for i, key in enumerate(self.params_info.keys())]
            max_params_over_time.append(max_params)
            print(self.time_point[i])
            params_at_max_prob = np.array(max_params_over_time)
            with open('params_at_max_prob.txt', 'w') as file:
                file.writelines('\t'.join(str(item) for item in row) + '\n' for row in  params_at_max_prob[:-1])
                file.write('\t'.join(str(item) for item in params_at_max_prob[-1]))
        self.max_params_over_time = max_params_over_time
        
    def do_plot(self, model:UserModel):
        plt.figure(figsize=(12,6))
        for i, key in enumerate(self.params_info.keys()):
            plt.plot(self.time_point, [params[i] for params in self.max_params_over_time], label = f"Evolution of {key}", marker = 'o')
        plt.xlabel('Time')
        plt.ylabel('Parameter Value')
        plt.title('Evolution of Parameters in Maximum Posterior Probability')
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def bayesian_update(prior, likelihood):
        posterior = prior * likelihood
        return posterior/np.sum(posterior)
