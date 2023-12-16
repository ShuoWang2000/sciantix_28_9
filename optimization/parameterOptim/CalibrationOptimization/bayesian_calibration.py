import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from user_model import UserModel
import os, shutil, copy
from itertools import product
from optimization import Optimization
from domain_reduction import DomainReduction
class BayesianCalibration:
    def __init__(self, keys, mean_values, stds, sampling_number, time_point, online):
        self.time_point = time_point
        self.sampling_number = sampling_number
        self.online = online
        self.params_info = {
            keys[i]: {
                'range': np.random.normal(mean_values[i], stds[i], sampling_number),
                'mu': mean_values[i],
                'sigma': stds[i]
            } for i in range(len(keys))
        }
        self.update_joint_prior()

    def update_joint_prior(self):
        """Update the joint prior based on current parameter ranges."""
        self.params_combination = product(*[info['range'] for info in self.params_info.values()])
        priors = [norm.pdf(grid, loc=info['mu'], scale=info['sigma']) 
                  for grid, info in zip(self.params_combination, self.params_info.values())]
        joint_prior = np.ones(len(priors))
        for prior in priors:
            joint_prior *= prior
        self.joint_prior = joint_prior / np.sum(joint_prior)

    def update_parameter_sampling(self, posterior):
        """Update the parameter range based on the current posterior distribution."""

        # Example: Update the parameter range to values around the maximum posterior
        reshaped_posterior = posterior.reshape(*[len(info['range']) for info in self.params_info.values()])
        max_index = np.unravel_index(np.argmax(reshaped_posterior), reshaped_posterior.shape)

        for i, key in enumerate(self.params_info):
            max_value = self.params_info[key]['range'][max_index[i]]
            # Update the range around the max_value
            # This is a simplistic approach; you'll need to adjust this logic to fit your model
            updated_range = np.random.normal(max_value, self.params_info[key]['sigma'], self.sampling_number)
            self.params_info[key]['range'] = updated_range
    
    def bayesian_calibration(self, model, op, dr):
        self.setup_directory('Bayesian_calibration')
        posteriors, max_params_over_time, optimized_params = [self.joint_prior.flatten()], [[info['mu'] for info in self.params_info.values()]], [[info['mu'] for info in self.params_info.values()]]
        bounds_reducted = [op.bounds_dr]
        optim_folder = 0
        for i in range(1, len(self.time_point)):
            print(f'current time: {self.time_point[i]}')
            t_0 = self.time_point[i-1] if self.online else 0
            sciantix_folder_path = model._independent_sciantix_folder('Bayesian_calibration',optim_folder, t_0, self.time_point[i])
            observed = model._exp(time_point=self.time_point[i])

            model_values = self.compute_model_values(model, sciantix_folder_path)
            likelihood = norm.pdf(observed[1], loc=model_values, scale=observed[2])
            posterior = self.bayesian_update(posteriors[-1], likelihood)
            posteriors.append(posterior)

            # Update the sampling based on the new posterior
            self.update_parameter_sampling(posteriors[-1])
            self.update_joint_prior()  # Also update the joint prior with new ranges

            max_params = self.find_max_params(posterior)
            max_params_over_time.append(max_params)
            self.write_to_file('params_at_max_prob.txt', max_params_over_time)
            
            optimize_result = op.optimize(model,t_0,self.time_point[i],optimized_params[-1],bounds_reducted[-1])
            optim_folder = op.optim_folder

            for key, value in optimize_result.items():
                if 'pre exponential' in key:
                    optimize_result[key] = np.log(value)
            
            optimized_param = [optimize_result[key] for key in self.params_info.keys()]
            optimized_params.append(optimized_param)
            params_optimized = np.array(optimized_params)
            print(f'optimized params: {params_optimized}')
            with open('params_optimized.txt','w') as file:
                file.writelines('\t'.join(str(item) for item in row) + '\n' for row in params_optimized[:-1])
                file.write('\t'.join(str(item) for item in params_optimized[-1]))
            
            bound = dr.transform(op)
            bounds_reducted.append(bound)

        self.max_params_over_time = max_params_over_time
        self.optimized_params = optimized_params
    
    def setup_directory(self, dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

    def compute_model_values(self, model, folder_path):
        model_values = []
        for combination in self.params_combination:
            params = {key: value for key, value in zip(self.params_info.keys(), combination)}
            model_value = model._sciantix(folder_path, params)[2]
            model_values.append(model_value)
        return model_values

    def find_max_params(self, posterior):
        reshaped_posterior = posterior.reshape(*[len(info['range']) for info in self.params_info.values()])
        max_index = np.unravel_index(np.argmax(reshaped_posterior), reshaped_posterior.shape)
        return [self.params_info[key]['range'][max_index[i]] for i, key in enumerate(self.params_info.keys())]

    def write_to_file(self, filename, data):
        with open(filename, 'w') as file:
            for row in data[:-1]:
                file.write('\t'.join(map(str, row)) + '\n')
            file.write('\t'.join(map(str, data[-1])))

    @staticmethod
    def bayesian_update(prior, likelihood):
        posterior = prior * likelihood
        return posterior / np.sum(posterior)



    def do_plot(self):
        plt.figure(figsize=(12,6))
        for i, key in enumerate(self.params_info.keys()):
            plt.plot(self.time_point, [params[i] for params in self.max_params_over_time], label = f"Calibration of {key}", marker = 'o')
            plt.plot(self.time_point, [params[i] for params in self.optimized_params], label = f"Optimization of {key}", marker = 'x')
        plt.xlabel('Time')
        plt.ylabel('Parameter Value')
        plt.title('Evolution of Parameters in Maximum Posterior Probability')
        plt.legend()
        plt.grid(True)
        plt.show()
