import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
from user_model import UserModel
import os, shutil, copy
from itertools import product
from optimization import Optimization
from domain_reduction import DomainReduction
from data_generation import DataGeneration


class BayesianCalibration:
    def __init__(self, keys, mean_values, stds, sampling_number, time_point, online):
        self.time_point = time_point
        self.sampling_number = sampling_number
        self.online = online
        params_info = {
            keys[i]: {
                'range':sorted(np.random.normal(mean_values[i], stds[i], sampling_number)),
                'mu': mean_values[i],
                'sigma': stds[i]
            } for i in range(len(keys))
        }
        self.params_info = {key:params_info[key] for key in sorted(params_info)}
        params_grid = np.meshgrid(*[info['range'] for info in self.params_info.values()], indexing = 'ij')
        self.params_grid = {key : grid for key, grid in zip(self.params_info.keys(), params_grid)}
        priors = [norm.pdf(grid, loc = info['mu'], scale = info['sigma']) for grid, info in zip(self.params_grid.values(), self.params_info.values())]
        joint_prior = np.ones(priors[0].shape)
        for prior in priors:
            joint_prior *= prior
        self.joint_prior = joint_prior/np.sum(joint_prior)
        
        params_combination = product(*[info['range'] for info in self.params_info.values()])

        self.points = np.array([[point for point in combination] for combination in params_combination])

    
    def update_joint_prior(self, data_generator: DataGeneration):
        """Update the joint prior based on current parameter ranges."""
        return data_generator.data_generated

    def update_parameter_sampling(self, data_generator:DataGeneration):
        return data_generator.probabilities_generated


    def bayesian_calibration(self, model, op, dr):
        self.setup_directory('Bayesian_calibration')
        max_params_over_time, optimized_params = [[info['mu'] for info in self.params_info.values()]], [[info['mu'] for info in self.params_info.values()]]
        priors_over_time = [self.joint_prior.flatten()]
        posteriors_over_time = []
        points_over_time = [self.points]
        bounds_reducted = [op.bounds_dr]
        optim_folder = 0
        for i in range(1, len(self.time_point)):
            print(f'current time: {self.time_point[i]}')
            t_0 = self.time_point[i-1] if self.online else 0
            sciantix_folder_path = model._independent_sciantix_folder('Bayesian_calibration',optim_folder, t_0, self.time_point[i])
            observed = model._exp(time_point=self.time_point[i])

            model_values = self.compute_model_values(model, sciantix_folder_path, points_over_time[-1])
            likelihood = norm.pdf(observed[1], loc=model_values, scale=observed[2])
            posterior = self.bayesian_update(priors_over_time[-1], likelihood)
            posteriors_over_time.append(posterior)

            max_params = self.find_max_params(posterior, points_over_time[-1])
            max_params_over_time.append(max_params)
            self.write_to_file('params_at_max_prob.txt', max_params_over_time)
            print(f'calibrated params(max prob): \n {max_params_over_time}')

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

            data_generator = DataGeneration(points_over_time[-1], posteriors_over_time[-1],501)
            new_points = data_generator.data_generated
            new_probabilities = data_generator.probabilities_generated
            
            priors_over_time.append(new_probabilities)
            points_over_time.append(new_points)
        
        self.max_params_over_time = max_params_over_time
        self.optimized_params = optimized_params
    
    def setup_directory(self, dirname):
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)

    def compute_model_values(self, model, folder_path, points):
        model_values = []

        for point in points:
            params = {key: value for key, value in zip(self.params_info.keys(), point)}
            model_value = model._sciantix(folder_path, params)[2]
            model_values.append(model_value)
        return model_values

    def find_max_params(self, posterior, points):
        max_index = np.argmax(posterior)
        point = points[max_index]
        return point

    def write_to_file(self, filename, data):
        with open(filename, 'w') as file:
            for row in data[:-1]:
                file.write('\t'.join(map(str, row)) + '\n')
            file.write('\t'.join(map(str, data[-1])))

    @staticmethod
    def bayesian_update(prior, likelihood):
        posterior = prior * likelihood
        return posterior / np.sum(posterior)
