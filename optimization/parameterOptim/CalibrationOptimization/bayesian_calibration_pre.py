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
    def __init__(
        self,
        keys:np.ndarray = None,
        mean_values:np.ndarray = None,
        stds:np.ndarray = None,
        sampling_number:int = 101,
        time_point:np.ndarray = None,
        online: bool = False
    ) -> None:
        self.time_point = time_point
        self.sampling_number = sampling_number
        self.online = online
        params_info = {
            keys[i]: {'range': np.linspace(mean_values[i]-2*stds[i], mean_values[i]+2*stds[i],sampling_number),
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

    def bayesian_calibration(self, model:UserModel, op:Optimization, dr:DomainReduction):
        posteriors = [self.joint_prior.flatten()]
        max_params_over_time = [[info['mu'] for info in self.params_info.values()]]
        destination_name = 'Bayesian_calibration'
        if not os.path.exists(destination_name):
            os.makedirs(destination_name)
        else:
            shutil.rmtree(destination_name)
            os.makedirs(destination_name)
        # initial_values = np.array([info['mu'] for info in self.params_info.values()])
        optim_folder = 0
        optimized_params = [[info['mu'] for info in self.params_info.values()]]
        bounds_reducted = [op.bounds_dr]
        for i in range(1,len(self.time_point)):
            print(f'current time:{self.time_point[i]}')
            if self.online == True:
                t_0 = self.time_point[i-1]
            else:
                t_0 = 0
            sciantix_folder_path = model._independent_sciantix_folder(destination_name, optim_folder,t_0, self.time_point[i])
            observed = model._exp(time_point=self.time_point[i])
            model_values = []
            params_combination = copy.deepcopy(self.params_combination)
            for combination in params_combination:
                params = {key:value for key, value in zip(self.params_info.keys(),combination)}
                model_value = model._sciantix(sciantix_folder_path, params)[2]
                model_values.append(model_value)
            likelihood = norm.pdf(observed[1], loc = model_values, scale = observed[2])
            posterior = self.bayesian_update(posteriors[-1], likelihood)
            posteriors.append(posterior)

            reshaped_posterior = posterior.reshape(*[len(self.params_info[key]['range']) for key in self.params_info.keys()])
            # print(f'reshaped_posterior: {reshaped_posterior}')
            max_index = np.unravel_index(np.argmax(reshaped_posterior), reshaped_posterior.shape)
            # print(max_index)
            max_params = [self.params_info[key]['range'][max_index[i]] for i, key in enumerate(self.params_info.keys())]
            # print(max_params)
            max_params_over_time.append(max_params)
            
            params_at_max_prob = np.array(max_params_over_time)
            print(f'calibrated params(max prob): {params_at_max_prob}')
            with open('params_at_max_prob.txt', 'w') as file:
                file.writelines('\t'.join(str(item) for item in row) + '\n' for row in  params_at_max_prob[:-1])
                file.write('\t'.join(str(item) for item in params_at_max_prob[-1]))
            
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

    @staticmethod
    def bayesian_update(prior, likelihood):
        posterior = prior * likelihood
        return posterior/np.sum(posterior)

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