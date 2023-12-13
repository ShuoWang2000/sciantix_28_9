import numpy as np
import matplotlib.pyplot as plt
from user_model import UserModel  # Make sure to import your classes
from optimization import Optimization
from domain_reduction import DomainReduction
from bayesian_calibration import BayesianCalibration
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
def main():

    # Initialize the UserModel with appropriate parameters
    model = UserModel(
        case_name='test_Talip2014_1600K',
        params=np.array(['helium diffusivity pre exponential', 'helium diffusivity activation energy']),
        params_initial_values=np.array([0,1.0]),
        params_stds=np.array([1.526,0.1])
    ) 
    params_info = model.params_info
    keys = np.array([key for key in params_info.keys()])
    initial_values = np.array([info['mu'] for info in params_info.values()])
    stds = np.array([info['sigma'] for info in params_info.values()])

    
    time_points = np.linspace(0, max(model.time_exp), 21)
    # Perform Bayesian Calibration
    bc = BayesianCalibration(keys=keys,mean_values=initial_values, stds=stds,sampling_number=21, time_point=time_points)
    bc.bayesian_calibration(model)
    # bc.do_plot(model)
    # Set up the Optimization
#     optim_online = Optimization(kind='online', method='som', keys=keys, initial_values=initial_values, stds=stds)
#     optim_offline = Optimization(kind='offline', method='som', keys=keys, initial_values=initial_values, stds=stds)

#     # Initialize Domain Reduction
#     domain_reduction_online = DomainReduction()
#     domain_reduction_offline = DomainReduction()
#     domain_reduction_online.initialize(optim_online)
#     domain_reduction_offline.initialize(optim_offline)

#     # Arrays for storing results
#     time_points = model.t
#     # real_output = np.array([model.black_box_function(model.n0_true,model.lambda_true, t, model.measure_std_relative) for t in time_points])  # Fill in the black_box_function args
#     real_output = model.real_output
#     online_output = np.zeros_like(time_points)
#     offline_output = np.zeros_like(time_points)
#     online_params = np.ones((len(time_points), len(keys)))
#     offline_params = np.ones_like(online_params)
#     updated_domain_online = {key:(bound[0], bound[1]) for key, bound in zip(keys, domain_reduction_online.bounds_original)}
#     updated_domain_offline = {key:(bound[0], bound[1]) for key, bound in zip(keys, domain_reduction_offline.bounds_original)}

#     # Main optimization loop
#     for i in range(1,len(time_points)):
#         t = time_points[i]
#         # Online Optimization
#         current_values_online = online_params[i - 1]
#         optimized_params_online = optim_online.optimize(model, t, current_values_online, updated_domain_online)
#         online_params[i] = [optimized_params_online[key] for key in keys]
#         online_output[i] = model.numerical_model(optimized_params_online['sf_n0'], optimized_params_online['sf_lambda'], model.n0_model, model.lambda_model, t)
#         updated_domain_online = domain_reduction_online.transform(optim_online)

#         # Offline Optimization
#         current_values_offline = offline_params[i - 1]
#         optimized_params_offline = optim_offline.optimize(model, t, current_values_offline, updated_domain_offline)
#         offline_params[i] = [optimized_params_offline[key] for key in keys]
#         offline_output[i] = model.numerical_model(optimized_params_offline['sf_n0'], optimized_params_offline['sf_lambda'], model.n0_model, model.lambda_model, t)
#         updated_domain_offline = domain_reduction_offline.transform(optim_offline)

#     # Visualization of results
#     t_tild = model.t_tild
#     calibrated_params = bc.max_params_over_time
#     plot_results(t_tild, real_output, online_output, offline_output, online_params, offline_params, calibrated_params, trues, keys)
    
# def plot_results(time_points, real_output, online_output, offline_output, online_params, offline_params, calibrated_params,true_params, keys):
#     # Plotting the outputs
#     plt.figure(figsize=(12, 6))
#     plt.scatter(time_points, real_output, label='Real Output', color='green')
#     plt.plot(time_points, online_output, label='Online Optimized Output', color='blue')
#     plt.plot(time_points, offline_output, label='Offline Optimized Output', color='red')
#     plt.xlabel('Time')
#     plt.ylabel('Output')
#     plt.title('Model Outputs Comparison')
#     plt.legend()
#     plt.show()

#     # Plotting the parameters
#     for i, key in enumerate(keys):
#         plt.figure(figsize=(12, 6))
#         plt.plot(time_points, online_params[:, i], label=f'Online Optimized {key}', color='blue')
#         plt.plot(time_points, offline_params[:, i], label=f'Offline Optimized {key}', color='red')
#         plt.plot(time_points,[params[i] for params in calibrated_params], label=f'Bayesican Calibrated {key}')
#         plt.plot([0,1],[true_params[i],true_params[i]], color='green', linestyle='--', label=f'True {key}')
#         plt.xlabel('Time')
#         plt.ylabel(f'Parameter Value ({key})')
#         plt.title(f'Parameter {key} Optimization Comparison')
#         plt.legend()
#         plt.show()

if __name__ == "__main__":
    main()
