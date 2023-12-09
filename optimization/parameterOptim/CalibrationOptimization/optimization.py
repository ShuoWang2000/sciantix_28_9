import numpy as np
from scipy.optimize import Bounds, minimize
from bayes_opt import BayesianOptimization, UtilityFunction, SequentialDomainReductionTransformer

class Optimization:
    """
    Optimization class for adjusting scale factors (sf) in a numerical model.

    Attributes:
        kind (str): Type of optimization ('online' or 'offline').
        method (str): Optimization algorithm ('som' for scipy.optimize.minimize or 'dr' for domain reduction).
        params_info (dict): Stores parameter information, including bounds and initial values.
        bounds_global (Bounds or dict): Global bounds for the optimization parameters.
    """

    def __init__(self, kind='online', method='som', keys=None, initial_values=None, stds=None):
        """
        Initializes the Optimization object.

        Args:
            kind (str): The optimization kind, either 'online' or 'offline'.
            method (str): The optimization method, either 'som' or 'dr'.
            keys (np.ndarray): Names of the parameters to optimize.
            initial_values (np.ndarray): Initial guess values for the parameters.
            stds (np.ndarray): Standard deviations for the parameters, used to set bounds.
        """
        if keys is None or initial_values is None or stds is None:
            raise ValueError("keys, initial_values, and stds must not be None.")

        self.kind = kind
        self.method = method
        self.params_info = self._create_params_info(keys, initial_values, stds)
        self.bounds_global = self._create_bounds_global(self.method)
        self.bounds_dr = self._create_bounds_global('dr')

    def _create_params_info(self, keys, initial_values, stds):
        bounds = np.vstack((initial_values - 4 * stds, initial_values + 4 * stds)).T
        params_info = {key: {'value': val, 'bounds': bound} for key, val, bound in zip(keys, initial_values, bounds)}
        return {key:params_info[key] for key in sorted(params_info)}

    def _create_bounds_global(self, method):
        if method == 'som':
            return Bounds([info['bounds'][0] for info in self.params_info.values()],
                          [info['bounds'][1] for info in self.params_info.values()])
        elif method == 'dr':
            return {key: info['bounds'] for key, info in self.params_info.items()}
        else:
            raise ValueError("Invalid method specified")

    def optimize(self, model, current_x, initial_values, initial_bounds_dr):
        """
        Performs optimization using the specified method.

        Args:
            model (UserModel): The model to optimize.
            current_x (float): The current time point for optimization.
            initial_values (np.ndarray): Initial values for the optimization.
            initial_bounds_dr (dict): Initial domain reduction bounds.

        Returns:
            dict: Optimized parameter values.
        """

        cost_function = self._create_cost_function(model, current_x)

        if self.method == 'som':
            optimizer_result = self._optimize_som(cost_function, initial_values)
        elif self.method == 'dr':
            optimizer_result = self._optimize_dr(cost_function, initial_bounds_dr)
        else:
            raise ValueError("Invalid optimization method.")
        self.optimizer_result = optimizer_result
        return optimizer_result

    def _create_cost_function(self, model, current_x):
        """
        Creates a cost function for optimization.

        Args:
            model (UserModel): The model to optimize.
            current_x (float): The current time point for optimization.

        Returns:
            function: A cost function for optimization.
        """
        if self.method == 'som':
            def cost_function(params):
                optimized_params = {key: param for key, param in zip(self.params_info.keys(), params)}
                # optimized_params = dict(zip(self.params_info.keys(), params))
                model_error = model.calculate_error(optimized_params, current_x, self.kind)

                return model_error
        if self.method == 'dr':
            def cost_function(**params):
                model_error = model.calculate_error(params, current_x, self.kind)
                model_error = -model_error
            
                return model_error


        return cost_function

    def _optimize_som(self, cost_function, initial_values):
        """
        Performs optimization using scipy.optimize.minimize.

        Args:
            cost_function (function): The cost function for optimization.
            initial_values (np.ndarray): Initial values for the optimization parameters.

        Returns:
            dict: Optimized parameter values.
        """
        solution = minimize(cost_function, initial_values, bounds=self.bounds_global, method='Powell')
        return {key: value for key, value in zip(self.params_info.keys(), solution.x)}

    def _optimize_dr(self, cost_function, initial_bounds_dr):
        """
        Performs optimization using Bayesian optimization with domain reduction.

        Args:
            cost_function (function): The cost function for optimization.
            initial_bounds_dr (dict): Initial domain reduction bounds.

        Returns:
            dict: Optimized parameter values.
        """
        bounds_transformer = SequentialDomainReductionTransformer()
        optimizer = BayesianOptimization(f=cost_function, pbounds=initial_bounds_dr, 
                                         verbose=0, bounds_transformer=bounds_transformer,
                                         allow_duplicate_points=True)
        
        acq_function = UtilityFunction(kind='ucb')
        optimizer.maximize(init_points=10, n_iter=50, acquisition_function=acq_function)

        return {key: optimizer.max['params'][key] for key in self.params_info.keys()}

    @property
    def value_optimized(self):
        """
        Returns the optimized values of the parameters.

        Returns:
            np.ndarray: An array of optimized parameter values.
        """
        return np.array([value for value in self.optimizer_result.values()])