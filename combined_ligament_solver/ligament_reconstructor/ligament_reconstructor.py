import numpy as np
import yaml
from ligament_models import *
from scipy.optimize import minimize
from .plot import make_plots
from .loss import loss, loss_jac, loss_hess
from ligament_models.constraints import ConstraintManager
from .utils import get_params_from_config, get_initial_guess, create_synthetic_data
from ligament_models.transformations import inverse_constraint_transform, constraint_transform

class LigamentReconstructor:
    def __init__(self):
        pass

    def reconstruct(self, x_data:np.ndarray, y_data:np.ndarray, initial_guess:dict):
        initial_guess_dict = get_initial_guess(initial_guess)
        initial_guess_list = list(initial_guess_dict.values())

        initial_guess_list = constraint_transform(initial_guess_list, self.constraint_manager)
        opt_result = self.solve(x_data, y_data, initial_guess_list)
        opt_result.x = inverse_constraint_transform(opt_result.x, self.constraint_manager)
        

        param_names = self.constraint_manager.get_param_names()
        params = dict(zip(param_names, opt_result.x))

        loss_value = loss(opt_result.x, x_data, y_data, funct=self.function)
        loss_jac_value = loss_jac(opt_result.x, x_data, y_data, funct=self.function, funct_jac=self.function.jac)
        loss_hess_value = loss_hess(opt_result.x, x_data, y_data, funct=self.function, funct_jac=self.function.jac, funct_hess=self.function.hess)

        predicted_y = self.function(opt_result.x)
        info_dict = {
            'opt_result': opt_result,
            'y_hat': predicted_y,
            'params': params,
            'x_data': x_data,
            'y_data': y_data,
            'function': self.function,
            'loss': loss_value,
            'loss_jac': loss_jac_value,
            'loss_hess': loss_hess_value,
        }

        return info_dict

    def get_param_values(self):
        if 'trilinear' in self.mode:
            return [self.params['k_1'], self.params['k_2'], self.params['k_3'], self.params['l_0'], self.params['a_1'], self.params['a_2']]
        elif 'blankevoort' in self.mode:
            return [self.params['k'], self.params['alpha'], self.params['l_0'], self.params['f_ref']]

    def get_params(self):
        if 'trilinear' in self.mode:
            return self.params
        elif 'blankevoort' in self.mode:
            return self.params

    def setup_model(self, mode, params):
        self.mode = mode
        if 'trilinear' in mode:
            self.params = params
            self.function = TrilinearFunction(params)
            self.constraint_manager = ConstraintManager(mode)
            
        elif 'blankevoort' in mode:
            self.params = params
            self.function = BlankevoortFunction(params)
            self.constraint_manager = ConstraintManager(mode)
            
    def solve(self, x_data, y_data, initial_guess):
        # loss_func = lambda params: loss(params, x_data, y_data, funct=self.function)
        # jac_func = lambda params: loss_jac(params, x_data, y_data, funct=self.function, funct_jac=self.function.jac)
        # hess_func = lambda params: loss_hess(params, x_data, y_data, funct=self.function, funct_jac=self.function.jac, funct_hess=self.function.hess)
        loss_func = lambda params: loss(inverse_constraint_transform(params, self.constraint_manager), x_data, y_data, funct=self.function)
        jac_func = lambda params: loss_jac(inverse_constraint_transform(params, self.constraint_manager), x_data, y_data, funct=self.function, funct_jac=self.function.jac)
        hess_func = lambda params: loss_hess(inverse_constraint_transform(params, self.constraint_manager), x_data, y_data, funct=self.function, funct_jac=self.function.jac, funct_hess=self.function.hess)

        result = minimize(loss_func, initial_guess, method='trust-constr', jac=jac_func)
        return result
