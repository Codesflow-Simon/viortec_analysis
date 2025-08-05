import numpy as np
import yaml
from function import *
from scipy.optimize import minimize
from plot import make_plots
from loss import *
import constraints
from constraints import trilinear_constraints, blankevoort_constraints
from sampling_covariance import compute_sampling_covariance
from utils import *
from bayesian import *


import matplotlib.pyplot as plt
import os

class LigamentReconstructor:
    def __init__(self):
        pass

    def reconstruct(self, x_data:np.ndarray, y_data:np.ndarray, initial_guess:dict):
        initial_guess =list(initial_guess.values())
        opt_result = self.solve(x_data, y_data, initial_guess)
        fitted_function = self.funct_class(*opt_result.x)
        info_dict = {
            'opt_result': opt_result,
            'fitted_function': fitted_function,
            'params': opt_result.x,
            'x_data': x_data,
            'y_data': y_data,
            'funct_tuple': self.funct_tuple,
            'param_names': self.param_names,
        }

        result_obj = LigamentReconstructor()
        result_obj.setup_model(self.mode)
        result_obj.set_params(opt_result.x)
        return result_obj, info_dict

    def get_param_values(self):
        if 'trilinear' in self.mode:
            return [self.params['k_1'], params['k_2'], params['k_3'], params['x_1'], params['x_2']]
        elif 'blankevoort' in self.mode:
            return [self.params['e_t'], self.params['k_1']]

    def get_params(self):
        if 'trilinear' in self.mode:
            return {"k_1": float(config[mode]['modulus_1']) * float(config[mode]['cross_section']), 
                    "k_2": float(config[mode]['modulus_2']) * float(config[mode]['cross_section']), 
                    "k_3": float(config[mode]['modulus_3']) * float(config[mode]['cross_section']), 
                    "x_1": float(config[mode]['x_1']), 
                    "x_2": float(config[mode]['x_2'])}

    def set_params(self, params):
        self.params = params

    def setup_model(self, mode):
        self.mode = mode
        if 'trilinear' in mode:
            self.param_names = constraints.trilinear_param_names
            self.funct_tuple = (trilinear_function, trilinear_function_jac, trilinear_function_hess)
            self.funct_class = TrilinearFunction
            self.vectorized_funct = trilinear_function_vectorized
            self.constraints_list = trilinear_constraints
            
        elif 'blankevoort' in mode:
            self.param_names = constraints.blankevoort_param_names
            self.funct_tuple = (blankevoort_function, blankevoort_function_jac, blankevoort_function_hess)
            self.funct_class = BlankevoortFunction
            self.vectorized_funct = blankevoort_function_vectorized
            self.constraints_list = blankevoort_constraints
            
    def solve(self, x_data, y_data, initial_guess):
        funct_tuple = self.funct_tuple
        loss_func = lambda params: loss(params, x_data, y_data, funct=funct_tuple[0])
        jac_func = lambda params: loss_jac(params, x_data, y_data, funct=funct_tuple[0], funct_jac=funct_tuple[1])
        hess_func = lambda params: loss_hess(params, x_data, y_data, funct=funct_tuple[0], funct_jac=funct_tuple[1], funct_hess=funct_tuple[2])

        constraints_list = self.constraints_list
        result = minimize(loss_func, initial_guess, method='trust-constr', 
                        jac=jac_func,
                        constraints=constraints_list)
        return result

if __name__ == "__main__":
    with open('ligament_reconstructor/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    gt_params = get_params_from_config(config, config['mode'])
    initial_guess = get_initial_guess(gt_params)

    ligament_reconstructor = LigamentReconstructor()
    ligament_reconstructor.setup_model(config['mode'])
    ligament_reconstructor.set_params(initial_guess)

    x_data, y_data = create_synthetic_data(ligament_reconstructor.vectorized_funct, gt_params.values(), x_noise=0.01, y_noise=0.01, x_min=0, x_max=0.2, n_points=100)

    result_obj, info_dict = ligament_reconstructor.reconstruct(x_data, y_data, initial_guess)
    
    gt_func = BlankevoortFunction(gt_params['e_t'], gt_params['k_1'])
    info_dict['ground_truth'] = gt_func

    result = bayesian_update(info_dict, config)
    make_plots(result)
