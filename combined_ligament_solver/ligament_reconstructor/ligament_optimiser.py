import numpy as np
import yaml
from .utils import  get_initial_guess
from scipy.optimize import minimize
from ligament_models import *
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import inverse_constraint_transform, constraint_transform

reg_coef = 1e-1

def loss(params, x_data, y_data, funct, include_reg=False):
    funct.set_params(params)
    y_pred = funct(x_data)
    squared_error = np.sum((y_data - y_pred)**2)
    l1_term = np.abs(params).sum()
    if include_reg:
        l1 =  reg_coef*l1_term
        loss = squared_error + l1
    else:
        loss = squared_error
    return loss

def loss_jac(params, x_data, y_data, funct, funct_jac, include_reg=False):
    n = len(x_data)
    p = len(params)
    funct.set_params(params)
    y_pred = funct(x_data)
    residuals = y_pred - y_data
    J_matrix = funct.jac(x_data)  # Shape: (n_params, n_points)
    # Compute gradient: 2/n * J_matrix @ residuals
    G = 2/n * J_matrix @ residuals
    if include_reg:
        G += reg_coef * np.sign(params)
    return G

def loss_hess(params, x_data, y_data, funct, funct_jac, funct_hess, include_reg=False):
    n = len(x_data)
    p = len(params)
    funct.set_params(params)
    y_pred = funct(x_data)
    J_matrix = funct.jac(x_data)  # Shape: (n_params, n_points)
    # Compute Gauss-Newton Hessian: J_matrix @ J_matrix.T
    H_gn = J_matrix @ J_matrix.T
    H_full = 2/n * H_gn
    if include_reg:
        H_full += reg_coef * np.eye(p)
    return H_full



def reconstruct_ligament(x_data:np.ndarray, y_data:np.ndarray):
    constraint_manager = ConstraintManager()

    initial_guess_dict = {'k': 0, 'alpha': 0, 'l_0': 0, 'f_ref': 0}

    # Calculate approximate derivatives between neighboring points
    dx = np.diff(x_data)
    dy = np.diff(y_data) 
    derivatives = dy/dx
    
    # Use maximum derivative as initial estimate for k
    k_init = np.mean(np.abs(derivatives))
    
    # Update initial guess with k estimate
    initial_guess_dict['k'] = k_init

    # Use lowest x value and half alpha as initial l_0
    l_0_init = x_data.min() - 0.5 * initial_guess_dict['alpha']
    
    # Use lowest y value as initial f_ref 
    f_ref_init = y_data.min()
    
    # Update initial guesses
    initial_guess_dict['l_0'] = l_0_init
    initial_guess_dict['f_ref'] = f_ref_init
    
    initial_guess_list = list(initial_guess_dict.values())

    # Get alpha constraints and set initial alpha to middle value
    alpha_constraints = constraint_manager.get_constraints_list()[1]  # Get alpha bounds
    alpha_init = (alpha_constraints[0] + alpha_constraints[1]) / 2  # Middle value
    initial_guess_dict['alpha'] = alpha_init
    initial_guess_list = list(initial_guess_dict.values())

    initial_guess_list = constraint_transform(initial_guess_list, constraint_manager)

    function = BlankevoortFunction(initial_guess_list)

    loss_func = lambda params: loss(inverse_constraint_transform(params, constraint_manager), x_data, y_data, funct=function, include_reg=True)
    jac_func = lambda params: loss_jac(inverse_constraint_transform(params, constraint_manager), x_data, y_data, funct=function, funct_jac=function.jac, include_reg=True)
    hess_func = lambda params: loss_hess(inverse_constraint_transform(params, constraint_manager), x_data, y_data, funct=function, funct_jac=function.jac, funct_hess=function.hess, include_reg=True)

    # Other possible optimizers:
    # Derivative-free methods:
    # - 'Nelder-Mead': Simplex method, no derivatives needed
    # - 'Powell': Direction set method, no derivatives needed
    
    # First derivative (Jacobian) methods:
    # - 'CG': Conjugate gradient algorithm
    # - 'BFGS': Quasi-Newton method with BFGS update
    # - 'L-BFGS-B': Limited memory BFGS with bounds
    # - 'TNC': Truncated Newton algorithm
    # - 'SLSQP': Sequential Least Squares Programming
    
    # Second derivative (Hessian) methods:
    # - 'Newton-CG': Newton's method with conjugate gradient
    # - 'dogleg': Dog-leg trust-region algorithm  
    # - 'trust-ncg': Newton conjugate gradient trust-region
    # - 'trust-krylov': Newton GLTR trust-region
    # - 'trust-exact': Nearly exact trust-region
    # - 'trust-constr': Trust-region method with constraints

    opt_result = minimize(loss_func, initial_guess_list, method='trust-constr', 
                         jac=jac_func, tol=1e-6)
    opt_result.x = inverse_constraint_transform(opt_result.x, constraint_manager)

    param_names = constraint_manager.get_param_names()
    params = dict(zip(param_names, opt_result.x))

    loss_value = loss(opt_result.x, x_data, y_data, funct=function)
    loss_jac_value = loss_jac(opt_result.x, x_data, y_data, funct=function, funct_jac=function.jac)
    loss_hess_value = loss_hess(opt_result.x, x_data, y_data, funct=function, funct_jac=function.jac, funct_hess=function.hess)

    predicted_y = function(x_data)
    info_dict = {
        'opt_result': opt_result,
        'y_hat': predicted_y,
        'params': params,
        'x_data': x_data,
        'y_data': y_data,
        'function': function,
        'loss': loss_value,
        'loss_jac': loss_jac_value,
        'loss_hess': loss_hess_value,
    }
    return info_dict