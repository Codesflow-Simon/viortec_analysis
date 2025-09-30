import numpy as np
import yaml
from .utils import  get_initial_guess
from scipy.optimize import minimize
from src.ligament_models import *
from src.ligament_models.constraints import ConstraintManager
from src.ligament_models.transformations import batch_inv_constraint_transform, batch_constraint_transform

reg_coef = 1e-2

def loss(params, x_data, y_data, funct, include_reg=False):
    # Convert single parameter set to 2D array for vectorized computation
    if params.ndim == 1:
        params_array = params.reshape(1, -1)
    else:
        params_array = params
    
    # Use vectorized function evaluation
    y_pred_matrix = funct.vectorized_function(x_data, params_array)
    
    # For single parameter set, extract the result
    if params.ndim == 1:
        y_pred = y_pred_matrix[0]
        squared_error = np.sum((y_data - y_pred)**2)
        l1_term = np.abs(params).sum()
        if include_reg:
            l1 = reg_coef * l1_term
            loss = squared_error + l1
        else:
            loss = squared_error
        return loss
    else:
        # Multiple parameter sets - compute loss for each
        squared_errors = np.sum((y_data - y_pred_matrix)**2, axis=1)
        l1_terms = np.sum(np.abs(params_array), axis=1)
        if include_reg:
            l1 = reg_coef * l1_terms
            losses = squared_errors + l1
        else:
            losses = squared_errors
        return losses

def loss_jac(params, x_data, y_data, funct, include_reg=False):
    n = len(x_data)
    p = len(params)
    
    # Convert single parameter set to 2D array for vectorized computation
    if params.ndim == 1:
        params_array = params.reshape(1, -1)
    else:
        params_array = params
    
    # Use vectorized function and Jacobian evaluation
    y_pred_matrix = funct.vectorized_function(x_data, params_array)
    J_matrix_3d = funct.vectorized_jacobian(x_data, params_array)  # Shape: (n_param_sets, n_params, n_points)
    
    # For single parameter set, extract the result
    if params.ndim == 1:
        y_pred = y_pred_matrix[0]
        J_matrix = J_matrix_3d[0]  # Shape: (n_params, n_points)
        residuals = y_pred - y_data
        # Compute gradient: 2/n * J_matrix @ residuals
        G = 2/n * J_matrix @ residuals
        if include_reg:
            G += reg_coef * np.sign(params)
        return G
    else:
        # Multiple parameter sets - compute gradient for each
        residuals_matrix = y_pred_matrix - y_data  # Shape: (n_param_sets, n_points)
        gradients = np.zeros_like(params_array)
        
        for i in range(params_array.shape[0]):
            J_matrix = J_matrix_3d[i]  # Shape: (n_params, n_points)
            residuals = residuals_matrix[i]  # Shape: (n_points,)
            G = 2/n * J_matrix @ residuals
            if include_reg:
                G += reg_coef * np.sign(params_array[i])
            gradients[i] = G
            
        return gradients

def loss_hess(params, x_data, y_data, funct, include_reg=False):
    n = len(x_data)
    
    # Convert single parameter set to 2D array for vectorized computation
    if params.ndim == 1:
        params_array = params.reshape(1, -1)
        p = len(params)  # Number of parameters per set
    else:
        params_array = params
        p = params.shape[1]  # Number of parameters per set
    
    # Use vectorized Hessian evaluation
    H_matrix_4d = funct.vectorized_hessian(x_data, params_array)  # Shape: (n_param_sets, n_points, n_params, n_params)
    
    # For single parameter set, extract the result
    if params.ndim == 1:
        # Compute residuals
        y_pred = funct.vectorized_function(x_data, params_array)[0]
        residuals = y_pred - y_data  # Shape: (n_points,)
        
        # Get Hessian matrix for this parameter set
        H_matrix = H_matrix_4d[0]  # Shape: (n_points, n_params, n_params)
        
        # Compute full Hessian: H = 2 * sum(r_i * H_i)
        H_full = np.zeros((p, p))
        for i in range(n):
            # Ensure proper broadcasting by explicitly handling scalar multiplication
            residual_scalar = float(residuals[i])
            H_full += 2 * residual_scalar * H_matrix[i]  # Shape: (n_params, n_params)
        
        # Add regularization if requested
        if include_reg:
            H_full += reg_coef * np.eye(p)
        return H_full
    else:
        # Multiple parameter sets - compute Hessian for each
        hessians = np.zeros((params_array.shape[0], p, p))
        
        for i in range(params_array.shape[0]):
            # Compute residuals for this parameter set
            y_pred = funct.vectorized_function(x_data, params_array)[i]
            residuals = y_pred - y_data  # Shape: (n_points,)
            
            # Get Hessian matrix for this parameter set
            H_matrix = H_matrix_4d[i]  # Shape: (n_points, n_params, n_params)
            
            # Compute full Hessian: H = 2 * sum(r_i * H_i)
            H_full = np.zeros((p, p))
            for j in range(n):
                # Ensure proper broadcasting by explicitly handling scalar multiplication
                residual_scalar = float(residuals[j])
                H_full += 2 * residual_scalar * H_matrix[j]  # Shape: (n_params, n_params)
            
            # Add regularization if requested
            if include_reg:
                H_full += reg_coef * np.eye(p)
            hessians[i] = H_full
            
        return hessians

def reconstruct_ligament(x_data:np.ndarray, y_data:np.ndarray, constraint_manager:ConstraintManager):

    initial_guess_dict = {'k': 0, 'alpha': 0, 'l_0': 0, 'f_ref': 0}

    # Calculate approximate derivatives between neighboring points
    dx = np.diff(x_data)
    dy = np.diff(y_data) 
    derivatives = dy/dx
    
    # Use maximum derivative as initial estimate for k
    k_init = np.percentile(np.abs(derivatives), 75)
    
    # Update initial guess with k estimate
    initial_guess_dict['k'] = k_init

    # Use lowest x value and half alpha as initial l_0
    l_0_init = x_data.min() - 0.5 * initial_guess_dict['alpha']
    
    # Use lowest y value as initial f_ref 
    # Seems to work better if we come from below
    f_ref_init = -y_data.min()
    
    # Update initial guesses
    initial_guess_dict['l_0'] = l_0_init
    initial_guess_dict['f_ref'] = f_ref_init
    
    initial_guess_list = list(initial_guess_dict.values())

    # Get alpha constraints and set initial alpha to middle value
    alpha_constraints = constraint_manager.get_constraints_list()[1]  # Get alpha bounds
    alpha_init = (alpha_constraints[0] + alpha_constraints[1]) / 2  # Middle value
    initial_guess_dict['alpha'] = alpha_init
    initial_guess_list = list(initial_guess_dict.values())

    initial_guess_list = batch_inv_constraint_transform(initial_guess_list, constraint_manager.get_constraints_list())
    function = BlankevoortFunction(initial_guess_list)


    loss_func = lambda params: loss(batch_constraint_transform(params, constraint_manager.get_constraints_list()), x_data, y_data, funct=function, include_reg=True)
    jac_func = lambda params: loss_jac(batch_constraint_transform(params, constraint_manager.get_constraints_list()), x_data, y_data, funct=function, include_reg=True)
    hess_func = lambda params: loss_hess(batch_constraint_transform(params, constraint_manager.get_constraints_list()), x_data, y_data, funct=function, include_reg=True)

    initial_loss = loss_func(initial_guess_list)


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

    opt_result = minimize(loss_func, initial_guess_list, method='Newton-CG', 
                         jac=jac_func, hess=hess_func, options={'maxiter': 1000}) #, Add hess=hess_func if you want to use the Hessian
    opt_result.x = batch_constraint_transform(opt_result.x, constraint_manager.get_constraints_list())

    param_names = constraint_manager.get_param_names()
    params = dict(zip(param_names, opt_result.x))

    loss_value = loss(opt_result.x, x_data, y_data, funct=function)
    loss_jac_value = loss_jac(opt_result.x, x_data, y_data, funct=function)
    loss_hess_value = loss_hess(opt_result.x, x_data, y_data, funct=function)

    predicted_y = function(x_data)
    function.set_params(params)
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
        'initial_guess': initial_guess_dict,
        'initial_loss': initial_loss,
    }
    return info_dict