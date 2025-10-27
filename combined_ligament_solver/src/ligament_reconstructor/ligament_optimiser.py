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

def least_squares_optimize_complete_model(thetas, applied_forces, lcl_lengths, mcl_lengths, 
                                        constraint_manager, knee_config, sigma_noise=1e3):
    """
    Perform least squares optimization for the complete knee model using MCMC-like loss function.
    
    Args:
        thetas: Array of knee angles
        applied_forces: Array of applied forces
        lcl_lengths: Array of LCL lengths
        mcl_lengths: Array of MCL lengths
        constraint_manager: Tuple of (MCL_constraint_manager, LCL_constraint_manager)
        knee_config: Knee configuration dictionary
        sigma_noise: Noise standard deviation for likelihood calculation
        
    Returns:
        Dictionary with optimization results
    """
    from src.ligament_models.blankevoort import BlankevoortFunction
    from src.statics_solver.models.statics_model import KneeModel
    from scipy.optimize import minimize
    
    mcl_constraint_manager, lcl_constraint_manager = constraint_manager
    
    def mcmc_like_loss(params):
        """
        Loss function similar to MCMC likelihood but for least squares optimization.
        Returns the negative log-likelihood (to minimize).
        """
        # Split parameters into MCL (first 4) and LCL (last 4) parameters
        mcl_params = params[:4]  # [k, alpha, l_0, f_ref]
        lcl_params = params[4:]  # [k, alpha, l_0, f_ref]
        
        # Validate parameter arrays
        if not np.all(np.isfinite(mcl_params)) or not np.all(np.isfinite(lcl_params)):
            return np.inf
        
        try:
            # Create ligament functions
            mcl_func = BlankevoortFunction(mcl_params)
            lcl_func = BlankevoortFunction(lcl_params)
            
            # Create knee model
            knee_model = KneeModel(knee_config, lcl_func, mcl_func, log=False)
            
            predicted_forces = []
            
            # For each theta, calculate predicted applied force
            for i, theta in enumerate(thetas):
                # Update model angle
                knee_model.knee_joint.theta = theta
                
                # Get ligament tensions using the model's ligament functions
                mcl_tension = float(knee_model.lig_function_left(mcl_lengths[i]))
                lcl_tension = float(knee_model.lig_function_right(lcl_lengths[i]))
                
                # Calculate moment arms and applied force (same as MCMC)
                contact_point = knee_model.knee_joint.get_contact_point(theta=theta)
                mcl_direction = knee_model.lig_springB.get_force_direction_on_p1()
                lcl_direction = knee_model.lig_springA.get_force_direction_on_p1()
                
                mcl_force_vector = abs(mcl_tension) * mcl_direction
                lcl_force_vector = abs(lcl_tension) * lcl_direction
                
                tibia_frame = knee_model.tibia_frame
                mcl_moment_arm = knee_model.calculate_moment_arm(
                    knee_model.lig_bottom_pointA.convert_to_frame(tibia_frame), 
                    mcl_direction.convert_to_frame(tibia_frame), 
                    contact_point.convert_to_frame(tibia_frame)
                )
                mcl_moment_arm = abs(float(mcl_moment_arm))
                
                lcl_moment_arm = knee_model.calculate_moment_arm(
                    knee_model.lig_bottom_pointB.convert_to_frame(tibia_frame), 
                    lcl_direction.convert_to_frame(tibia_frame), 
                    contact_point.convert_to_frame(tibia_frame)
                )
                lcl_moment_arm = abs(float(lcl_moment_arm))
                
                from src.statics_solver.src.reference_frame import Point
                applied_moment_arm = knee_model.calculate_moment_arm(
                    knee_model.application_point.convert_to_frame(tibia_frame), 
                    Point([1,0,0], tibia_frame), 
                    contact_point.convert_to_frame(tibia_frame)
                )
                applied_moment_arm = abs(float(applied_moment_arm))
                
                applied_force = (mcl_tension * mcl_moment_arm - lcl_tension * lcl_moment_arm) / applied_moment_arm
                applied_force = float(applied_force)
                predicted_forces.append(applied_force)
            
            # Compute residuals
            residuals = np.array(applied_forces) - np.array(predicted_forces)
            
            # Gaussian log-likelihood (negative for minimization)
            log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(thetas) * np.log(sigma_noise * np.sqrt(2 * np.pi))
            
            return -log_like  # Return negative for minimization
            
        except Exception as e:
            print(f"Error in loss function: {e}")
            return np.inf
    
    def constraint_loss(params):
        """Add constraint penalties to the loss function."""
        mcl_params = params[:4]
        lcl_params = params[4:]
        
        penalty = 0.0
        
        # Check MCL constraints
        mcl_constraints = mcl_constraint_manager.get_constraints_list()
        for i, (lower, upper) in enumerate(mcl_constraints):
            if mcl_params[i] < lower:
                penalty += 1e6 * (lower - mcl_params[i])**2
            elif mcl_params[i] > upper:
                penalty += 1e6 * (mcl_params[i] - upper)**2
        
        # Check LCL constraints
        lcl_constraints = lcl_constraint_manager.get_constraints_list()
        for i, (lower, upper) in enumerate(lcl_constraints):
            if lcl_params[i] < lower:
                penalty += 1e6 * (lower - lcl_params[i])**2
            elif lcl_params[i] > upper:
                penalty += 1e6 * (lcl_params[i] - upper)**2
        
        return penalty
    
    def total_loss(params):
        """Combined loss function with constraints."""
        return mcmc_like_loss(params) + constraint_loss(params)
    
    # Initial guess - use reasonable starting values
    mcl_start = [40, 0.06, 90.0, 0.0]
    lcl_start = [60, 0.06, 60.0, 0.0]
    initial_params = np.array(mcl_start + lcl_start)
    
    print(f"Starting least squares optimization...")
    print(f"Initial parameters: {initial_params}")
    
    # Run optimization
    result = minimize(
        total_loss, 
        initial_params, 
        method='L-BFGS-B',
        options={'maxiter': 1000, 'disp': True}
    )
    
    if not result.success:
        print(f"Optimization failed: {result.message}")
        # Try with different method
        result = minimize(
            total_loss, 
            initial_params, 
            method='TNC',
            options={'maxiter': 1000, 'disp': True}
        )
    
    print(f"Optimization result: {result.message}")
    print(f"Final parameters: {result.x}")
    print(f"Final loss: {result.fun}")
    
    # Extract results
    mcl_params = result.x[:4]
    lcl_params = result.x[4:]
    
    # Create final ligament functions
    mcl_func = BlankevoortFunction(mcl_params)
    lcl_func = BlankevoortFunction(lcl_params)
    
    # Calculate final predictions
    knee_model = KneeModel(knee_config, lcl_func, mcl_func, log=False)
    predicted_forces = []
    
    for i, theta in enumerate(thetas):
        knee_model.knee_joint.theta = theta
        mcl_tension = float(knee_model.lig_function_right(mcl_lengths[i]))
        lcl_tension = float(knee_model.lig_function_left(lcl_lengths[i]))
        
        contact_point = knee_model.knee_joint.get_contact_point(theta=theta)
        mcl_direction = knee_model.lig_springA.get_force_direction_on_p2()
        lcl_direction = knee_model.lig_springB.get_force_direction_on_p2()
        
        tibia_frame = knee_model.tibia_frame
        mcl_moment_arm = knee_model.calculate_moment_arm(
            knee_model.lig_bottom_pointA.convert_to_frame(tibia_frame), 
            mcl_direction.convert_to_frame(tibia_frame), 
            contact_point.convert_to_frame(tibia_frame)
        )
        mcl_moment_arm = abs(float(mcl_moment_arm))
        
        lcl_moment_arm = knee_model.calculate_moment_arm(
            knee_model.lig_bottom_pointB.convert_to_frame(tibia_frame), 
            lcl_direction.convert_to_frame(tibia_frame), 
            contact_point.convert_to_frame(tibia_frame)
        )
        lcl_moment_arm = abs(float(lcl_moment_arm))
        
        from src.statics_solver.src.reference_frame import Point
        applied_moment_arm = knee_model.calculate_moment_arm(
            knee_model.application_point.convert_to_frame(tibia_frame), 
            Point([1,0,0], tibia_frame), 
            contact_point.convert_to_frame(tibia_frame)
        )
        applied_moment_arm = abs(float(applied_moment_arm))
        
        applied_force = (mcl_tension * mcl_moment_arm - lcl_tension * lcl_moment_arm) / applied_moment_arm
        predicted_forces.append(float(applied_force))
    
    # Calculate residuals and statistics
    residuals = np.array(applied_forces) - np.array(predicted_forces)
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    
    return {
        'mcl_params': mcl_params,
        'lcl_params': lcl_params,
        'predicted_forces': predicted_forces,
        'residuals': residuals,
        'rmse': rmse,
        'mae': mae,
        'optimization_result': result,
        'mcl_function': mcl_func,
        'lcl_function': lcl_func
    }