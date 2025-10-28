import numpy as np
import yaml
from scipy.optimize import minimize
from src.statics_model import blankevoort_func
from src.statics_model import KneeModel

def parse_constraints(constraints_config):
    """
    Parse constraints configuration and return bounds for optimization.
    
    Args:
        constraints_config: Dictionary containing constraint configurations for different ligament models
        
    Returns:
        Dictionary with bounds for each ligament model
    """
    bounds = {}
    
    for model_name, model_config in constraints_config.items():
        if 'constraints' not in model_config:
            continue
            
        model_bounds = []
        parameter_order = model_config.get('parameters', {})
        
        # Sort parameters by their index values to maintain consistent order
        sorted_params = sorted(parameter_order.items(), key=lambda x: x[1])
        
        for param_name, param_index in sorted_params:
            if param_name in model_config['constraints']:
                constraint = model_config['constraints'][param_name]
                if constraint['type'] == 'between':
                    model_bounds.append((constraint['lower'], constraint['upper']))
                else:
                    # For other constraint types, use a large range as fallback
                    model_bounds.append((-1e6, 1e6))
            else:
                # If no constraint specified, use a large range
                model_bounds.append((-1e6, 1e6))
        
        bounds[model_name] = model_bounds

    return bounds

def clip_params_to_bounds(params, bounds):
    """
    Clip parameters to their constraint bounds.
    
    Args:
        params: Parameter array
        bounds: List of (lower, upper) bounds for each parameter
        
    Returns:
        Clipped parameter array
    """
    clipped_params = np.array(params)
    for i, (lower, upper) in enumerate(bounds):
        clipped_params[i] = np.clip(clipped_params[i], lower, upper)
    return clipped_params

def least_squares_optimize_complete_model(thetas, applied_forces, knee_config, constraints_config, sigma_noise=1e1):
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
    
    knee_model = KneeModel(knee_config, log=False)
    knee_model.build_geometry()
    
    bounds = parse_constraints(constraints_config)
    
    def mcmc_like_loss(params, thetas, applied_forces):
        """
        Loss function similar to MCMC likelihood but for least squares optimization.
        Returns the negative log-likelihood (to minimize).
        """
        # Split parameters into MCL (first 4) and LCL (last 4) parameters
        mcl_params = params[:4]  # [k, alpha, l_0, f_ref]
        lcl_params = params[4:]  # [k, alpha, l_0, f_ref]
        
        # Apply constraint clipping
        mcl_params = clip_params_to_bounds(mcl_params, bounds['blankevoort_mcl'])
        lcl_params = clip_params_to_bounds(lcl_params, bounds['blankevoort_lcl'])

        estimated_applied_forces = knee_model.solve_applied(thetas, mcl_params, lcl_params)['applied_forces']
        estimated_applied_forces = np.array(estimated_applied_forces).reshape(-1)
        return np.sum((applied_forces - estimated_applied_forces)**2)
    
    applied_forces = np.array(applied_forces).reshape(-1)
    thetas = np.array(thetas)
    loss_func = lambda params: mcmc_like_loss(params, thetas, applied_forces)

    # Initial guess - use reasonable starting values
    mcl_start = [40, 0.06, 90.0, 0.0]
    lcl_start = [60, 0.06, 60.0, 0.0]
    initial_params = np.array(mcl_start + lcl_start)
    inital_loss = loss_func(initial_params)

    
    mcl_gt = [33.5, 0.06, 89.43, 0]  # Ground truth MCL parameters from config
    lcl_gt = [42.8, 0.06, 59.528, 0]  # Ground truth LCL parameters from config
    gt_params = np.array(mcl_gt + lcl_gt)
    gt_loss = loss_func(gt_params)

    # Create bounds for optimization (combine MCL and LCL bounds)
    mcl_bounds = bounds['blankevoort_mcl']
    lcl_bounds = bounds['blankevoort_lcl']
    optimization_bounds = mcl_bounds + lcl_bounds
    
    # Run optimization with bounds
    result = minimize(
        loss_func,
        initial_params,
        method='L-BFGS-B',
        bounds=optimization_bounds
    )
    
    if not result.success:
        print(f"Optimization failed: {result.message}")
        # Try with different method
        result = minimize(
            loss_func, 
            initial_params, 
            method='TNC',
            bounds=optimization_bounds,
            options={'maxiter': 1000, 'disp': True}
        )
    
    print(f"Optimization result: {result.message}")
    print(f"Final parameters: {result.x}")
    print(f"Final loss: {result.fun}")
    
    # Extract results and clip to bounds
    mcl_params = clip_params_to_bounds(result.x[:4], mcl_bounds)
    lcl_params = clip_params_to_bounds(result.x[4:], lcl_bounds)

    predicted_forces = knee_model.solve_applied(thetas, mcl_params, lcl_params)['applied_forces']
    predicted_forces = np.array(predicted_forces).reshape(-1)

    # Calculate residuals and statistics
    residuals = applied_forces - predicted_forces
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
        'predicted_forces': predicted_forces,
        'residuals': residuals,
        'rmse': rmse,
        'mae': mae
    }