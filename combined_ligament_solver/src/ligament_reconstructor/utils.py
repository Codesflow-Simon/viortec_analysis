import numpy as np

def get_initial_guess(params):
    initial_guess = params.values()
    # Add 10% noise to initial guess
    noise = np.random.normal(0, 0.4, len(initial_guess))  # 40% noise
    initial_guess = [val * (1 + noise[i]) for i, val in enumerate(initial_guess)]
    # initial_guess = enforce_constraints(initial_guess)
    initial_guess = {k: v for k, v in zip(params.keys(), initial_guess)}
    return initial_guess

def function_norm(params, gt_params, funct_vectorized):
    """
    Calculate the percentage deviation of fitted parameters from ground truth.
    
    Args:
        params: Fitted parameter values
        gt_params: Ground truth parameter values
        
    Returns:
        List of percentage deviations for each parameter
    """
    x_data = np.linspace(0, 100, 100)
    temp_func_1 = deepcopy(funct_vectorized)
    temp_func_1.set_params(gt_params)
    temp_func_2 = deepcopy(funct_vectorized)
    temp_func_2.set_params(params)
    y_data = np.array(temp_func_1(x_data))
    y_hat = np.array(temp_func_2(x_data))
    loss = np.sqrt(np.mean((y_data - y_hat)**2))
    return loss

def parameter_norm(params, gt_params):
    """
    Calculate the percentage deviation of fitted parameters from ground truth.
    """
    # Calculate relative error for each parameter, normalized by ground truth values
    if isinstance(params, dict):
        params = np.array(list(params.values()))
        gt_params = np.array(list(gt_params.values()))
    
    # Avoid division by zero by adding small epsilon where gt_params are zero
    scales = np.where(gt_params != 0, np.abs(gt_params), 1e-10)
    
    # Calculate normalized differences
    rel_errors = (params - gt_params) / scales
    
    return np.sqrt(np.mean(rel_errors**2))

def create_synthetic_data(gt_func_vec, params, x_noise=0, y_noise=0, x_min=0, x_max=0.2, n_points=100):
    # Generate data points
    x_data = np.linspace(x_min, x_max, n_points)  # Sample points from before x_0 to after x_2
    y_data = gt_func_vec(x_data, params)
    x_noise = np.random.normal(0, x_noise, len(x_data))
    y_noise = np.random.normal(0, y_noise, len(y_data))
    x_data = x_data + x_noise
    y_data = y_data + y_noise
    return x_data, y_data

def get_params_from_config(config, mode):
    if 'trilinear' in mode:
        return {"k_1": float(config[mode]['stiffness_1']), 
                "k_2": float(config[mode]['stiffness_2']), 
                "k_3": float(config[mode]['stiffness_3']), 
                "l_0": float(config[mode]['length_0']), 
                "a_1": float(config[mode]['a_1']), 
                "a_2": float(config[mode]['a_2'])}
    elif 'blankevoort' in mode:
        return {"k": float(config[mode]['k']),
                "alpha": float(config[mode]['alpha']), 
                "l_0": float(config[mode]['l_0']),
                "f_ref": float(config[mode]['f_ref'])}