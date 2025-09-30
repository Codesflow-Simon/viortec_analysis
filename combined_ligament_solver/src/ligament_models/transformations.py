import numpy as np
from src.ligament_models.constraints import ConstraintManager
from typing import List, Tuple

def inv_constraint_transform(param: float, lower: float, upper: float) -> float:
    """
    Transform a single parameter to be within the constraints using log transformation.
    """
    param = np.clip(param, -700, 700)
    return lower + (upper - lower) / (1 + np.exp(-param))

def constraint_transform(param: float, lower: float, upper: float) -> float:
    """
    Transform a single parameter to be within the constraints using log transformation.
    """
    # Clip parameters to avoid numerical issues at boundaries
    param = np.clip(param, lower + 1e-10, upper - 1e-10)
    return -np.log((upper - param) / (param - lower))

def batch_constraint_transform(params: np.ndarray, bounds_list: List[Tuple[float, float]]) -> np.ndarray:
    """
    Transform unconstrained parameters to be within the constraints using log transformation.
    
    The transformation is: theta = a + (b - a) / (1 + exp(-phi))
    where phi is the unconstrained parameter and theta is the constrained parameter
    that lies in the interval (a, b).
    
    Args:
        params: Unconstrained parameter array, matrix (n_samples x n_params), or list
        constraint_manager: ConstraintManager instance with bounds
        
    Returns:
        Transformed parameters within constraints (same shape as input)
    """
    
    # Convert list to numpy array if needed
    params = np.array(params)
    
    # Handle both 1D and 2D inputs
    input_is_1d = params.ndim == 1
    if input_is_1d:
        params = params.reshape(1, -1)
        
    transformed_params = np.copy(params)
    
    for i, (lower, upper) in enumerate(bounds_list):
        if i < params.shape[1]:
            # Apply the log transformation: theta = a + (b - a) / (1 + exp(-phi))
            phi = params[:, i]
            a, b = lower, upper
            theta = inv_constraint_transform(phi, a, b)
            transformed_params[:, i] = theta    
    return transformed_params[0] if input_is_1d else transformed_params

def batch_inv_constraint_transform(params: np.ndarray, bounds_list: List[Tuple[float, float]]) -> np.ndarray:
    """
    Transform of constrained parameters to unconstrained space.
    
    The transformation is: phi = -log((b - theta) / (theta - a))
    where theta is the constrained parameter in (a, b) and phi is the unconstrained parameter.
    
    Args:
        params: Constrained parameter array, matrix (n_samples x n_params), or list
        constraint_manager: ConstraintManager instance with bounds
        
    Returns:
        Unconstrained parameters (same shape as input)
    """
    # Convert list to numpy array if needed
    params = np.array(params)
    
    if not bounds_list:
        bounds_list = constraint_manager.get_constraints_list()
    
    # Handle both 1D and 2D inputs
    input_is_1d = params.ndim == 1
    if input_is_1d:
        params = params.reshape(1, -1)
        
    unconstrained_params = np.copy(params)
    
    for i, (lower, upper) in enumerate(bounds_list):
        if i < params.shape[1]:
            # Apply the inverse log transformation: phi = -log((b - theta) / (theta - a))
            theta = params[:, i]
            a, b = lower, upper
            
            phi = constraint_transform(theta, a, b)
            unconstrained_params[:, i] = phi
    
    return unconstrained_params[0] if input_is_1d else unconstrained_params

def sliding_operation(params: dict, slide_factor: float) -> dict:
    """
    Slide the parameters by a factor of slide_factor.
    This preserves forces at the same x-coordinates in the linear region.
    """
    params = params.copy()
    params['l_0'] = params['l_0'] + slide_factor
    # The correct adjustment for f_ref to preserve forces in the linear region
    # accounts for the fact that transition_length = l_0 * alpha changes
    params['f_ref'] = params['f_ref'] - params['k'] * slide_factor * (1 + params['alpha']/2)
    return params

def slide_domain_inv_constraint_transform(params: dict, standard_bounds_list: List[Tuple[float, float]], map_params: dict) -> dict:
    """
    Slide the parameters by a factor of slide_factor.
    This preserves forces at the same x-coordinates in the linear region.
    """
    param_names = ['k', 'alpha', 'l_0', 'l_slide']        
    input_params = params.copy()
    input_params = dict(zip(param_names, input_params))
        
    k_bounds, alpha_bounds, l_0_bounds, _ = standard_bounds_list
    l_slide_bounds = np.array([l_0_bounds[0], l_0_bounds[1]]) - map_params['l_0']

    bounds_list = [k_bounds, alpha_bounds, l_0_bounds, l_slide_bounds]

    # Transform parameters to constrained space
    if isinstance(params, dict):
        params = np.array(list(params.values()))
    params = batch_inv_constraint_transform(params, bounds_list)
    params = dict(zip(['k', 'alpha', 'l_0', 'l_slide'], params))
    return params

def slide_domain_constraint_transform(params: dict, standard_bounds_list: List[Tuple[float, float]], map_params: dict) -> dict:
    """
    Slide the parameters by a factor of slide_factor.
    This preserves forces at the same x-coordinates in the linear region.
    """
    param_names = ['k', 'alpha', 'l_0', 'l_slide']        
    input_params = params.copy()
    input_params = dict(zip(param_names, input_params))
        
    k_bounds, alpha_bounds, l_0_bounds, _ = standard_bounds_list
    l_slide_bounds = np.array([l_0_bounds[0], l_0_bounds[1]]) - map_params['l_0']

    bounds_list = [k_bounds, alpha_bounds, l_0_bounds, l_slide_bounds]

    # Transform parameters to constrained space
    if isinstance(params, dict):
        params = np.array(list(params.values()))
    params = batch_constraint_transform(params, bounds_list)
    params = dict(zip(['k', 'alpha', 'l_0', 'l_slide'], params))
    return params


def slide_to_standard_domain(params: dict, constraint_manager: ConstraintManager, map_params: dict) -> dict:
    """
    Slide the parameters to the standard domain.
    """
    params = params.copy()
    params['f_ref'] = map_params['f_ref']

    slide = params['l_slide']
    del params['l_slide']
    temp_params = sliding_operation(params, slide)
    return temp_params

def standard_to_slide_domain(params: dict, constraint_manager: ConstraintManager, map_params: dict) -> dict:
    """
    Slide the parameters from the standard domain to the slide domain.
    This is the inverse of slide_to_standard_domain.
    """
    params = params.copy()
    params['f_ref'] = map_params['f_ref']
    
    # Get l_slide bounds from constraint manager
    l_0_bounds = constraint_manager.get_constraints_list()[2]
    l_slide = params['l_0'] - map_params['l_0']
    
    # Apply inverse sliding operation
    temp_params = sliding_operation(params, -l_slide)
    temp_params['l_slide'] = l_slide
    return temp_params