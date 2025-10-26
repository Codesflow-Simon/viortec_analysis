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