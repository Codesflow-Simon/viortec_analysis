import numpy as np
from ligament_models.constraints import ConstraintManager

def inverse_constraint_transform(params: np.ndarray, constraint_manager: ConstraintManager) -> np.ndarray:
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
    constraints_list = constraint_manager.get_constraints_list()
    
    # Convert list to numpy array if needed
    params = np.array(params)
    
    # Handle both 1D and 2D inputs
    input_is_1d = params.ndim == 1
    if input_is_1d:
        params = params.reshape(1, -1)
        
    transformed_params = np.copy(params)
    
    for i, (lower, upper) in enumerate(constraints_list):
        if i < params.shape[1]:
            # Apply the log transformation: theta = a + (b - a) / (1 + exp(-phi))
            phi = params[:, i]
            a, b = lower, upper
            # Clip phi to avoid overflow in exp
            phi = np.clip(phi, -700, 700)  # exp(700) is near float max
            theta = a + (b - a) / (1 + np.exp(-phi))
            transformed_params[:, i] = theta
    return transformed_params[0] if input_is_1d else transformed_params
def constraint_transform(params: np.ndarray, constraint_manager: ConstraintManager) -> np.ndarray:
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
    
    constraints_list = constraint_manager.get_constraints_list()
    
    # Handle both 1D and 2D inputs
    input_is_1d = params.ndim == 1
    if input_is_1d:
        params = params.reshape(1, -1)
        
    unconstrained_params = np.copy(params)
    
    for i, (lower, upper) in enumerate(constraints_list):
        if i < params.shape[1]:
            # Apply the inverse log transformation: phi = -log((b - theta) / (theta - a))
            theta = params[:, i]
            a, b = lower, upper
            
            # Ensure theta is within bounds (with small tolerance for numerical stability)
            theta = np.clip(theta, a + 1e-10, b - 1e-10)
            
            phi = -np.log((b - theta) / (theta - a))
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