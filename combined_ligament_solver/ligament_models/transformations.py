import numpy as np
from ligament_models.constraints import ConstraintManager

def inverse_constraint_transform(params: np.ndarray, constraint_manager: ConstraintManager) -> np.ndarray:
    """
    Transform unconstrained parameters to be within the constraints using log transformation.
    
    The transformation is: theta = a + (b - a) / (1 + exp(-phi))
    where phi is the unconstrained parameter and theta is the constrained parameter
    that lies in the interval (a, b).
    
    Args:
        params: Unconstrained parameter array
        constraint_manager: ConstraintManager instance with bounds
        
    Returns:
        Transformed parameters within constraints
    """
    transformed_params = np.copy(params)
    constraints_list = constraint_manager.get_constraints_list()
    
    for i, (lower, upper) in enumerate(constraints_list):
        if i < len(params):
            # Apply the log transformation: theta = a + (b - a) / (1 + exp(-phi))
            
            phi = params[i]
            a, b = lower, upper
            theta = a + (b - a) / (1 + np.exp(-phi))
            transformed_params[i] = theta
    
    return transformed_params

def constraint_transform(params: np.ndarray, constraint_manager: ConstraintManager) -> np.ndarray:
    """
    Transform of constrained parameters to unconstrained space.
    
    The transformation is: phi = -log((b - theta) / (theta - a))
    where theta is the constrained parameter in (a, b) and phi is the unconstrained parameter.
    
    Args:
        params: Constrained parameter array
        constraint_manager: ConstraintManager instance with bounds
        
    Returns:
        Unconstrained parameters
    """
    unconstrained_params = np.copy(params)
    constraints_list = constraint_manager.get_constraints_list()
    
    for i, (lower, upper) in enumerate(constraints_list):
        if i < len(params):
            # Apply the inverse log transformation: phi = -log((b - theta) / (theta - a))
            theta = params[i]
            a, b = lower, upper
            
            # Ensure theta is within bounds (with small tolerance for numerical stability)
            theta = np.clip(theta, a + 1e-10, b - 1e-10)
            
            phi = -np.log((b - theta) / (theta - a))
            unconstrained_params[i] = phi
    
    return unconstrained_params
