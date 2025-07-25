from function import TrilinearFunction, trilinear_function, trilinear_function_jac, trilinear_function_hess
from function import BlankevoortFunction, blankevoort_function, blankevoort_function_jac
import numpy as np
reg_coef = 1e-4

def loss(params, x_data, y_data, include_reg=True):
    # k_1, k_2, k_3, x_0, x_1, x_2 = params
    transition_length, k_1, x0 = params

    
    # Calculate predictions
    # y_pred = np.array([float(trilinear_function(x, k_1, k_2, k_3, x_0, x_1, x_2)) for x in x_data])
    y_pred = np.array([float(blankevoort_function(x, transition_length, k_1, x0)) for x in x_data])
    
    # Squared error term
    squared_error = np.sum((y_data - y_pred)**2)
    
    # L1 regularization term
    # l1_term = np.abs([k_1, k_2, k_3]).sum() + np.abs([x_1, x_2]).sum()
    l1_term = np.abs([transition_length, k_1, x0]).sum()
    if include_reg:
        l1 =  reg_coef*l1_term
        loss = squared_error + l1
    else:
        loss = squared_error
    
    return loss

def loss_jac(params, x_data, y_data, include_reg=True):
    """
    Compute gradient following the pseudocode:
    G = sum(2/n * r_i * J_i) + σ * sign_vec
    where r_i = ŷ_i - y_i is the residual
    """
    n = len(x_data)
    p = len(params)  # 6 parameters
    
    # Vectorized predictions and residuals
    # y_pred = np.array([trilinear_function(x, *params) for x in x_data])
    y_pred = np.array([blankevoort_function(x, *params) for x in x_data])
    residuals = y_pred - y_data
    
    # Vectorized Jacobian computation
    # Stack all Jacobians into a matrix of shape (n, p)
    # J_matrix = np.array([trilinear_function_jac(x, *params) for x in x_data])
    J_matrix = np.array([blankevoort_function_jac(x, *params) for x in x_data])
    
    # Compute gradient: G = 2/n * sum(r_i * J_i) + σ * sign_vec
    G = 2/n * np.sum(residuals[:, np.newaxis] * J_matrix, axis=0)
    
    # Add L1 regularization subgradient
    if include_reg:
        G += reg_coef * np.sign(params)
    
    return G

def loss_hess(params, x_data, y_data, include_reg=True):
    """
    Compute full Hessian following the pseudocode:
    H_full = sum(2/n * (J_i ⊗ J_i + r_i * H_i))
    where J_i is the Jacobian and H_i is the model Hessian at data point i
    """
    n = len(x_data)
    p = len(params)  # 6 parameters
    
    # Vectorized predictions and residuals
    y_pred = np.array([trilinear_function(x, *params) for x in x_data])
    residuals = y_pred - y_data
    
    # Vectorized Jacobian and Hessian computation
    # Stack all Jacobians into a matrix of shape (n, p)
    J_matrix = np.array([trilinear_function_jac(x, *params) for x in x_data])
    
    # Stack all Hessians into a 3D array of shape (n, p, p)
    H_matrix = np.array([trilinear_function_hess(x, *params) for x in x_data])
    
    # Compute full Hessian: H_full = 2/n * sum(J_i ⊗ J_i + r_i * H_i)
    # Gauss-Newton term: J_matrix.T @ J_matrix
    H_gn = J_matrix.T @ J_matrix
    
    # Model Hessian term: sum(r_i * H_i)
    H_model = np.sum(residuals[:, np.newaxis, np.newaxis] * H_matrix, axis=0)
    
    # Combine both terms
    H_full = 2/n * (H_gn + H_model)

    if include_reg:
        H_full += reg_coef * np.eye(p)
    
    return H_full
