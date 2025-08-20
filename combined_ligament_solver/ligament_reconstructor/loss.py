import numpy as np
reg_coef = 1e-2

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

