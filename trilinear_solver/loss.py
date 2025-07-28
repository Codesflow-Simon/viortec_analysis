from function import TrilinearFunction, trilinear_function, trilinear_function_jac, trilinear_function_hess
from function import BlankevoortFunction, blankevoort_function, blankevoort_function_jac
import numpy as np
reg_coef = 1e-2
measurement_noise = 100

def loss(params, x_data, y_data, funct, include_reg=True):
    y_pred = np.array([float(funct(x, *params)) for x in x_data])
    squared_error = np.sum((y_data - y_pred)**2)
    l1_term = np.abs(params).sum()
    if include_reg:
        l1 =  reg_coef*l1_term
        loss = squared_error + l1
    else:
        loss = squared_error
    return loss/measurement_noise**2


def loss_jac(params, x_data, y_data, funct, funct_jac, include_reg=True):
    n = len(x_data)
    p = len(params)
    y_pred = np.array([funct(x, *params) for x in x_data])
    residuals = y_pred - y_data
    J_matrix = np.array([funct_jac(x, *params) for x in x_data])
    G = 2/n * np.sum(residuals[:, np.newaxis] * J_matrix, axis=0)
    if include_reg:
        G += reg_coef * np.sign(params)
    return G/measurement_noise**2


def loss_hess(params, x_data, y_data, funct, funct_jac, funct_hess, include_reg=True):
    n = len(x_data)
    p = len(params)
    y_pred = np.array([funct(x, *params) for x in x_data])
    J_matrix = np.array([funct_jac(x, *params) for x in x_data])
    H_gn = J_matrix.T @ J_matrix
    H_full = 2/n * H_gn
    if include_reg:
        H_full += reg_coef * np.eye(p)
    return H_full/measurement_noise**2
