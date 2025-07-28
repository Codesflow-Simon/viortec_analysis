import numpy as np
from scipy.optimize import minimize
from scipy.stats import bootstrap
from loss import loss, loss_jac, loss_hess
import constraints

def compute_sampling_covariance(map_params, x_data, y_data, funct_tuple):
