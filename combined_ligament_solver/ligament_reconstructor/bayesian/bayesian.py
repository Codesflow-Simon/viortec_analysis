import numpy as np
import yaml
from scipy.optimize import minimize
from .sampling_covariance import compute_sampling_covariance
from ..utils import *



def bayesian_update(result, config):
    fitted_function = result['fitted_function']
    ground_truth = result['ground_truth']

    x_data = result['x_data']
    y_data = result['y_data']
    funct_tuple = result['funct_tuple']
    data_config = config['data']

    prior_mean = np.array(config['prior_distribution_lcl_tri']['mean'])
    prior_std = np.array(config['prior_distribution_lcl_tri']['std'])
    prior_cov_matrix = np.diag(prior_std**2)

    # Compute posterior distribution using Bayesian update
    # For Gaussian distributions: posterior = prior * likelihood
    # The MAP estimate gives us the likelihood mean and covariance
    likelihood_mean = list(fitted_function.get_params().values())
    
    data_cov_matrix, std, samples, acceptance_rate = compute_sampling_covariance(
        likelihood_mean, x_data, y_data, funct_tuple, sigma_noise=data_config['y_noise'])
    likelihood_cov = data_cov_matrix

    # Bayesian update for Gaussian distributions
    prior_precision = np.linalg.inv(prior_cov_matrix)
    likelihood_precision = np.linalg.inv(likelihood_cov)
    
    posterior_precision = prior_precision + likelihood_precision
    posterior_cov_matrix = np.linalg.inv(posterior_precision)
    
    posterior_mean = posterior_cov_matrix @ (prior_precision @ prior_mean + likelihood_precision @ likelihood_mean)

    # Store distributions for plotting
    distributions = {
        'prior': {'mean': prior_mean, 'cov': prior_cov_matrix},
        'data': {'mean': likelihood_mean, 'cov': likelihood_cov},
        'posterior': {'mean': posterior_mean, 'cov': posterior_cov_matrix}
    }

    result.update({
        'distributions': distributions,
    })
    return result

