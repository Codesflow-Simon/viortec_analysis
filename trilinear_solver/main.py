import numpy as np
import yaml
from function import TrilinearFunction, trilinear_function, trilinear_function_jac, trilinear_function_hess
from function import BlankevoortFunction, blankevoort_function, blankevoort_function_jac, blankevoort_function_hess
from scipy.optimize import minimize
from plot import generate_plots, plot_hessian, plot_loss_cross_sections
from loss import loss, loss_jac, loss_hess
import constraints
from constraints import trilinear_constraints, blankevoort_constraints
from sampling_covariance import compute_sampling_covariance


import matplotlib.pyplot as plt
import os

def solve(x_data, y_data, initial_guess, funct_tuple, constraints_list):
    loss_func = lambda params: loss(params, x_data, y_data, funct=funct_tuple[0])
    jac_func = lambda params: loss_jac(params, x_data, y_data, funct=funct_tuple[0], funct_jac=funct_tuple[1])
    hess_func = lambda params: loss_hess(params, x_data, y_data, funct=funct_tuple[0], funct_jac=funct_tuple[1], funct_hess=funct_tuple[2])

    result = minimize(loss_func, initial_guess, method='trust-constr', 
                     jac=jac_func,
                     constraints=constraints_list)
    
    # print("Optimization result:")
    # print(f"Success: {result.success}")
    # print(f"Optimal parameters: {result.x}")
    # print("Lagrange multipliers:")
    # for i, multiplier in enumerate(result.v):
    #     print(f"λ_{i+1}: {multiplier[0]:.5f}")
    # print("")
    return result

def parameter_norm(params, gt_params):
    """
    Calculate the percentage deviation of fitted parameters from ground truth.
    
    Args:
        params: Fitted parameter values
        gt_params: Ground truth parameter values
        
    Returns:
        List of percentage deviations for each parameter
    """
    deviations = []
    for p, gt in zip(params, gt_params):
        dev = abs(p - gt) / gt
        deviations.append(dev)
    return np.array(deviations).mean()
    

def get_initial_guess(params):
    initial_guess = params
    # Add 10% noise to initial guess
    noise = np.random.normal(0, 0.2, len(initial_guess))  # 10% noise
    initial_guess = [val * (1 + noise[i]) for i, val in enumerate(initial_guess)]
    # initial_guess = enforce_constraints(initial_guess)
    return initial_guess

def setup_model(mode, config):
    if 'trilinear' in mode:
        param_names = constraints.trilinear_param_names
        funct_tuple = (trilinear_function, trilinear_function_jac, trilinear_function_hess)
        funct_class = TrilinearFunction
        
        params = {"k_1": float(config[mode]['modulus_1']) * float(config[mode]['cross_section']), 
                    "k_2": float(config[mode]['modulus_2']) * float(config[mode]['cross_section']), 
                    "k_3": float(config[mode]['modulus_3']) * float(config[mode]['cross_section']), 
                    "x_1": float(config[mode]['x_1']), 
                    "x_2": float(config[mode]['x_2']) }
        
        ground_truth = TrilinearFunction(params['k_1'], 
                                            params['k_2'], 
                                            params['k_3'], 
                                            params['x_1'], params['x_2'])
        constraints_list = trilinear_constraints

    elif 'blankevoort' in mode:
        param_names = constraints.blankevoort_param_names
        params = {"e_t": float(config[mode]['e_t']), 
                    "k_1": float(config[mode]['linear_elastic']) * float(config[mode]['cross_section'])}

        funct_tuple = (blankevoort_function, blankevoort_function_jac, blankevoort_function_hess)
        funct_class = BlankevoortFunction
        ground_truth = BlankevoortFunction(params['e_t'], params['k_1'])
        constraints_list = blankevoort_constraints
        
    return param_names, params, funct_tuple, funct_class, ground_truth, constraints_list

def main(config):
    mode = config['mode']
        
    param_names, params, funct_tuple, funct_class, ground_truth, constraints_list = setup_model(mode, config)

    data_config = config['data']

    # Generate data points
    x_data = np.linspace(data_config['x_min'], data_config['x_max'], data_config['n_points'])  # Sample points from before x_0 to after x_2
    y_data = np.array([float(ground_truth(x)) for x in x_data])
    x_noise = np.random.normal(0, data_config['x_noise'], len(x_data))
    y_noise = np.random.normal(0, data_config['y_noise'], len(y_data))
    x_data = x_data + x_noise
    y_data = y_data + y_noise

    # Solve the optimization problem
    initial_guess = get_initial_guess(params.values())
    opt_result = solve(x_data, y_data, initial_guess, funct_tuple, constraints_list)
    
    # Create TrilinearFunction object from optimization result
    fitted_function = funct_class(*opt_result.x)
    
    mean = opt_result.x

    parameter_norms = parameter_norm(opt_result.x, ground_truth.get_params().values())
    plt.close('all')

    return {
        'x_data': x_data,
        'y_data': y_data,
        'fitted_function': fitted_function,
        'ground_truth': ground_truth,
        'parameter_norms': parameter_norms,
        'funct_tuple': funct_tuple,
        'param_names': param_names,
        'success': opt_result.success,
    }

def bayesian_update(result, config):
    fitted_function = result['fitted_function']
    ground_truth = result['ground_truth']

    x_data = result['x_data']
    y_data = result['y_data']
    funct_tuple = result['funct_tuple']
    data_config = config['data']

    prior_mean = np.array(config['prior_distribution_lcl']['mean'])
    prior_std = np.array(config['prior_distribution_lcl']['std'])
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

    # Print shapes for debugging
    print("\nMatrix/Vector Shapes:")
    print("--------------------")
    print(f"prior_mean shape: {prior_mean.shape}")
    print(f"prior_cov_matrix shape: {prior_cov_matrix.shape}")
    print(f"prior_precision shape: {prior_precision.shape}")
    
    print(f"likelihood_mean shape: {np.array(likelihood_mean).shape}")
    print(f"likelihood_cov shape: {likelihood_cov.shape}") 
    print(f"likelihood_precision shape: {likelihood_precision.shape}")
    
    print(f"posterior_precision shape: {posterior_precision.shape}")
    print(f"posterior_cov_matrix shape: {posterior_cov_matrix.shape}")
    
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


def make_plots(result):
    x_data = result['x_data']
    y_data = result['y_data']
    fitted_function = result['fitted_function']
    ground_truth = result['ground_truth']
    distributions = result['distributions']
    param_names = result['param_names']

    posterior_mean = distributions['posterior']['mean']
    posterior_cov_matrix = distributions['posterior']['cov']

    data_mean = distributions['data']['mean']
    data_cov_matrix = distributions['data']['cov']
    data_std = np.sqrt(np.diag(data_cov_matrix))

    prior_mean = distributions['prior']['mean']
    prior_cov_matrix = distributions['prior']['cov']
    prior_std = np.sqrt(np.diag(prior_cov_matrix))

    # Generate plots
    print("\n=== Generating Plots ===")

    print("\nPosterior Distribution:")
    print("-----------------------")
    print(f"Mean:  {posterior_mean}")
    print(f"Std:   {np.sqrt(np.diag(posterior_cov_matrix))}")

    print("\nLikelihood/Data:")
    print("---------------")
    print(f"Covariance Shape: {data_cov_matrix.shape}")
    print(f"MAP Estimate:     {data_mean}")

    print("\nPrior Distribution:")
    print("------------------") 
    print(f"Mean:  {prior_mean}")
    print(f"Std:   {prior_std}")
    
    std = np.sqrt(np.diag(posterior_cov_matrix))
    generate_plots(x_data, y_data, fitted_function, ground_truth, std)
    
    plot_hessian(data_cov_matrix, path='./figures/sampling_covariance_heatmap.png')
        
    from bayesian_plots import plot_bayesian_distributions
    plot_bayesian_distributions(distributions, param_names)

if __name__ == "__main__":
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    result = main(config)
    result = bayesian_update(result, config)
    make_plots(result)