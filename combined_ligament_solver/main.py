from src.statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament
from src.ligament_reconstructor.utils import get_params_from_config
from src.ligament_models.blankevoort import BlankevoortFunction
import json
import pickle
from datetime import datetime
import os
import csv
from scipy.stats import gaussian_kde

def process_ligament(ligament_type, length_estimates, force_estimates, theta_list, gt_params, constraint_manager, config):
    """Process a single ligament (LCL or MCL) - run optimization and MCMC"""
    
    print(f"\n{'='*60}")
    print(f"Processing {ligament_type}")
    print(f"{'='*60}")
    
    # Check if we have any data
    if len(theta_list) == 0 or len(length_estimates) == 0 or len(force_estimates) == 0:
        print(f"WARNING: No data collected for {ligament_type}. The moment limit may be too low or initial configuration may be invalid.")
        print(f"Returning NaN values for {ligament_type}")
        
        # Return NaN result data
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        n_params = len(param_names)
        result_data = {
            'cov_matrix': np.full((n_params, n_params), np.nan),
            'std_params': np.full(n_params, np.nan),
            'samples': np.full((1, n_params), np.nan),
            'acceptance_rate': np.nan,
            'optimized_params': np.full(n_params, np.nan),
            'gt_params': gt_params,
            'reference_point': np.nan,
            'initial_loss': np.nan,
            'final_loss': np.nan,
            'gt_log_likelihood_kde': np.nan,
            'gt_log_likelihood_posterior': np.nan,
            'function': None,
            'sampler': None,
            'theta_stats': {
                'max_theta': np.nan,
                'min_theta': np.nan,
                'lowest_force_theta': np.nan
            },
            'data': {
                'length': np.array([]),
                'force': np.array([]),
                'relative_force': np.array([])
            }
        }
        return result_data
    
    length = np.array(length_estimates, dtype=np.float64)
    force = np.array(force_estimates, dtype=np.float64)
    
    # Track theta statistics before adding noise
    theta_array = np.array(theta_list)
    max_theta = float(np.max(theta_array))
    min_theta = float(np.min(theta_array))
    min_force_idx = np.argmin(force)
    lowest_force_theta = float(theta_array[min_force_idx])
    
    # Get reference force at theta=0
    reference_point = gt_params['f_ref']

    force = force + np.random.normal(0, config['data']['y_noise'], len(force))

    print(f"Reference force: {reference_point}")
    print(f"Theta range: [{np.degrees(min_theta):.2f}°, {np.degrees(max_theta):.2f}°]")
    print(f"Lowest force at theta: {np.degrees(lowest_force_theta):.2f}°")
    relative_force = force - reference_point # We only measure relative changes in force

    def sort_data(length, force, relative_force):
        sort_idx = np.argsort(length)
        length = length[sort_idx]
        force = force[sort_idx]
        relative_force = relative_force[sort_idx]
        return length, force, relative_force
        
    length, force, relative_force = sort_data(length, force, relative_force)

    print(length, force, relative_force)
    result_obj = reconstruct_ligament(length, relative_force, constraint_manager)
    function = result_obj['function']
    params = result_obj['params']
    initial_loss = result_obj['initial_loss']
    final_loss = result_obj['loss']
    print(result_obj['params'])
    print(f"Optimization: Initial loss={initial_loss:.3f}, Final loss={final_loss:.3f}")

    # Run MCMC
    from src.ligament_reconstructor.mcmc_sampler import MCMCSampler
    sampler = MCMCSampler(constraint_manager=constraint_manager)
    
    try:
        cov_matrix, std_params, samples, acceptance_rate = sampler.sample(
            params, length, relative_force, function, 
            sigma_noise=config['data']['sigma_noise'],
        )
        print(f"Acceptance rate: {acceptance_rate}")
    except ValueError as e:
        print(f"WARNING: MCMC sampling failed for {ligament_type}: {e}")
        print(f"Returning NaN values for MCMC results")
        
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        n_params = len(param_names)
        result_data = {
            'cov_matrix': np.full((n_params, n_params), np.nan),
            'std_params': np.full(n_params, np.nan),
            'samples': np.full((1, n_params), np.nan),
            'acceptance_rate': np.nan,
            'optimized_params': params,
            'gt_params': gt_params,
            'reference_point': reference_point,
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'gt_log_likelihood_kde': np.nan,
            'gt_log_likelihood_posterior': np.nan,
            'function': function,
            'sampler': sampler,
            'theta_stats': {
                'max_theta': max_theta,
                'min_theta': min_theta,
                'lowest_force_theta': lowest_force_theta
            },
            'data': {
                'length': length,
                'force': force,
                'relative_force': relative_force
            }
        }
        return result_data

    # Remove outliers using IQR method for each parameter
    if samples is not None:
        Q1 = np.percentile(samples, 25, axis=0)
        Q3 = np.percentile(samples, 75, axis=0)
        IQR = Q3 - Q1
        
        # Define bounds for outlier detection (1.5 * IQR)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Create mask for samples within bounds for all parameters
        mask = np.all((samples >= lower_bound) & (samples <= upper_bound), axis=1)
        
        # Filter samples
        samples = samples[mask]
        print(f"Removed {(~mask).sum()} outlier samples out of {len(mask)} total samples")

    # Print parameter statistics
    print(f"\nMCMC Parameter Statistics ({ligament_type}):")
    print("-" * 50)
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    for i, param_name in enumerate(param_names):
        mean_val = np.mean(samples[:, i])
        std_val = np.std(samples[:, i])
        gt_val = gt_params[param_name]
        print(f"{param_name:>8}: Mean={mean_val:8.3f}, Std={std_val:8.3f}, GT={gt_val:8.3f}")
    
    # Calculate ground truth log-likelihood using KDE
    try:
        kde = gaussian_kde(samples.T)
        gt_vector = np.array([gt_params[param] for param in param_names], dtype=np.float64)
        gt_log_likelihood_kde = float(kde.logpdf(gt_vector))
        print(f"\nGround truth log-likelihood (KDE): {gt_log_likelihood_kde:.3f}")
    except Exception as e:
        print(f"\nWarning: Could not calculate KDE log-likelihood: {e}")
        gt_log_likelihood_kde = np.nan
    
    # Calculate ground truth log-likelihood using posterior evaluation
    try:
        gt_vector = np.array([gt_params[param] for param in param_names], dtype=np.float64)
        gt_log_likelihood_posterior = float(sampler.log_probability(gt_vector, length, relative_force, function, config['data']['sigma_noise']))
        print(f"Ground truth log-probability (posterior): {gt_log_likelihood_posterior:.3f}")
    except Exception as e:
        print(f"Warning: Could not calculate posterior log-probability: {e}")
        gt_log_likelihood_posterior = np.nan
    
    # Prepare return data
    result_data = {
        'cov_matrix': cov_matrix,
        'std_params': std_params,
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'optimized_params': params,
        'gt_params': gt_params,
        'reference_point': reference_point,
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'gt_log_likelihood_kde': gt_log_likelihood_kde,
        'gt_log_likelihood_posterior': gt_log_likelihood_posterior,
        'function': function,
        'sampler': sampler,
        'theta_stats': {
            'max_theta': max_theta,
            'min_theta': min_theta,
            'lowest_force_theta': lowest_force_theta
        },
        'data': {
            'length': length,
            'force': force,
            'relative_force': relative_force
        }
    }
    
    return result_data

def main(config, constraints_config):
    from src.ligament_models.constraints import ConstraintManager
    constraint_manager_mcl = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    constraint_manager_lcl = ConstraintManager(constraints_config=constraints_config['blankevoort_lcl'])

    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    model = KneeModel(config['mechanics'], lig_left, lig_right, log=False)
    
    solutions = model.solve()
    model.plot_model(show_forces=True)
    plt.show()

    length_estimates_a = [] # LCL
    force_estimates_a = []

    length_estimates_b = [] # MCL
    force_estimates_b = []

    # Get reference forces at theta=0
    mechanics = config['mechanics'].copy()
    mechanics['theta'] = 0
    model = KneeModel(mechanics, lig_left, lig_right, log=False)
    solutions = model.solve()
    reference_point_lcl = float(solutions['lig_springA_force'].get_force().norm())
    reference_point_mcl = float(solutions['lig_springB_force'].get_force().norm())

    theta_list = []

    theta = 0
    moment_limit = 12_000 # In N(mm)
    
    # Collect data for both ligaments over theta range
    print("\nCollecting ligament data over theta range...")
    while True:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()

        moment = float(solutions['applied_force'].get_moment().norm())
        if moment > moment_limit:
            print(f"Moment too high at theta: {theta}")
            break
        else:
            print(f"Theta: {np.degrees(theta)}, force: {solutions['applied_force'].get_force().norm()}, moment: {moment}")
            length_estimates_a.append(float(solutions['lig_springA_length']))
            force_estimates_a.append(float(solutions['lig_springA_force'].get_force().norm()))

            length_estimates_b.append(float(solutions['lig_springB_length']))
            force_estimates_b.append(float(solutions['lig_springB_force'].get_force().norm()))
            theta_list.append(theta)

            theta += 1/3 * np.pi/180

    theta = 0
    while True:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()

        moment = float(solutions['applied_force'].get_moment().norm())
        if moment > moment_limit:
            print(f"Moment too high at theta: {theta}, moment: {moment}")
            break
        else:
            print(f"Theta: {np.degrees(theta)}, force: {solutions['applied_force'].get_force().norm()}, moment: {moment}")
            length_estimates_a.append(float(solutions['lig_springA_length']))
            force_estimates_a.append(float(solutions['lig_springA_force'].get_force().norm()))

            length_estimates_b.append(float(solutions['lig_springB_length']))
            force_estimates_b.append(float(solutions['lig_springB_force'].get_force().norm()))
            theta_list.append(theta)
        
            theta -= 1/3 * np.pi/180

    # Process LCL
    gt_params_lcl = config['blankevoort_lcl'].copy()
    gt_params_lcl['f_ref'] = reference_point_lcl
    result_lcl = process_ligament('LCL', length_estimates_a, force_estimates_a, theta_list, 
                                   gt_params_lcl, constraint_manager_lcl, config)
    
    # Process MCL
    gt_params_mcl = config['blankevoort_mcl'].copy()
    gt_params_mcl['f_ref'] = reference_point_mcl
    result_mcl = process_ligament('MCL', length_estimates_b, force_estimates_b, theta_list, 
                                   gt_params_mcl, constraint_manager_mcl, config)
    
    return {'LCL': result_lcl, 'MCL': result_mcl}

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load constraints configuration
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)

    results = []

    reference_strains_lcl = [0.06]
    reference_strains_mcl = [0.06]
    
    for ref_strain_lcl in reference_strains_lcl:
        for ref_strain_mcl in reference_strains_mcl:
            config['blankevoort_lcl']['l_0'] = config['mechanics']['left_length']/(ref_strain_lcl + 1)
            config['blankevoort_mcl']['l_0'] = config['mechanics']['right_length']/(ref_strain_mcl + 1)
            results.append({
                'lcl_strain': ref_strain_lcl,
                'mcl_strain': ref_strain_mcl,
                'result': main(config, constraints_config)
            })
