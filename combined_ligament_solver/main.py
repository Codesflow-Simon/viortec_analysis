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
    # model.plot_model(show_forces=True)
    # plt.show()

    applied_force = []
    applied_moment = []

    length_estimates_a = [] # LCL
    force_known_a = []
    force_estimated_a = []
    moment_known_a = []

    length_estimates_b = [] # MCL
    force_known_b = []
    force_estimated_b = []
    moment_known_b = []
    
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
            applied_force.append(float(solutions['applied_force'].get_force().norm()))
            applied_moment.append(moment)

            length_estimates_a.append(float(solutions['lig_springA_length']))
            force_known_a.append(float(solutions['lig_springA_force'].get_force().norm()))
            force_estimated_a.append(float(solutions['estimated_lig_springA_force']))   

            length_estimates_b.append(float(solutions['lig_springB_length']))
            force_known_b.append(float(solutions['lig_springB_force'].get_force().norm()))
            force_estimated_b.append(float(solutions['estimated_lig_springB_force']))

            # Calculate ligament moments
            contact_point = model.knee_joint.get_contact_point(theta=theta)
            
            # Calculate moment arms for ligaments (distance from contact point to ligament attachment)
            lig_force_a = solutions['lig_springA_force'].get_force()
            lig_force_b = solutions['lig_springB_force'].get_force()
            
            # Moment = r x F, where r is from contact point to force application point
            # We'll use the tibia attachment point for the moment arm
            r_a = model.lig_bottom_pointA.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
            r_b = model.lig_bottom_pointB.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
            
            # Calculate moment magnitude (in 2D, cross product gives z-component)
            moment_a = float((r_a.cross(lig_force_a.convert_to_frame(model.tibia_frame))).norm())
            moment_b = float((r_b.cross(lig_force_b.convert_to_frame(model.tibia_frame))).norm())
            
            moment_known_a.append(moment_a)
            moment_known_b.append(moment_b)

            theta_list.append(theta)

            theta += 0.1 * np.pi/180

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
            applied_force.append(float(solutions['applied_force'].get_force().norm()))
            applied_moment.append(moment)
            
            length_estimates_a.append(float(solutions['lig_springA_length']))
            force_known_a.append(float(solutions['lig_springA_force'].get_force().norm()))
            force_estimated_a.append(float(solutions['estimated_lig_springA_force']))

            length_estimates_b.append(float(solutions['lig_springB_length']))
            force_known_b.append(float(solutions['lig_springB_force'].get_force().norm()))
            force_estimated_b.append(float(solutions['estimated_lig_springB_force']))

            # Calculate ligament moments
            contact_point = model.knee_joint.get_contact_point(theta=theta)
            
            # Calculate moment arms for ligaments
            lig_force_a = solutions['lig_springA_force'].get_force()
            lig_force_b = solutions['lig_springB_force'].get_force()
            
            # Moment = r x F
            r_a = model.lig_bottom_pointA.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
            r_b = model.lig_bottom_pointB.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
            
            # Calculate moment magnitude
            moment_a = float((r_a.cross(lig_force_a.convert_to_frame(model.tibia_frame))).norm())
            moment_b = float((r_b.cross(lig_force_b.convert_to_frame(model.tibia_frame))).norm())
            
            moment_known_a.append(moment_a)
            moment_known_b.append(moment_b)

            theta_list.append(theta)
        
            theta -= 1/3 * np.pi/180

    # Convert theta to degrees for plotting
    theta_degrees = [np.degrees(t) for t in theta_list]
    
    # Find indices where theta = 0
    zero_theta_indices = [i for i, theta in enumerate(theta_list) if abs(theta) < 1e-6]
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    plt.plot(theta_degrees, applied_force, 'b-', label='Applied Force')
    
    # Add labels and title
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')
    plt.title('Applied Force vs Joint Angle')
    plt.grid(True)
    plt.legend()
    
    # Create force vs theta plots for ligaments
    plt.figure(figsize=(12, 5))
    
    # LCL force vs theta
    plt.subplot(1, 2, 1)
    plt.scatter(theta_degrees, force_known_a, s=20, alpha=0.6, label='LCL Force', color='blue')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([theta_degrees[zero_theta_idx]], [force_known_a[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')
    plt.title('LCL Force vs Joint Angle')
    plt.grid(True)
    plt.legend()
    
    # MCL force vs theta
    plt.subplot(1, 2, 2)
    plt.scatter(theta_degrees, force_known_b, s=20, alpha=0.6, label='MCL Force', color='orange')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([theta_degrees[zero_theta_idx]], [force_known_b[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Force (N)')
    plt.title('MCL Force vs Joint Angle')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ligament_forces_vs_theta.png')
    
    # Create force vs theta plots for ligaments - moments
    plt.figure(figsize=(12, 5))
    
    # LCL moment vs theta
    plt.subplot(1, 2, 1)
    plt.scatter(theta_degrees, moment_known_a, s=20, alpha=0.6, label='LCL Moment', color='blue')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([theta_degrees[zero_theta_idx]], [moment_known_a[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Moment (N⋅mm)')
    plt.title('LCL Moment vs Joint Angle')
    plt.grid(True)
    plt.legend()
    
    # MCL moment vs theta
    plt.subplot(1, 2, 2)
    plt.scatter(theta_degrees, moment_known_b, s=20, alpha=0.6, label='MCL Moment', color='orange')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([theta_degrees[zero_theta_idx]], [moment_known_b[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Moment (N⋅mm)')
    plt.title('MCL Moment vs Joint Angle')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ligament_moments_vs_theta.png')
    plt.show()
    
    # Create elongation vs force plots for ligaments
    plt.figure(figsize=(12, 5))
    
    # Get rest lengths (l_0) for zero-force point
    l_0_lcl = config['blankevoort_lcl']['l_0']
    l_0_mcl = config['blankevoort_mcl']['l_0']
    
    # Generate ground truth curves extending to l_0
    min_length_lcl = min(min(length_estimates_a), l_0_lcl)
    max_length_lcl = max(length_estimates_a)
    length_range_lcl = np.linspace(min_length_lcl, max_length_lcl, 200)
    force_gt_lcl = [lig_left(l) for l in length_range_lcl]
    
    min_length_mcl = min(min(length_estimates_b), l_0_mcl)
    max_length_mcl = max(length_estimates_b)
    length_range_mcl = np.linspace(min_length_mcl, max_length_mcl, 200)
    force_gt_mcl = [lig_right(l) for l in length_range_mcl]
    
    # LCL plot
    plt.subplot(1, 2, 1)
    plt.scatter(length_estimates_a, force_known_a, s=20, alpha=0.6, label='Computed Force', color='blue')
    plt.plot(length_range_lcl, force_gt_lcl, 'g-', linewidth=2, label='Ground Truth Function')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([length_estimates_a[zero_theta_idx]], [force_known_a[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Elongation (m)')
    plt.ylabel('Force (N)')
    plt.title('LCL Force vs Elongation')
    plt.grid(True)
    plt.legend()
    
    # MCL plot
    plt.subplot(1, 2, 2)
    plt.scatter(length_estimates_b, force_known_b, s=20, alpha=0.6, label='Computed Force', color='blue')
    plt.plot(length_range_mcl, force_gt_mcl, 'g-', linewidth=2, label='Ground Truth Function')
    
    # Mark zero-theta point
    if zero_theta_indices:
        zero_theta_idx = zero_theta_indices[0]
        plt.scatter([length_estimates_b[zero_theta_idx]], [force_known_b[zero_theta_idx]], 
                   s=100, color='red', marker='*', label='Zero Theta', zorder=5)
    
    plt.xlabel('Elongation (m)')
    plt.ylabel('Force (N)')
    plt.title('MCL Force vs Elongation')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ligament_forces.png')
    plt.show()

    
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

    reference_strains_lcl = [0.02]
    reference_strains_mcl = [0.02]
    
    for ref_strain_lcl in reference_strains_lcl:
        for ref_strain_mcl in reference_strains_mcl:
            config['blankevoort_lcl']['l_0'] = config['mechanics']['left_length']/(ref_strain_lcl + 1)
            config['blankevoort_mcl']['l_0'] = config['mechanics']['right_length']/(ref_strain_mcl + 1)
            results.append({
                'lcl_strain': ref_strain_lcl,
                'mcl_strain': ref_strain_mcl,
                'result': main(config, constraints_config)
            })
