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
from src.ligament_reconstructor.mcmc_sampler import CompleteMCMCSampler

def analyse_data(config, data, constraint_manager):
    thetas = data['thetas']
    applied_forces = data['applied_force']

    # Pass knee configuration to sampler
    knee_config = config['mechanics']

    pre_compute_lcl_lengths = data['length_known_a']
    pre_compute_mcl_lengths = data['length_known_b']

    sampler = CompleteMCMCSampler(knee_config, constraint_manager)
    cov_matrix, std_params, samples, acceptance_rate = sampler.sample(thetas, applied_forces, lcl_lengths=pre_compute_lcl_lengths, 
        mcl_lengths=pre_compute_mcl_lengths, use_screening=True, screen_percentage=0.1, sigma_noise=1e2)
    
    print(f"MCMC completed with {len(samples)} samples")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Visualize results
    visualize_ligament_curves(config, samples, data)
    visualize_theta_force_curves(config, samples, data)
    
    return {
        'cov_matrix': cov_matrix,
        'std_params': std_params,
        'samples': samples,
        'acceptance_rate': acceptance_rate
    }

def main(config, constraints_config):
    from src.ligament_models.constraints import ConstraintManager
    constraint_manager_mcl = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    constraint_manager_lcl = ConstraintManager(constraints_config=constraints_config['blankevoort_lcl'])
    constraint_manager = (constraint_manager_mcl, constraint_manager_lcl)
    
    data = collect_data(config)
    
    result = analyse_data(config, data, constraint_manager)
    return result

def collect_data(config):
    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    
    # Initialize data collection lists
    data_lists = {
        'applied_force': [], 'applied_moment': [],
        'length_known_a': [], 'force_known_a': [], 'moment_known_a': [],  # LCL
        'length_known_b': [], 'force_known_b': [], 'moment_known_b': [],  # MCL
        'thetas': []  # Store thetas for visualization
    }
    moment_limit = 12_000  # In N(mm)
    
    def collect_at_theta(theta):
        """Helper function to collect data at a specific theta value"""
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()
        
        moment = float(solutions['applied_force'].get_moment().norm())
        if moment > moment_limit:
            return False, moment
        
        print(f"Theta: {np.degrees(theta):.2f}°, force: {solutions['applied_force'].get_force().norm():.1f}, moment: {moment:.1f}")
        
        # Collect basic data
        data_lists['applied_force'].append(float(solutions['applied_force'].get_force().norm()))
        data_lists['applied_moment'].append(moment)
        data_lists['length_known_a'].append(float(solutions['lig_springA_length']))
        data_lists['force_known_a'].append(float(solutions['lig_springA_force'].get_force().norm()))
        data_lists['length_known_b'].append(float(solutions['lig_springB_length']))
        data_lists['force_known_b'].append(float(solutions['lig_springB_force'].get_force().norm()))
        
        # Calculate ligament moments
        contact_point = model.knee_joint.get_contact_point(theta=theta)
        lig_force_a = solutions['lig_springA_force'].get_force()
        lig_force_b = solutions['lig_springB_force'].get_force()
        
        r_a = model.lig_bottom_pointA.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
        r_b = model.lig_bottom_pointB.convert_to_frame(model.tibia_frame) - contact_point.convert_to_frame(model.tibia_frame)
        
        moment_a = float((r_a.cross(lig_force_a.convert_to_frame(model.tibia_frame))).norm())
        moment_b = float((r_b.cross(lig_force_b.convert_to_frame(model.tibia_frame))).norm())
        
        data_lists['moment_known_a'].append(moment_a)
        data_lists['moment_known_b'].append(moment_b)
        data_lists['thetas'].append(theta)
        
        return True, moment
    
    # Collect data in both directions from theta=0
    print("\nCollecting ligament data over theta range...")
    
    # Positive direction (increasing theta)
    theta = 0
    while True:
        success, moment = collect_at_theta(theta)
        if not success:
            print(f"Moment too high at theta: {np.degrees(theta):.2f}°")
            break
        theta += 0.3 * np.pi/180
    
    # Negative direction (decreasing theta) - skip theta=0 to avoid duplicate
    theta = -0.3 * np.pi/180
    while True:
        success, moment = collect_at_theta(theta)
        if not success:
            print(f"Moment too high at theta: {np.degrees(theta):.2f}°")
            break
        theta -= 0.3 * np.pi/180
    
    return data_lists

def visualize_ligament_curves(config, samples, data):
    """Plot MCL and LCL tension vs elongation curves for ground truth, MCMC samples, and mean."""
    
    # Ground truth parameters
    ground_truth_lcl = config['blankevoort_lcl']
    ground_truth_mcl = config['blankevoort_mcl']
    
    # Create ground truth ligament functions
    gt_lcl_func = BlankevoortFunction(np.array([ground_truth_lcl['k'], ground_truth_lcl['alpha'], 
                                                ground_truth_lcl['l_0'], ground_truth_lcl['f_ref']]))
    gt_mcl_func = BlankevoortFunction(np.array([ground_truth_mcl['k'], ground_truth_mcl['alpha'], 
                                                ground_truth_mcl['l_0'], ground_truth_mcl['f_ref']]))
    
    # Calculate elongation from data
    lcl_lengths = np.array(data['length_known_a'])
    mcl_lengths = np.array(data['length_known_b'])
    lcl_forces = np.array(data['force_known_a'])
    mcl_forces = np.array(data['force_known_b'])
    
    lcl_elongations = lcl_lengths
    mcl_elongations = mcl_lengths
    
    # Determine appropriate elongation range
    # Start at l_0 (since we're plotting raw lengths, not elongation)
    lcl_start = ground_truth_lcl['l_0']
    mcl_start = ground_truth_mcl['l_0']
    
    # End just past the last data point
    lcl_end = np.max(lcl_elongations) + 5
    mcl_end = np.max(mcl_elongations) + 5
    
    # Create elongation ranges for plotting
    lcl_elongation_range = np.linspace(lcl_start, lcl_end, 200)
    mcl_elongation_range = np.linspace(mcl_start, mcl_end, 200)
    
    # Calculate ground truth curves
    gt_lcl_tension = gt_lcl_func(lcl_elongation_range)
    gt_mcl_tension = gt_mcl_func(mcl_elongation_range)
    
    # Calculate MCMC sample curves
    sample_lcl_tensions = []
    sample_mcl_tensions = []
    
    for sample in samples[::10]:  # Subsample for visualization (every 10th sample)
        lcl_params = sample[4:]  # Last 4 parameters are LCL
        mcl_params = sample[:4]  # First 4 parameters are MCL
        
        lcl_func = BlankevoortFunction(lcl_params, compile_derivatives=False)
        mcl_func = BlankevoortFunction(mcl_params, compile_derivatives=False)
        
        sample_lcl_tensions.append(lcl_func(lcl_elongation_range))
        sample_mcl_tensions.append(mcl_func(mcl_elongation_range))
    
    # Calculate mean sample
    mean_lcl_params = np.mean(samples[:, 4:], axis=0)
    mean_mcl_params = np.mean(samples[:, :4], axis=0)
    
    mean_lcl_func = BlankevoortFunction(mean_lcl_params, compile_derivatives=False)
    mean_mcl_func = BlankevoortFunction(mean_mcl_params, compile_derivatives=False)
    
    mean_lcl_tension = mean_lcl_func(lcl_elongation_range)
    mean_mcl_tension = mean_mcl_func(mcl_elongation_range)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # LCL plot
    ax1.plot(lcl_elongation_range, gt_lcl_tension, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(lcl_elongation_range, mean_lcl_tension, 'r--', linewidth=2, label='Mean MCMC Sample')
    
    for i, tension in enumerate(sample_lcl_tensions[:50]):  # Show first 50 samples
        ax1.plot(lcl_elongation_range, tension, 'b-', alpha=0.1, linewidth=0.5)
    
    # Add data points overlay
    ax1.scatter(lcl_elongations, lcl_forces, color='green', s=30, alpha=0.8, 
                label='Data Points', zorder=5)
    
    ax1.set_xlabel('Length (mm)')
    ax1.set_ylabel('Tension (N)')
    ax1.set_title('LCL Tension vs Length')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MCL plot
    ax2.plot(mcl_elongation_range, gt_mcl_tension, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax2.plot(mcl_elongation_range, mean_mcl_tension, 'r--', linewidth=2, label='Mean MCMC Sample')
    
    for i, tension in enumerate(sample_mcl_tensions[:50]):  # Show first 50 samples
        ax2.plot(mcl_elongation_range, tension, 'b-', alpha=0.1, linewidth=0.5)
    
    # Add data points overlay
    ax2.scatter(mcl_elongations, mcl_forces, color='green', s=30, alpha=0.8, 
                label='Data Points', zorder=5)
    
    ax2.set_xlabel('Length (mm)')
    ax2.set_ylabel('Tension (N)')
    ax2.set_title('MCL Tension vs Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ligament_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_theta_force_curves(config, samples, data):
    """Plot theta vs applied force with MCMC samples."""
    
    # Ground truth data
    all_thetas = np.array(data['thetas'])
    all_forces = np.array(data['applied_force'])
    
    # Calculate MCMC sample predictions
    knee_config = config['mechanics'].copy()
    sample_predictions = []
    
    for sample in samples[::20]:  # Subsample for visualization
        lcl_params = sample[4:]  # Last 4 parameters are LCL
        mcl_params = sample[:4]  # First 4 parameters are MCL
        
        lcl_func = BlankevoortFunction(lcl_params, compile_derivatives=False)
        mcl_func = BlankevoortFunction(mcl_params, compile_derivatives=False)
        
        # Create model with sample parameters
        
        sample_forces = []
        for theta in all_thetas:
            knee_config['theta'] = theta
            model = KneeModel(knee_config, lcl_func, mcl_func, log=False)
            solutions = model.solve()
            sample_forces.append(float(solutions['applied_force'].get_force().norm()))
        
        sample_predictions.append(sample_forces)
    
    # Calculate mean prediction
    mean_lcl_params = np.mean(samples[:, 4:], axis=0)
    mean_mcl_params = np.mean(samples[:, :4], axis=0)
    
    mean_lcl_func = BlankevoortFunction(mean_lcl_params, compile_derivatives=False)
    mean_mcl_func = BlankevoortFunction(mean_mcl_params, compile_derivatives=False)
    
    mean_forces = []
    for theta in all_thetas:
        knee_config['theta'] = theta
        mean_model = KneeModel(knee_config, mean_lcl_func, mean_mcl_func, log=False)
        solutions = mean_model.solve()
        mean_forces.append(float(solutions['applied_force'].get_force().norm()))
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot ground truth as points only
    plt.scatter(np.degrees(all_thetas), all_forces, color='black', s=50, 
                label='Ground Truth', alpha=0.8, zorder=5)
    
    # Plot mean prediction as points only
    plt.scatter(np.degrees(all_thetas), mean_forces, color='red', s=30, 
                label='Mean MCMC Prediction', alpha=0.8, zorder=4)
    
    # Plot sample predictions as points only
    for i, forces in enumerate(sample_predictions[:100]):  # Show first 100 samples
        plt.scatter(np.degrees(all_thetas), forces, color='blue', s=10, alpha=0.1, zorder=1)
    
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Applied Force (N)')
    plt.title('Theta vs Applied Force: Ground Truth vs MCMC Samples')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('theta_force_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

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
