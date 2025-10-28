from src.statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament, least_squares_optimize_complete_model
from src.ligament_reconstructor.utils import get_params_from_config
from src.ligament_models.blankevoort import BlankevoortFunction
import json
import pickle
from datetime import datetime
import os
import csv
from scipy.stats import gaussian_kde
from src.ligament_reconstructor.mcmc_sampler import CompleteMCMCSampler
from plotting_tools import visualize_ligament_curves, visualize_theta_force_curves

def analyse_data(config, data, constraint_manager):
    thetas = data['thetas']
    applied_forces = data['applied_force']

    # Pass knee configuration to sampler
    knee_config = config['mechanics']

    pre_compute_lcl_lengths = data['length_known_a']
    pre_compute_mcl_lengths = data['length_known_b']

    # Run least squares optimization first
    print("=" * 50)
    print("RUNNING LEAST SQUARES OPTIMIZATION")
    print("=" * 50)
    
    ls_result = least_squares_optimize_complete_model(
        thetas, applied_forces, pre_compute_lcl_lengths, pre_compute_mcl_lengths,
        constraint_manager, knee_config, sigma_noise=1e3
    )
    
    print(f"Least squares RMSE: {ls_result['rmse']:.2f}")
    print(f"Least squares MAE: {ls_result['mae']:.2f}")
    print(f"MCL parameters: {ls_result['mcl_params']}")
    print(f"LCL parameters: {ls_result['lcl_params']}")
    
    # Run MCMC sampling
    print("\n" + "=" * 50)
    print("RUNNING MCMC SAMPLING")
    print("=" * 50)
    
    sampler = CompleteMCMCSampler(knee_config, constraint_manager)
    cov_matrix, std_params, samples, acceptance_rate = sampler.sample(
        thetas, applied_forces, 
        lcl_lengths=pre_compute_lcl_lengths, 
        mcl_lengths=pre_compute_mcl_lengths, 
        use_screening=True, 
        screen_percentage=0.1, 
        sigma_noise=1e3,
        ls_result=ls_result
    )
    
    print(f"MCMC completed with {len(samples)} samples")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON: LEAST SQUARES vs MCMC")
    print("=" * 50)
    print(f"Least squares RMSE: {ls_result['rmse']:.2f}")
    print(f"Least squares MAE:  {ls_result['mae']:.2f}")
    
    # Calculate MCMC mean parameters for comparison
    mcmc_mcl_params = np.mean(samples[:, :4], axis=0)
    mcmc_lcl_params = np.mean(samples[:, 4:], axis=0)
    print(f"\nParameter comparison:")
    print(f"MCL - LS: {ls_result['mcl_params']}")
    print(f"MCL - MCMC mean: {mcmc_mcl_params}")
    print(f"LCL - LS: {ls_result['lcl_params']}")
    print(f"LCL - MCMC mean: {mcmc_lcl_params}")
    
    # Visualize results
    visualize_ligament_curves(config, samples, data, ls_result)
    visualize_theta_force_curves(config, samples, data, ls_result)
    
    return {
        'cov_matrix': cov_matrix,
        'std_params': std_params,
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'least_squares_result': ls_result
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
    
    def collect_at_theta(theta, knee_model):
        """Helper function to collect data at a specific theta value"""
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        
        knee_model.assemble_equations(theta)
        solutions = knee_model.solve()
        
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
        contact_point = knee_model.knee_joint.get_contact_point(theta=theta)
        lig_force_a = solutions['lig_springA_force'].get_force()
        lig_force_b = solutions['lig_springB_force'].get_force()
        
        r_a = knee_model.lig_bottom_pointA.convert_to_frame(knee_model.tibia_frame) - contact_point.convert_to_frame(knee_model.tibia_frame)
        r_b = knee_model.lig_bottom_pointB.convert_to_frame(knee_model.tibia_frame) - contact_point.convert_to_frame(knee_model.tibia_frame)
        
        moment_a = float((r_a.cross(lig_force_a.convert_to_frame(knee_model.tibia_frame))).norm())
        moment_b = float((r_b.cross(lig_force_b.convert_to_frame(knee_model.tibia_frame))).norm())
        
        data_lists['moment_known_a'].append(moment_a)
        data_lists['moment_known_b'].append(moment_b)
        data_lists['thetas'].append(theta)
        
        return True, moment
    
    # Collect data in both directions from theta=0
    print("\nCollecting ligament data over theta range...")
    
    # Positive direction (increasing theta)
    theta = 0 * np.pi/180
    knee_model = KneeModel(config['mechanics'], log=False)
    knee_model.build_geometry()
    knee_model.build_ligament_forces(lig_left, lig_right)

    while True:
        success, moment = collect_at_theta(theta, knee_model)
        if not success:
            print(f"Moment too high at theta: {np.degrees(theta):.2f}°")
            break
        theta += 0.3 * np.pi/180
    
    # Negative direction (decreasing theta) - skip theta=0 to avoid duplicate
    theta = -0.3 * np.pi/180
    while True:
        success, moment = collect_at_theta(theta, knee_model)
        if not success:
            print(f"Moment too high at theta: {np.degrees(theta):.2f}°")
            break
        theta -= 0.3 * np.pi/180

    return data_lists

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
