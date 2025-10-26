from src.statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament
from src.ligament_reconstructor.utils import get_params_from_config
from src.ligament_models.blankevoort import BlankevoortFunction
from src.ligament_models.constraints import ConstraintManager
from src.ligament_reconstructor.mcmc_sampler import MCMCSampler
import itertools
import json
from datetime import datetime
import os


def generate_synthetic_data(config, thetas):
    """
    Generate synthetic ligament data by varying theta angles.
    
    Args:
        config: Configuration dictionary
        thetas: Array of theta angles to test
        
    Returns:
        dict: Contains length_estimates, force_estimates, reference_point
    """
    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    
    length_estimates_a = []  # LCL
    force_estimates_a = []
    length_estimates_b = []  # MCL
    force_estimates_b = []

    # First get reference force at theta=0
    mechanics = config['mechanics'].copy()
    mechanics['theta'] = 0
    model = KneeModel(mechanics, lig_left, lig_right, log=False)
    solutions = model.solve()
    reference_point = float(solutions['lig_springA_force'].get_force().norm())

    for theta in thetas:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()

        length_estimates_a.append(float(solutions['lig_springA_length']))
        force_estimates_a.append(float(solutions['lig_springA_force'].get_force().norm()))

        length_estimates_b.append(float(solutions['lig_springB_length']))
        force_estimates_b.append(float(solutions['lig_springB_force'].get_force().norm()))

    return {
        'length_estimates_a': np.array(length_estimates_a, dtype=np.float64),
        'force_estimates_a': np.array(force_estimates_a, dtype=np.float64),
        'length_estimates_b': np.array(length_estimates_b, dtype=np.float64),
        'force_estimates_b': np.array(force_estimates_b, dtype=np.float64),
        'reference_point': reference_point
    }


def add_noise_and_process_data(data, config):
    """
    Add noise to force data and process for reconstruction.
    
    Args:
        data: Dictionary from generate_synthetic_data
        config: Configuration dictionary
        
    Returns:
        dict: Processed data ready for reconstruction
    """
    length = data['length_estimates_a']
    force = data['force_estimates_a']
    
    # Add noise
    force_noisy = force + np.random.normal(0, config['data']['y_noise'], len(force))
    
    # Calculate relative force
    relative_force = force_noisy - data['reference_point']
    
    # Sort data
    def sort_data(length, force, relative_force):
        sort_idx = np.argsort(length)
        return length[sort_idx], force[sort_idx], relative_force[sort_idx]
        
    length_sorted, force_sorted, relative_force_sorted = sort_data(length, force, relative_force)
    
    return {
        'length': length_sorted,
        'force': force_sorted,
        'relative_force': relative_force_sorted,
        'reference_point': data['reference_point']
    }

def remove_outliers(samples):
    """
    Remove outliers from MCMC samples using IQR method.
    
    Args:
        samples: MCMC samples array
        
    Returns:
        samples_clean: Samples with outliers removed
    """
    Q1 = np.percentile(samples, 25, axis=0)
    Q3 = np.percentile(samples, 75, axis=0)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = np.all((samples >= lower_bound) & (samples <= upper_bound), axis=1)
    samples_clean = samples[mask]
    print(f"Removed {(~mask).sum()} outlier samples out of {len(mask)} total samples")
    return samples_clean

def plot_experiment_results(results):
    """Plot results for a single experiment."""
    data = results['data']
    function = results['reconstruction']['function']
    gt_params = results['ground_truth']
    samples = results['mcmc']['samples']
    
    plt.figure(figsize=(12, 5))
    
    # Main plot
    plt.subplot(1, 2, 1)
    plt.scatter(data['length'], data['relative_force'], c='r', label='Data', s=8, alpha=0.5)
    
    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(data['length'])), 
                        max(gt_params['l_0']*1.2, np.max(data['length'])), 100)
    plt.plot(x_data, function(x_data), c='b', label='Reconstructed Model')
    
    # Plot ground truth
    function.set_params(np.array(list(gt_params.values())))
    plt.plot(x_data, function(x_data), c='g', label='Ground Truth', linestyle='--')

    # Plot MCMC samples if available
    if samples is not None: 
        samples_to_plot = min(100, len(samples))
        plot_indices = np.random.choice(len(samples), samples_to_plot, replace=False)
        for idx in plot_indices:
            sample = samples[idx]
            plt.plot(x_data, function.vectorized_function(x_data, sample).flatten(), 
                    c='grey', alpha=0.1)
        
    plt.legend()
    plt.xlabel('Ligament Length')
    plt.ylabel('Ligament Relative Force')
    plt.title('Ligament Force vs Length')
    
    # Parameter comparison
    plt.subplot(1, 2, 2)
    if samples is not None and results['mcmc']['param_stats']:
        param_names = list(results['mcmc']['param_stats'].keys())
        means = [results['mcmc']['param_stats'][p]['mean'] for p in param_names]
        stds = [results['mcmc']['param_stats'][p]['std'] for p in param_names]
        gts = [results['mcmc']['param_stats'][p]['gt'] for p in param_names]
        
        x_pos = np.arange(len(param_names))
        plt.errorbar(x_pos, means, yerr=stds, fmt='o', capsize=5, label='MCMC Mean ± Std')
        plt.scatter(x_pos, gts, c='red', s=100, marker='x', label='Ground Truth')
        plt.xticks(x_pos, param_names, rotation=45)
        plt.ylabel('Parameter Value')
        plt.title('Parameter Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()