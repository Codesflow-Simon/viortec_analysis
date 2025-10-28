from src.ligament_models.blankevoort import BlankevoortFunction
import numpy as np
import matplotlib.pyplot as plt
from src.statics_solver.models.statics_model import KneeModel


def visualize_ligament_curves(config, samples, data, ls_result=None):
    """Plot MCL and LCL tension vs elongation curves for ground truth, MCMC samples, mean, and least squares."""
    
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
        mcl_params = sample[:4]  # First 4 parameters are MCL
        lcl_params = sample[4:]  # Last 4 parameters are LCL
        
        lcl_func = BlankevoortFunction(lcl_params, )
        mcl_func = BlankevoortFunction(mcl_params, )
        
        sample_lcl_tensions.append(lcl_func(lcl_elongation_range))
        sample_mcl_tensions.append(mcl_func(mcl_elongation_range))
    
    # Calculate mean sample
    mean_mcl_params = np.mean(samples[:, :4], axis=0)
    mean_lcl_params = np.mean(samples[:, 4:], axis=0)
    
    mean_lcl_func = BlankevoortFunction(mean_lcl_params, )
    mean_mcl_func = BlankevoortFunction(mean_mcl_params, )
    
    mean_lcl_tension = mean_lcl_func(lcl_elongation_range)
    mean_mcl_tension = mean_mcl_func(mcl_elongation_range)
    
    # Calculate least squares curves if available
    if ls_result is not None:
        ls_lcl_func = ls_result['lcl_function']
        ls_mcl_func = ls_result['mcl_function']
        ls_lcl_tension = ls_lcl_func(lcl_elongation_range)
        ls_mcl_tension = ls_mcl_func(mcl_elongation_range)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # LCL plot
    ax1.plot(lcl_elongation_range, gt_lcl_tension, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    ax1.plot(lcl_elongation_range, mean_lcl_tension, 'r--', linewidth=2, label='Mean MCMC Sample')
    
    if ls_result is not None:
        ax1.plot(lcl_elongation_range, ls_lcl_tension, 'g-', linewidth=2, label='Least Squares', alpha=0.8)
    
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
    
    if ls_result is not None:
        ax2.plot(mcl_elongation_range, ls_mcl_tension, 'g-', linewidth=2, label='Least Squares', alpha=0.8)
    
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

def visualize_theta_force_curves(config, samples, data, ls_result=None):
    """Plot theta vs applied force with MCMC samples and least squares results."""
    
    # Ground truth data
    all_thetas = np.array(data['thetas'])
    all_forces = np.array(data['applied_force'])
    
    # Calculate MCMC sample predictions
    knee_config = config['mechanics'].copy()
    sample_predictions = []
    
    for sample in samples[::20]:  # Subsample for visualization
        mcl_params = sample[:4]  # First 4 parameters are MCL
        lcl_params = sample[4:]  # Last 4 parameters are LCL
        
        lcl_func = BlankevoortFunction(lcl_params, )
        mcl_func = BlankevoortFunction(mcl_params, )
        
        # Create model with sample parameters
        model = KneeModel(knee_config, log=False)
        model.build_geometry()
        model.build_ligament_forces(mcl_func, lcl_func)  # MCL on left, LCL on right
        results = model.calculate_thetas(all_thetas)
        sample_forces = results['applied_forces']
        model.reset()
        
        sample_predictions.append(sample_forces)
    
    # Calculate mean prediction
    mean_mcl_params = np.mean(samples[:, :4], axis=0)
    mean_lcl_params = np.mean(samples[:, 4:], axis=0)
    
    mean_lcl_func = BlankevoortFunction(mean_lcl_params, )
    mean_mcl_func = BlankevoortFunction(mean_mcl_params, )
    
    mean_model = KneeModel(knee_config, log=False)
    mean_model.build_geometry()
    mean_model.build_ligament_forces(mean_mcl_func, mean_lcl_func)  # MCL on left, LCL on right
    results = mean_model.calculate_thetas(all_thetas)
    mean_forces = results['applied_forces']
    mean_model.reset()
    
    # Calculate least squares prediction if available
    if ls_result is not None:
        ls_forces = ls_result['predicted_forces']
    else:
        ls_forces = None
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot ground truth as points only
    plt.scatter(np.degrees(all_thetas), all_forces, color='black', s=50, 
                label='Ground Truth', alpha=0.8, zorder=5)
    
    # Plot mean prediction as points only
    plt.scatter(np.degrees(all_thetas), mean_forces, color='red', s=30, 
                label='Mean MCMC Prediction', alpha=0.8, zorder=4)
    
    # Plot least squares prediction if available
    if ls_forces is not None:
        plt.scatter(np.degrees(all_thetas), ls_forces, color='green', s=30, 
                    label='Least Squares Prediction', alpha=0.8, zorder=3)
    
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