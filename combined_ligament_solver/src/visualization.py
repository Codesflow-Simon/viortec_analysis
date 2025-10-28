import numpy as np
import matplotlib.pyplot as plt
from src.statics_model import KneeModel, blankevoort_func


def visualize_ligament_curves(config, samples, data, ls_result):
    """
    Plot length-force curves for MCL and LCL.
    - Figure 1 (LS-focused): GT (green dashed), LS (blue), and GT/LS data points.
    - Figure 2 (MCMC-focused): GT (green dashed), LS (blue), and many MCMC samples (red, alpha=0.1).

    Args:
        config: Configuration dictionary
        samples: MCMC samples (n_samples, 8) - first 4 are MCL, last 4 are LCL
        data: Data dictionary containing 'thetas' and 'applied_forces'
        ls_result: Least squares optimization results
    """
    # Create knee model for calculations
    knee_model = KneeModel(config['mechanics'], log=False)
    knee_model.build_geometry()
    
    # Get ground truth parameters from config
    mcl_gt = [33.5, 0.06, 89.43, 0]  # Ground truth MCL parameters
    lcl_gt = [42.8, 0.06, 59.528, 0]  # Ground truth LCL parameters
    
    # Get least squares parameters
    mcl_ls = ls_result['mcl_params']
    lcl_ls = ls_result['lcl_params']
    
    # Create absolute length range for plotting (around reference lengths)
    mcl_lengths = np.linspace(80, 100, 200)  # mm - around MCL reference length
    lcl_lengths = np.linspace(50, 70, 200)   # mm - around LCL reference length

    # Data thetas (ensure numpy arrays)
    thetas_data = np.asarray(data['thetas'])

    # Compute GT and LS points at thetas for scatter overlays
    gt_eval = knee_model.solve_applied(thetas_data, mcl_gt, lcl_gt)
    ls_eval = knee_model.solve_applied(thetas_data, mcl_ls, lcl_ls)
    
    # ------------------------------------------------------------
    # Figure 1: LS-focused plots with GT/LS curves and data points
    # ------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # MCL plot
    ax1.set_title('MCL Length-Force Curve')
    ax1.set_xlabel('Length (mm)')
    ax1.set_ylabel('Force (N)')
    
    # Ground truth MCL curve
    forces_gt_mcl = blankevoort_func(mcl_lengths, mcl_gt)
    ax1.plot(mcl_lengths, forces_gt_mcl, 'g--', linewidth=2, alpha=0.8, label='Ground Truth')
    
    # Least squares MCL curve
    forces_ls_mcl = blankevoort_func(mcl_lengths, mcl_ls)
    ax1.plot(mcl_lengths, forces_ls_mcl, 'b.-', linewidth=2, alpha=0.8, label='Least Squares')
    
    # Data points (GT and LS) for MCL
    ax1.scatter(gt_eval['mcl_lengths'], gt_eval['mcl_forces'], c='g', s=18, alpha=0.6, label='GT data')
    ax1.scatter(ls_eval['mcl_lengths'], ls_eval['mcl_forces'], c='b', s=18, alpha=0.6, label='LS data')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # LCL plot
    ax2.set_title('LCL Length-Force Curve')
    ax2.set_xlabel('Length (mm)')
    ax2.set_ylabel('Force (N)')
    
    # Ground truth LCL curve
    forces_gt_lcl = blankevoort_func(lcl_lengths, lcl_gt)
    ax2.plot(lcl_lengths, forces_gt_lcl, 'g--', linewidth=2, label='Ground Truth')
    
    # Least squares LCL curve
    forces_ls_lcl = blankevoort_func(lcl_lengths, lcl_ls)
    ax2.plot(lcl_lengths, forces_ls_lcl, 'b.-', linewidth=2, label='Least Squares')
    
    # Data points (GT and LS) for LCL
    ax2.scatter(gt_eval['lcl_lengths'], gt_eval['lcl_forces'], c='g', s=18, alpha=0.8, label='GT data')
    ax2.scatter(ls_eval['lcl_lengths'], ls_eval['lcl_forces'], c='b', s=18, alpha=0.8, label='LS data')
    
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()

    # ------------------------------------------------------------
    # Figure 2: MCMC-focused plots with many sample curves
    # ------------------------------------------------------------
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(13, 5))

    # MCL panel (MCMC)
    bx1.set_title('MCL Length-Force (MCMC)')
    bx1.set_xlabel('Length (mm)')
    bx1.set_ylabel('Force (N)')

    # Baselines
    bx1.plot(mcl_lengths, forces_gt_mcl, 'g--', linewidth=2, label='Ground Truth')
    bx1.plot(mcl_lengths, forces_ls_mcl, 'b.-', linewidth=2, label='Least Squares')

    # MCMC MCL samples
    for sample in samples:
        mcl_params = sample[:4]
        forces_mcmc_mcl = blankevoort_func(mcl_lengths, mcl_params)
        bx1.plot(mcl_lengths, forces_mcmc_mcl, 'r-', alpha=0.1, linewidth=0.6)
    bx1.legend()
    bx1.grid(True, alpha=0.3)

    # LCL panel (MCMC)
    bx2.set_title('LCL Length-Force (MCMC)')
    bx2.set_xlabel('Length (mm)')
    bx2.set_ylabel('Force (N)')

    bx2.plot(lcl_lengths, forces_gt_lcl, 'g--', linewidth=2, label='Ground Truth')
    bx2.plot(lcl_lengths, forces_ls_lcl, 'b.-', linewidth=2, label='Least Squares')

    for sample in samples:
        lcl_params = sample[4:]
        forces_mcmc_lcl = blankevoort_func(lcl_lengths, lcl_params)
        bx2.plot(lcl_lengths, forces_mcmc_lcl, 'r-', alpha=0.1, linewidth=0.6)
    bx2.legend()
    bx2.grid(True, alpha=0.3)

    plt.tight_layout()


def visualize_theta_force_curves(config, samples, data, ls_result):
    """
    Create applied forces vs theta plots.
    
    Args:
        config: Configuration dictionary
        samples: MCMC samples (n_samples, 8) - first 4 are MCL, last 4 are LCL
        data: Data dictionary containing thetas and applied_forces
        ls_result: Least squares optimization results
    """
    # Create knee model for calculations
    knee_model = KneeModel(config['mechanics'], log=False)
    knee_model.build_geometry()
    
    # Get ground truth parameters from config
    mcl_gt = [33.5, 0.06, 89.43, 0]  # Ground truth MCL parameters
    lcl_gt = [42.8, 0.06, 59.528, 0]  # Ground truth LCL parameters
    
    # Get least squares parameters
    mcl_ls = ls_result['mcl_params']
    lcl_ls = ls_result['lcl_params']
    
    # Get data and ensure they are numpy arrays
    thetas = np.asarray(data['thetas'])
    applied_forces = np.asarray(data['applied_forces'])
    
    # Sort theta values for proper line plotting
    theta_sorted_indices = np.argsort(thetas)
    thetas_sorted = thetas[theta_sorted_indices]
    applied_forces_sorted = applied_forces[theta_sorted_indices]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.title('Applied Forces vs Theta')
    plt.xlabel('Theta (rad)')
    plt.ylabel('Applied Force (N)')
    
    # Ground truth curve
    gt_result = knee_model.solve_applied(thetas_sorted, mcl_gt, lcl_gt)
    gt_forces = np.array(gt_result['applied_forces']).reshape(-1)
    plt.plot(thetas_sorted, gt_forces, 'g--', linewidth=2, label='Ground Truth')
    
    # Least squares curve
    ls_result_curve = knee_model.solve_applied(thetas_sorted, mcl_ls, lcl_ls)
    ls_forces = np.array(ls_result_curve['applied_forces']).reshape(-1)
    plt.plot(thetas_sorted, ls_forces, 'b.-', linewidth=2, label='Least Squares')
    
    # MCMC samples
    for i, sample in enumerate(samples):
        mcl_params = sample[:4]
        lcl_params = sample[4:]
        mcmc_result = knee_model.solve_applied(thetas_sorted, mcl_params, lcl_params)
        mcmc_forces = np.array(mcmc_result['applied_forces']).reshape(-1)
        plt.plot(thetas_sorted, mcmc_forces, 'r-', alpha=0.1, linewidth=0.5)
    
    # Data points as scatter (use original unsorted data for scatter)
    plt.scatter(thetas, applied_forces, color='black', s=30, alpha=0.7, label='Data', zorder=5)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
