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

def plot_results(result_data):
    length = result_data['data']['length']
    relative_force = result_data['data']['relative_force']
    function = result_data['function']
    samples = result_data['samples']
    gt_params = result_data['gt_params']
    reference_point = result_data['reference_point']
    
    plt.figure()
    plt.scatter(length, relative_force, c='r', label='Data', s=8, alpha=0.5)

    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), max(gt_params['l_0']*1.1, np.max(length)), 100)
    plt.plot(x_data, function(x_data), c='b', label='Model')

    print(gt_params)
    function.set_params(np.array(list(gt_params.values())))

    if samples is not None: 
        # Handle both list of arrays and 2D array cases
        if isinstance(samples, list):
            # Convert list of arrays to 2D array
            samples = np.array(samples)
        
        samples_to_plot = min(100, len(samples))
        plot_indices = np.random.choice(len(samples), samples_to_plot, replace=False)
        print(f"Plotting {len(plot_indices)} samples from {len(samples)} total samples")
        for idx in plot_indices:
            sample = samples[idx]
            plt.plot(x_data, function.vectorized_function(x_data, sample).flatten(), c='grey', alpha=0.1)
        
    plt.plot(x_data, function(x_data), c='g', label='Ground Truth', linestyle='--')
    plt.legend()
    plt.xlabel('Ligament Length')
    plt.ylabel('Ligament Relative Force')

    # Plot marginal distributions from Monte Carlo samples
    if samples is not None:
        # Handle both list of arrays and 2D array cases
        if isinstance(samples, list):
            # Convert list of arrays to 2D array
            samples = np.array(samples)
            
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        n_params = len(param_names)
        
        # Create subplots for marginal distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, param_name in enumerate(param_names):
            ax = axes[i]
            
            # Plot histogram of samples
            ax.hist(samples[:, i], bins=50, alpha=0.7, density=True, color='skyblue', edgecolor='black')
            
            # Add ground truth value as vertical line
            gt_value = gt_params[param_name]
            ax.axvline(gt_value, color='red', linestyle='--', linewidth=2, label=f'Ground Truth: {gt_value:.3f}')
            
            # Calculate and display statistics
            mean_val = np.mean(samples[:, i])
            std_val = np.std(samples[:, i])
            ax.axvline(mean_val, color='green', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.3f}')
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Density')
            ax.set_title(f'Marginal Distribution: {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('MCMC Marginal Distributions', fontsize=16, y=1.02)
       
        # Plot correlation matrix
        plt.figure(figsize=(8, 6))
        # Ensure samples is 2D array for correlation calculation
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)
        correlation_matrix = np.corrcoef(samples.T)
        im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, label='Correlation')
        plt.xticks(range(n_params), param_names)
        plt.yticks(range(n_params), param_names)
        plt.title('Parameter Correlation Matrix')
        
        # Add correlation values as text
        for i in range(n_params):
            for j in range(n_params):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}', 
                        ha='center', va='center', 
                        color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')
        
        plt.tight_layout()
        plt.show()

def plot_likelihood_comparison(result_data, sampler, config, ligament_type=''):
    """
    Plot likelihood surfaces comparing KDE and posterior methods.
    
    Args:
        result_data: Dictionary containing samples, gt_params, and data
        sampler: MCMCSampler instance for posterior evaluation
        config: Configuration dictionary
        ligament_type: String identifier for the ligament (e.g., 'LCL', 'MCL')
    """
    samples = result_data['samples']
    gt_params = result_data['gt_params']
    length = result_data['data']['length']
    relative_force = result_data['data']['relative_force']
    function = result_data.get('function')
    
    # Skip if no valid samples
    if samples is None or len(samples) == 0 or np.all(np.isnan(samples)):
        print(f"Skipping likelihood comparison plot for {ligament_type}: no valid samples")
        return
    
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    n_params = len(param_names)
    
    # Build KDE from samples
    try:
        kde = gaussian_kde(samples.T)
    except Exception as e:
        print(f"Could not build KDE for {ligament_type}: {e}")
        return
    
    # Create figure with subplots for parameter pairs
    # We'll plot a few key parameter pairs
    pairs_to_plot = [(0, 2), (0, 1), (1, 2), (2, 3)]  # (k, l_0), (k, alpha), (alpha, l_0), (l_0, f_ref)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, (i, j) in enumerate(pairs_to_plot):
        ax_kde = axes[0, idx]
        ax_posterior = axes[1, idx]
        
        param_i_name = param_names[i]
        param_j_name = param_names[j]
        
        # Get parameter ranges from samples
        i_min, i_max = np.percentile(samples[:, i], [1, 99])
        j_min, j_max = np.percentile(samples[:, j], [1, 99])
        
        # Expand range slightly
        i_range = i_max - i_min
        j_range = j_max - j_min
        i_min -= 0.1 * i_range
        i_max += 0.1 * i_range
        j_min -= 0.1 * j_range
        j_max += 0.1 * j_range
        
        # Create grid
        n_grid = 50
        i_grid = np.linspace(i_min, i_max, n_grid)
        j_grid = np.linspace(j_min, j_max, n_grid)
        I, J = np.meshgrid(i_grid, j_grid)
        
        # Evaluate KDE on grid
        kde_values = np.zeros_like(I)
        for gi in range(n_grid):
            for gj in range(n_grid):
                # Build full parameter vector with means for other params
                test_params = np.mean(samples, axis=0).copy()
                test_params[i] = I[gi, gj]
                test_params[j] = J[gi, gj]
                try:
                    kde_values[gi, gj] = kde.logpdf(test_params)
                except:
                    kde_values[gi, gj] = -np.inf
        
        # Plot KDE
        kde_values_finite = kde_values.copy()
        kde_values_finite[~np.isfinite(kde_values_finite)] = np.nanmin(kde_values_finite[np.isfinite(kde_values_finite)])
        contour_kde = ax_kde.contourf(I, J, kde_values_finite, levels=20, cmap='viridis')
        ax_kde.scatter(samples[:, i], samples[:, j], c='red', s=1, alpha=0.3, label='MCMC Samples')
        ax_kde.scatter(gt_params[param_i_name], gt_params[param_j_name], 
                      c='white', s=100, marker='*', edgecolor='black', linewidth=2, label='Ground Truth')
        ax_kde.set_xlabel(param_i_name)
        ax_kde.set_ylabel(param_j_name)
        ax_kde.set_title(f'KDE Log-Likelihood\n{param_i_name} vs {param_j_name}')
        ax_kde.legend(loc='upper right', fontsize=8)
        plt.colorbar(contour_kde, ax=ax_kde, label='Log-Likelihood')
        
        # Evaluate posterior on grid (if we have the necessary data)
        if function is not None and length is not None and relative_force is not None:
            posterior_values = np.zeros_like(I)
            for gi in range(n_grid):
                for gj in range(n_grid):
                    # Build full parameter vector with means for other params
                    test_params = np.mean(samples, axis=0).copy()
                    test_params[i] = I[gi, gj]
                    test_params[j] = J[gi, gj]
                    try:
                        posterior_values[gi, gj] = sampler.log_probability(
                            test_params, length, relative_force, function, 
                            config['data']['sigma_noise']
                        )
                    except:
                        posterior_values[gi, gj] = -np.inf
            
            # Plot Posterior
            posterior_values_finite = posterior_values.copy()
            posterior_values_finite[~np.isfinite(posterior_values_finite)] = np.nanmin(posterior_values_finite[np.isfinite(posterior_values_finite)])
            contour_post = ax_posterior.contourf(I, J, posterior_values_finite, levels=20, cmap='viridis')
            ax_posterior.scatter(samples[:, i], samples[:, j], c='red', s=1, alpha=0.3, label='MCMC Samples')
            ax_posterior.scatter(gt_params[param_i_name], gt_params[param_j_name], 
                               c='white', s=100, marker='*', edgecolor='black', linewidth=2, label='Ground Truth')
            ax_posterior.set_xlabel(param_i_name)
            ax_posterior.set_ylabel(param_j_name)
            ax_posterior.set_title(f'Posterior Log-Probability\n{param_i_name} vs {param_j_name}')
            ax_posterior.legend(loc='upper right', fontsize=8)
            plt.colorbar(contour_post, ax=ax_posterior, label='Log-Probability')
        else:
            ax_posterior.text(0.5, 0.5, 'No posterior data available', 
                            ha='center', va='center', transform=ax_posterior.transAxes)
    
    plt.suptitle(f'Likelihood Comparison: KDE vs Posterior {ligament_type}', fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load constraints configuration
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)

    reference_strains = [0.02, 0.04, 0.06, 0.08, 0.10]
    results = []

    reference_strains_lcl = [0.02, 0.04, 0.06, 0.08, 0.10]
    reference_strains_mcl = [0.02, 0.04, 0.06, 0.08, 0.10]
    
    for ref_strain_lcl in reference_strains_lcl:
        for ref_strain_mcl in reference_strains_mcl:
            config['blankevoort_lcl']['l_0'] = config['mechanics']['left_length']/(ref_strain_lcl + 1)
            config['blankevoort_mcl']['l_0'] = config['mechanics']['right_length']/(ref_strain_mcl + 1)
            results.append({
                'lcl_strain': ref_strain_lcl,
                'mcl_strain': ref_strain_mcl,
                'result': main(config, constraints_config)
            })

    # Tidy up and save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('results', exist_ok=True)
    
    # Save pickle
    with open(f'results/results_{timestamp}.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary CSV
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    with open(f'results/summary_{timestamp}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['ligament_type', 'lcl_strain', 'mcl_strain', 'acceptance_rate', 'initial_loss', 'final_loss', 'gt_log_likelihood_kde', 'gt_log_likelihood_posterior', 'max_theta_deg', 'min_theta_deg', 'lowest_force_theta_deg']
        for param in param_names:
            header.extend([f'{param}_mean', f'{param}_std', f'{param}_median', f'{param}_gt'])
        writer.writerow(header)
        
        for result in results:
            # Write a row for each ligament type (LCL and MCL)
            for lig_type in ['LCL', 'MCL']:
                lig_result = result['result'][lig_type]
                samples = lig_result['samples']
                theta_stats = lig_result['theta_stats']
                row = [
                    lig_type,
                    result['lcl_strain'], 
                    result['mcl_strain'], 
                    lig_result['acceptance_rate'],
                    lig_result['initial_loss'],
                    lig_result['final_loss'],
                    lig_result['gt_log_likelihood_kde'],
                    lig_result['gt_log_likelihood_posterior'],
                    np.degrees(theta_stats['max_theta']),
                    np.degrees(theta_stats['min_theta']),
                    np.degrees(theta_stats['lowest_force_theta'])
                ]
                for i, param in enumerate(param_names):
                    row.extend([
                        np.mean(samples[:, i]),
                        np.std(samples[:, i]),
                        np.median(samples[:, i]),
                        lig_result['gt_params'][param]
                    ])
                writer.writerow(row)
    
    # Save individual sample CSVs
    samples_dir = f'results/samples_{timestamp}'
    os.makedirs(samples_dir, exist_ok=True)
    for result in results:
        lcl = result['lcl_strain']
        mcl = result['mcl_strain']
        
        for lig_type in ['LCL', 'MCL']:
            samples = result['result'][lig_type]['samples']
            
            csv_file = f'{samples_dir}/samples_{lig_type.lower()}_lcl{lcl:.2f}_mcl{mcl:.2f}.csv'
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(param_names)
                writer.writerows(samples)
    
    print(f"Saved results to results/results_{timestamp}.pkl")
    print(f"Saved summary to results/summary_{timestamp}.csv")
    print(f"Saved sample CSVs to {samples_dir}/")
    
    # Example: Plot likelihood comparison for the first result
    # Uncomment the following lines to generate plots:
    # print("\nGenerating likelihood comparison plots for first result...")
    # first_result = results[0]
    # lcl_data = first_result['result']['LCL']
    # mcl_data = first_result['result']['MCL']
    # 
    # if lcl_data['sampler'] is not None:
    #     plot_likelihood_comparison(lcl_data, lcl_data['sampler'], config, ligament_type='LCL')
    # 
    # if mcl_data['sampler'] is not None:
    #     plot_likelihood_comparison(mcl_data, mcl_data['sampler'], config, ligament_type='MCL')

