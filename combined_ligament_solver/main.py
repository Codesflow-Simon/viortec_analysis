from src.statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament
from src.ligament_reconstructor.utils import get_params_from_config
from src.ligament_models.blankevoort import BlankevoortFunction


if __name__ == "__main__":    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load constraints configuration
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)

    from src.ligament_models.constraints import ConstraintManager
    constraint_manager_mcl = ConstraintManager(constraints_config=constraints_config['blankevoort_lcl'])
    constraint_manager_lcl = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])

    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    model = KneeModel(config['mechanics'], lig_left, lig_right, log=False)
    
    solutions = model.solve()
    # model.plot_model(show_forces=False)
    # plt.show()

    thetas = np.linspace(-np.radians(3), np.radians(3), 15)

    length_estimates_a = [] # LCL
    force_estimates_a = []

    length_estimates_b = [] # MCL
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

    length = np.array(length_estimates_a, dtype=np.float64)
    force = np.array(force_estimates_a, dtype=np.float64)

    force = force + np.random.normal(0, config['data']['y_noise'], len(force))

    print(f"Reference force: {reference_point}")
    relative_force = force - reference_point # We only measure relative changes in force

    def sort_data(length, force, relative_force):
        sort_idx = np.argsort(length)
        length = length[sort_idx]
        force = force[sort_idx]
        relative_force = relative_force[sort_idx]
        return length, force, relative_force
        
    length, force, relative_force = sort_data(length, force, relative_force)

    result_obj = reconstruct_ligament(length, relative_force, constraint_manager_mcl)
    function = result_obj['function']
    params = result_obj['params']
    print(result_obj['params'])

    gt_params = config['blankevoort_lcl'].copy()
    gt_params['f_ref'] = reference_point

    try:
        from src.ligament_reconstructor.mcmc_sampler import MCMCSampler
        sampler = MCMCSampler(constraint_manager=constraint_manager_mcl)
        # Use screened initialization with top 10% of candidates
        cov_matrix, std_params, samples, acceptance_rate = sampler.sample(
            params, length, relative_force, function, 
            sigma_noise=config['data']['sigma_noise'],
            use_screening=True,
            screen_percentage=0.05,
            visualize_only=True
        )
        print(f"Acceptance rate: {acceptance_rate}")
    except Exception as e:
        print(e)
        print("MCMC sampler failed")
        samples = None

    plt.figure()
    plt.scatter(length, relative_force, c='r', label='Data', s=8, alpha=0.5)

    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), np.max(length), 100)
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

    plt.show()

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
        plt.show()
        
        # Print parameter statistics
        print("\nMCMC Parameter Statistics:")
        print("-" * 50)
        for i, param_name in enumerate(param_names):
            mean_val = np.mean(samples[:, i])
            std_val = np.std(samples[:, i])
            gt_val = gt_params[param_name]
            print(f"{param_name:>8}: Mean={mean_val:8.3f}, Std={std_val:8.3f}, GT={gt_val:8.3f}")
        
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
