import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from itertools import product

# Add the parent directory to the path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main, setup_model, get_initial_guess, solve

def run_single_experiment(args):
    """
    Run a single experiment with given x_min and x_max values.
    This function is designed to be called in parallel.
    """
    x_min, x_max, config, mode = args
    
    # Skip invalid combinations where x_min >= x_max
    if x_min >= x_max:
        return x_min, x_max, np.nan, 0
    
    # Create a copy of config to avoid modifying the original
    config_copy = config.copy()
    config_copy['data'] = config_copy['data'].copy()
    config_copy['data']['x_min'] = x_min
    config_copy['data']['x_max'] = x_max
    
    # Calculate number of points based on interval size
    base_points = config_copy['data']['n_points']  # Original number for full range
    points_per_unit = base_points / 0.2  # Points per unit strain (0.2 is full range)
    config_copy['data']['n_points'] = int(points_per_unit * (x_max - x_min))
    
    try:
        results = [main(config_copy) for _ in range(10)]
        loss_value = np.mean([float(r['parameter_norms']) for r in results])
        success_rate = np.mean([1 if r['success'] else 0 for r in results])
        return x_min, x_max, loss_value, success_rate
    except Exception as e:
        print(f"Error in experiment x_min={x_min:.3f}, x_max={x_max:.3f}: {e}")
        return x_min, x_max, np.nan, 0

@lru_cache(maxsize=128)
def get_cached_model_setup(mode_key):
    """
    Cache the model setup to avoid repeated computation.
    """
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    param_names, params, funct_tuple, funct_class, ground_truth, constraints_list = setup_model(mode, config)
    return param_names, params, funct_tuple, funct_class, ground_truth, constraints_list

def run_x_range_experiment_parallel(n_workers=None, early_stopping_threshold=0.01):
    """
    Run experiments across different x_min and x_max values using parallel processing.
    """
    
    # Load the base configuration
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Define the range of x values to test
    x_range = np.linspace(0.0, 0.2, 20)  # 0% to 20% strain
    
    # Initialize square arrays to store results
    n = len(x_range)
    loss_values = np.full((n, n), np.nan)
    success_rates = np.zeros((n, n))
    
    # Get model setup once (but don't pass to workers)
    mode = config['mode']
    
    # Generate all valid combinations
    valid_combinations = []
    for i, x_min in enumerate(x_range):
        for j, x_max in enumerate(x_range):
            if x_min < x_max:  # Only valid combinations
                valid_combinations.append((x_min, x_max, i, j))
    
    print(f"Running {len(valid_combinations)} experiments using parallel processing...")
    print(f"Using {n_workers or 'auto'} workers")
    
    start_time = time.time()
    completed_experiments = 0
    
    # Use parallel processing
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Prepare arguments for each experiment (only pass what's needed)
        experiment_args = [
            (x_min, x_max, config, mode)
            for x_min, x_max, _, _ in valid_combinations
        ]
        
        # Submit all experiments
        future_to_experiment = {
            executor.submit(run_single_experiment, args): (x_min, x_max, i, j)
            for args, (x_min, x_max, i, j) in zip(experiment_args, valid_combinations)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_experiment):
            x_min, x_max, i, j = future_to_experiment[future]
            try:
                result_x_min, result_x_max, loss_value, success_rate = future.result()
                loss_values[i, j] = loss_value
                success_rates[i, j] = success_rate
                
                completed_experiments += 1
                elapsed_time = time.time() - start_time
                avg_time_per_exp = elapsed_time / completed_experiments
                remaining_exps = len(valid_combinations) - completed_experiments
                eta = remaining_exps * avg_time_per_exp
                
                print(f"Progress: {completed_experiments}/{len(valid_combinations)} "
                      f"({completed_experiments/len(valid_combinations)*100:.1f}%) - "
                      f"x_min: {x_min:.3f}, x_max: {x_max:.3f}, "
                      f"loss: {loss_value:.3f}, success: {success_rate}, "
                      f"ETA: {eta/60:.1f}min")
                
                # Early stopping check (optional)
                if early_stopping_threshold and loss_value < early_stopping_threshold:
                    print(f"Early stopping: Found loss < {early_stopping_threshold}")
                    break
                    
            except Exception as e:
                print(f"Error processing result: {e}")
                loss_values[i, j] = np.nan
                success_rates[i, j] = 0
    
    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")
    print(f"Average time per experiment: {total_time/len(valid_combinations):.2f} seconds")

    # Create plots
    create_loss_plots(x_range, x_range, loss_values, success_rates)
    
    return x_range, x_range, loss_values, success_rates

def run_x_range_experiment():
    """
    Run experiments across different x_min and x_max values and plot the loss landscape.
    This is the original sequential version for comparison.
    """
    
    # Load the base configuration
    with open('../config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Define the range of x values to test
    x_range = np.linspace(0.0, 0.2, 20)  # 0% to 20% strain
    
    # Initialize square arrays to store results
    n = len(x_range)
    loss_values = np.zeros((n, n))
    success_rates = np.zeros((n, n))
    
    # Get model setup once
    mode = config['mode']
    param_names, params, funct_tuple, funct_class, ground_truth, constraints_list = setup_model(mode, config)
    
    # Update number of data points based on interval size
    base_points = config['data']['n_points']  # Original number for full range
    points_per_unit = base_points / 0.2  # Points per unit strain (0.2 is full range)
    
    print(f"Running experiments for {n} x values...")
    print(f"Total valid experiments: {(n * (n-1))//2}")  # Only cases where min < max
    
    start_time = time.time()
    
    # Iterate over square domain
    for i, x_min in enumerate(x_range):
        for j, x_max in enumerate(x_range):
            
            # Skip invalid combinations where x_min >= x_max
            if x_min >= x_max:
                loss_values[i, j] = np.nan
                success_rates[i, j] = 0
                continue

            config['data']['x_min'] = x_min
            config['data']['x_max'] = x_max
            config['data']['n_points'] = int(points_per_unit * (x_max - x_min))
            
            result = main(config)
                
            # Store results
            loss_values[i, j] = float(result['parameter_norms'])
            success_rates[i, j] = 1 if result['success'] else 0
            
            elapsed_time = time.time() - start_time
            progress = i*len(x_range) + j + 1
            total_experiments = n * n
            eta = (elapsed_time / progress) * (total_experiments - progress)
            
            print(f"Progress: {progress}/{total_experiments} - x_min: {x_min:.3f}, x_max: {x_max:.3f}, "
                  f"loss: {loss_values[i, j]:.3f}, success: {success_rates[i, j]}, ETA: {eta/60:.1f}min")

    total_time = time.time() - start_time
    print(f"\nExperiment completed in {total_time/60:.1f} minutes")

    # Create plots
    create_loss_plots(x_range, x_range, loss_values, success_rates)
    
    return x_range, x_range, loss_values, success_rates

def create_loss_plots(x_min_range, x_max_range, loss_values, success_rates):
    """
    Create plots showing the loss landscape across x_min and x_max space.
    """
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Loss heatmap
    im1 = ax1.contourf(x_max_range, x_min_range, loss_values, levels=20, cmap='viridis')
    ax1.set_xlabel('x_max (strain)')
    ax1.set_ylabel('x_min (strain)')
    ax1.set_title('Loss Landscape Across x_min and x_max')
    plt.colorbar(im1, ax=ax1, label='Loss Value')
    
    # Plot 2: Success rate heatmap
    im2 = ax2.contourf(x_max_range, x_min_range, success_rates, levels=[0, 0.5, 1], cmap='RdYlGn')
    ax2.set_xlabel('x_max (strain)')
    ax2.set_ylabel('x_min (strain)')
    ax2.set_title('Optimization Success Rate')
    plt.colorbar(im2, ax=ax2, label='Success Rate')
    
    # Plot 3: Loss vs x_max for different x_min values
    for i, x_min in enumerate(x_min_range):
        valid_indices = ~np.isnan(loss_values[i, :])
        if np.any(valid_indices):
            ax3.plot(x_max_range[valid_indices], loss_values[i, valid_indices], 
                    label=f'x_min={x_min:.3f}', marker='o', markersize=3)
    ax3.set_xlabel('x_max (strain)')
    ax3.set_ylabel('Loss Value')
    ax3.set_title('Loss vs x_max for Different x_min Values')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Loss vs x_min for different x_max values
    for j, x_max in enumerate(x_max_range):
        valid_indices = ~np.isnan(loss_values[:, j])
        if np.any(valid_indices):
            ax4.plot(x_min_range[valid_indices], loss_values[valid_indices, j], 
                    label=f'x_max={x_max:.3f}', marker='s', markersize=3)
    ax4.set_xlabel('x_min (strain)')
    ax4.set_ylabel('Loss Value')
    ax4.set_title('Loss vs x_min for Different x_max Values')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = '../figures'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/x_range_loss_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total experiments: {np.sum(~np.isnan(loss_values))}")
    print(f"Successful optimizations: {np.sum(success_rates)}")
    print(f"Success rate: {np.sum(success_rates) / np.sum(~np.isnan(loss_values)):.2%}")
    print(f"Average loss: {np.nanmean(loss_values):.2f}")
    print(f"Min loss: {np.nanmin(loss_values):.2f}")
    print(f"Max loss: {np.nanmax(loss_values):.2f}")
    
    # Find best parameters
    min_loss_idx = np.nanargmin(loss_values)
    min_loss_i, min_loss_j = np.unravel_index(min_loss_idx, loss_values.shape)
    best_x_min = x_min_range[min_loss_i]
    best_x_max = x_max_range[min_loss_j]
    best_loss = loss_values[min_loss_i, min_loss_j]
    
    print(f"\nBest parameters:")
    print(f"x_min: {best_x_min:.3f}")
    print(f"x_max: {best_x_max:.3f}")
    print(f"Loss: {best_loss:.2f}")

if __name__ == "__main__":
    # Choose which version to run
    use_parallel = True  # Set to False to use the original sequential version
    
    if use_parallel:
        # Run the parallel version
        x_min_range, x_max_range, loss_values, success_rates = run_x_range_experiment_parallel(
            n_workers=None,  # Use all available CPU cores
            early_stopping_threshold=None  # Set to a value like 0.01 for early stopping
        )
    else:
        # Run the original sequential version
        x_min_range, x_max_range, loss_values, success_rates = run_x_range_experiment()
    
    # Save results to file
    results = {
        'x_min_range': x_min_range.tolist(),
        'x_max_range': x_max_range.tolist(),
        'loss_values': loss_values.tolist(),
        'success_rates': success_rates.tolist()
    }
    
    import json
    with open('../figures/x_range_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nExperiment completed! Results saved to ../figures/x_range_experiment_results.json") 