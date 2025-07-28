import numpy as np
import yaml
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os

# Add the parent directory to the path so we can import main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main import main, setup_model, get_initial_guess, solve

def run_x_range_experiment():
    """
    Run experiments across different x_min and x_max values and plot the loss landscape.
    """
    
    # Load the base configuration
    with open('../config.yaml', 'r') as file:
        base_config = yaml.safe_load(file)
    
    # Define the range of x values to test
    x_range = np.linspace(0.0, 0.2, 20)  # 0% to 20% strain
    
    # Initialize square arrays to store results
    n = len(x_range)
    loss_values = np.zeros((n, n))
    success_rates = np.zeros((n, n))
    
    # Get model setup once
    mode = base_config['mode']
    param_names, params, funct_tuple, funct_class, ground_truth, constraints_list = setup_model(mode, base_config)
    
    # Update number of data points based on interval size
    base_points = base_config['data']['n_points']  # Original number for full range
    points_per_unit = base_points / 0.2  # Points per unit strain (0.2 is full range)
    
    print(f"Running experiments for {n} x values...")
    print(f"Total valid experiments: {(n * (n-1))//2}")  # Only cases where min < max
    
    # Iterate over square domain
    for i, x_min in enumerate(x_range):
        for j, x_max in enumerate(x_range):
            print(f"Progress: {i*len(x_range) + j + 1}/{n * n} - x_min: {x_min:.3f}, x_max: {x_max:.3f}")
            
            # Skip invalid combinations where x_min >= x_max
            if x_min >= x_max:
                loss_values[i, j] = np.nan
                success_rates[i, j] = 0
                continue
            
            # Create a copy of the config with new x_min and x_max
            config = base_config.copy()
            config['data']['x_min'] = float(x_min)
            config['data']['x_max'] = float(x_max)
            
            try:
                # Generate data points
                x_data = np.linspace(config['data']['x_min'], config['data']['x_max'], config['data']['n_points'])
                y_data = np.array([float(ground_truth(x)) for x in x_data])
                x_noise = np.random.normal(0, config['data']['x_noise'], len(x_data))
                y_noise = np.random.normal(0, config['data']['y_noise'], len(y_data))
                x_data = x_data + x_noise
                y_data = y_data + y_noise
                
                # Solve the optimization problem
                result = main(config)
                
                # Store results
                loss_values[i, j] = float(result['parameter_norms'])
                success_rates[i, j] = 1 if result['success'] else 0
                
            except Exception as e:
                print(f"Error for x_min={x_min:.3f}, x_max={x_max:.3f}: {e}")
                loss_values[i, j] = np.nan
                success_rates[i, j] = 0
    
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
    # Run the experiment
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