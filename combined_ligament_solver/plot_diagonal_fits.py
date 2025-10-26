"""
Script to plot individual fits for diagonal cases (MCL strain = LCL strain)
Similar to the plot in main.py line 215
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.gridspec import GridSpec

def load_results(pickle_path):
    """Load results from pickle file"""
    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded {len(results)} results from {pickle_path}")
    return results

def filter_diagonal_cases(results, ligament_type='MCL'):
    """Filter for cases where MCL strain = LCL strain and extract specified ligament"""
    diagonal_results = []
    for result in results:
        if result['lcl_strain'] == result['mcl_strain']:
            # Create a new result with the specified ligament type
            diagonal_result = {
                'lcl_strain': result['lcl_strain'],
                'mcl_strain': result['mcl_strain'],
                'ligament_type': ligament_type,
                'result': result['result'][ligament_type]  # Extract specific ligament data
            }
            diagonal_results.append(diagonal_result)
    
    print(f"Filtered to {len(diagonal_results)} diagonal cases for {ligament_type}")
    return diagonal_results

def plot_single_fit(result_data, strain, ligament_type='MCL', output_path=None):
    """Plot fit for a single case, similar to main.py line 215"""
    length = result_data['data']['length']
    relative_force = result_data['data']['relative_force']
    samples = result_data['samples']
    gt_params = result_data['gt_params']
    optimized_params = result_data['optimized_params']
    
    # We need to reconstruct the function - import it
    from src.ligament_models.blankevoort import BlankevoortFunction
    
    # Create function with optimized parameters
    function = BlankevoortFunction({'k': 1, 'alpha': 1, 'l_0': 1, 'f_ref': 1})  # Dummy config
    # Convert optimized_params dict to array if needed
    if isinstance(optimized_params, dict):
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        optimized_params_array = np.array([optimized_params[p] for p in param_names])
    else:
        optimized_params_array = optimized_params
    function.set_params(optimized_params_array)
    
    # Create figure
    fig = plt.figure(figsize=(12, 5))
    
    # Main plot
    ax1 = plt.subplot(1, 2, 1)
    
    # Scatter plot of data
    ax1.scatter(length, relative_force, c='r', label='Data', s=20, alpha=0.6, zorder=3)
    
    # Create x range for plotting
    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), 
                        max(gt_params['l_0']*1.1, np.max(length)), 200)
    
    # Plot MCMC samples in background (grey lines)
    if samples is not None: 
        if isinstance(samples, list):
            samples = np.array(samples)
        
        samples_to_plot = min(100, len(samples))
        plot_indices = np.random.choice(len(samples), samples_to_plot, replace=False)
        
        for idx in plot_indices:
            sample = samples[idx]
            y_sample = function.vectorized_function(x_data, sample).flatten()
            ax1.plot(x_data, y_sample, c='grey', alpha=0.15, linewidth=1.0, zorder=1)
    
    # Plot optimized model (blue line)
    function.set_params(optimized_params_array)
    ax1.plot(x_data, function(x_data), c='b', label='Fitted Model', linewidth=2, zorder=2)
    
    # Plot ground truth (green dashed line)
    gt_params_array = np.array([gt_params['k'], gt_params['alpha'], 
                                gt_params['l_0'], gt_params['f_ref']])
    function.set_params(gt_params_array)
    ax1.plot(x_data, function(x_data), c='g', label='Ground Truth', 
            linestyle='--', linewidth=2, zorder=2)
    
    ax1.legend(loc='best', fontsize=10)
    ax1.set_xlabel('Ligament Length (mm)', fontsize=11)
    ax1.set_ylabel('Relative Force (N)', fontsize=11)
    ax1.set_title(f'{ligament_type} Force-Length Curve (Strain = {strain:.2f})', 
                 fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Parameter comparison plot
    ax2 = plt.subplot(1, 2, 2)
    
    if samples is not None:
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        param_labels = ['k', 'α', 'l₀', 'f_ref']
        
        means = [np.mean(samples[:, i]) for i in range(4)]
        stds = [np.std(samples[:, i]) for i in range(4)]
        gts = [gt_params[p] for p in param_names]
        opts = [optimized_params[p] if isinstance(optimized_params, dict) else optimized_params[i] for i, p in enumerate(param_names)]
        
        x_pos = np.arange(len(param_names))
        width = 0.3
        
        # Plot ground truth
        ax2.scatter(x_pos - width, gts, c='green', s=150, marker='D', 
                   label='Ground Truth', zorder=3, edgecolors='black', linewidths=1)
        
        # Plot optimized values
        ax2.scatter(x_pos, opts, c='blue', s=150, marker='o', 
                   label='Optimized', zorder=3, edgecolors='black', linewidths=1)
        
        # Plot MCMC mean with error bars
        ax2.errorbar(x_pos + width, means, yerr=stds, fmt='s', capsize=5, 
                    label='MCMC Mean ± Std', markersize=8, color='purple', 
                    zorder=2, alpha=0.7, linewidth=2)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(param_labels, fontsize=11)
        ax2.set_ylabel('Parameter Value', fontsize=11)
        ax2.set_title('Parameter Comparison', fontsize=12, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot for strain={strain:.2f}")
    
    return fig

def create_combined_plot(diagonal_results, output_path=None):
    """Create a single figure with all diagonal cases"""
    n_cases = len(diagonal_results)
    fig = plt.figure(figsize=(16, 5*n_cases))
    
    for idx, result in enumerate(diagonal_results):
        strain = result['mcl_strain']
        result_data = result['result']
        
        length = result_data['data']['length']
        relative_force = result_data['data']['relative_force']
        samples = result_data['samples']
        gt_params = result_data['gt_params']
        optimized_params = result_data['optimized_params']
        
        # Import function
        from src.ligament_models.blankevoort import BlankevoortFunction
        function = BlankevoortFunction({'k': 1, 'alpha': 1, 'l_0': 1, 'f_ref': 1})
        
        # Convert optimized_params dict to array if needed
        if isinstance(optimized_params, dict):
            param_names = ['k', 'alpha', 'l_0', 'f_ref']
            optimized_params_array = np.array([optimized_params[p] for p in param_names])
        else:
            optimized_params_array = optimized_params
        
        # Create subplot
        ax = plt.subplot(n_cases, 1, idx + 1)
        
        # Scatter plot of data
        ax.scatter(length, relative_force, c='r', label='Data', s=30, alpha=0.6, zorder=3)
        
        # Create x range for plotting
        x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), 
                            max(gt_params['l_0']*1.1, np.max(length)), 200)
        
        # Plot MCMC samples in background
        if samples is not None:
            if isinstance(samples, list):
                samples = np.array(samples)
            
            samples_to_plot = min(100, len(samples))
            plot_indices = np.random.choice(len(samples), samples_to_plot, replace=False)
            
            for sample_idx in plot_indices:
                sample = samples[sample_idx]
                y_sample = function.vectorized_function(x_data, sample).flatten()
                ax.plot(x_data, y_sample, c='grey', alpha=0.15, linewidth=1.0, zorder=1)
        
        # Plot optimized model
        function.set_params(optimized_params_array)
        ax.plot(x_data, function(x_data), c='b', label='Fitted Model', linewidth=3, zorder=2)
        
        # Plot ground truth
        gt_params_array = np.array([gt_params['k'], gt_params['alpha'], 
                                    gt_params['l_0'], gt_params['f_ref']])
        function.set_params(gt_params_array)
        ax.plot(x_data, function(x_data), c='g', label='Ground Truth', 
               linestyle='--', linewidth=3, zorder=2)
        
        ax.legend(loc='best', fontsize=12)
        ax.set_xlabel('Ligament Length (mm)', fontsize=13)
        ax.set_ylabel('Relative Force (N)', fontsize=13)
        ax.set_title(f'Strain = {strain:.2f}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot")
    
    return fig

def main(pickle_path, ligament_type='MCL'):
    """Main function"""
    
    # Load results
    results = load_results(pickle_path)
    
    # Filter for diagonal cases
    diagonal_results = filter_diagonal_cases(results, ligament_type)
    
    if len(diagonal_results) == 0:
        print("ERROR: No diagonal cases found!")
        return
    
    # Create output directory
    pickle_dir = os.path.dirname(pickle_path)
    pickle_name = os.path.splitext(os.path.basename(pickle_path))[0]
    output_dir = os.path.join(pickle_dir, f'{pickle_name}_{ligament_type.lower()}_diagonal_fits')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nGenerating fit plots in {output_dir}...")
    
    # Create individual plots for each case
    for result in diagonal_results:
        strain = result['mcl_strain']
        output_path = os.path.join(output_dir, f'fit_strain_{strain:.2f}.png')
        plot_single_fit(result['result'], strain, ligament_type, output_path)
    
    # Create combined plot
    combined_path = os.path.join(output_dir, 'all_fits_combined.png')
    create_combined_plot(diagonal_results, combined_path)
    
    print(f"\n✓ All fit plots saved to: {output_dir}")

if __name__ == "__main__":
    # Parse arguments
    ligament_type = 'MCL'  # Default
    pickle_path = None
    
    if len(sys.argv) > 1:
        pickle_path = sys.argv[1]
    if len(sys.argv) > 2:
        ligament_type = sys.argv[2].upper()
        if ligament_type not in ['MCL', 'LCL']:
            print(f"ERROR: Invalid ligament type '{sys.argv[2]}'. Must be 'MCL' or 'LCL'.")
            sys.exit(1)
    
    if pickle_path is None:
        # Find most recent pickle file
        results_dir = 'results'
        if os.path.exists(results_dir):
            pickle_files = [f for f in os.listdir(results_dir) if f.startswith('results_') and f.endswith('.pkl')]
            if pickle_files:
                pickle_path = os.path.join(results_dir, sorted(pickle_files)[-1])
                print(f"Using most recent pickle: {pickle_path}")
            else:
                print("No pickle files found in results/")
                print("Usage: python plot_diagonal_fits.py [path_to_results.pkl] [ligament_type]")
                print("       ligament_type: MCL or LCL (default: MCL)")
                sys.exit(1)
        else:
            print("No results directory found.")
            print("Usage: python plot_diagonal_fits.py [path_to_results.pkl] [ligament_type]")
            print("       ligament_type: MCL or LCL (default: MCL)")
            sys.exit(1)
    
    main(pickle_path, ligament_type)

