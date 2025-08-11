import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ligament_models import TrilinearFunction
from .modelling.loss import loss

def plot_loss_cross_sections(x_data, y_data, function: TrilinearFunction, n_points=50):
    """
    Plot cross-sections of the loss function for different pairs of variables.
    
    Args:
        x_data: Input data points
        y_data: Target data points  
        function: TrilinearFunction object with optimal parameters
        n_points: Number of points for each variable range
    """

    params = function.get_params()
    
    # Define ranges for each parameter (centered around optimal values)
    ranges = {}
    for key, value in params.items():
        ranges[key] = (value * 0.5, value * 1.5)
    
    # Variable pairs to plot
    if ('k_1' in params) and ('k_2' in params) and \
        ('k_3' in params) and ('x_1' in params) and \
        ('x_2' in params):
        variable_pairs = [
            ('k_1', 'k_2'),
            ('k_2', 'k_3'), 
            ('x_1', 'x_2'),
            ('k_1', 'x_1'),
            ('k_2', 'x_2')
        ]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))


    elif ('e_t' in params) and ('k_1' in params):
        variable_pairs = [
            ('e_t', 'k_1'),
        ]

        fig, axes = plt.subplots(1, 1, figsize=(18, 12))

    else:
        raise ValueError(f"Unsupported parameter set: {params}")
    
    axes = axes.flatten()
    
    for idx, (var1, var2) in enumerate(variable_pairs):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Create meshgrid for the two variables
        var1_range = np.linspace(ranges[var1][0], ranges[var1][1], n_points)
        var2_range = np.linspace(ranges[var2][0], ranges[var2][1], n_points)
        X, Y = np.meshgrid(var1_range, var2_range)
        
        # Initialize loss values
        Z = np.zeros_like(X)
        
        # Calculate loss for each point
        for i in range(n_points):
            for j in range(n_points):
                params = function.get_params()
                params[var1] = X[i, j]
                params[var2] = Y[i, j]
                
                # Calculate loss
                try:
                    Z[i, j] = loss(list(params.values()), x_data, y_data)
                except:
                    Z[i, j] = np.nan
        
        # Create contour plot
        contour = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Add filled contour
        filled_contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        
        # Mark optimal point
        opt_var1 = function.get_params()[var1]
        opt_var2 = function.get_params()[var2]
        ax.plot(opt_var1, opt_var2, 'r*', markersize=15, label='Optimal')
        
        # Add colorbar
        plt.colorbar(filled_contour, ax=ax, label='Loss')
        
        ax.set_xlabel(var1)
        ax.set_ylabel(var2)
        ax.set_title(f'Loss: {var1} vs {var2}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot if needed
    if len(variable_pairs) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('./figures/loss_cross_sections.png', dpi=300, bbox_inches='tight')


def plot_data_and_fit(x_data, y_data, function: TrilinearFunction, ground_truth_function=None, std_devs=None):
    """
    Plot the original data points and the fitted trilinear function.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        function: TrilinearFunction object with optimal parameters
        ground_truth_function: Optional ground truth TrilinearFunction for comparison
        std_devs: Optional array of standard deviations for parameters [k_1, k_2, k_3, x_1, x_2]
    """
    params = function.get_params()
    
    # Generate smooth curve for plotting
    x_smooth = np.linspace(0.18, 0.30, 1000)
    y_smooth = np.array([float(function(x)) for x in x_smooth])
    
    plt.figure(figsize=(12, 8))
    
    # Plot data points
    plt.scatter(x_data, y_data, alpha=0.6, label='Data points', s=30)
    
    # Plot fitted curve
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted trilinear function')
    
    # Plot ground truth if provided
    if ground_truth_function is not None:
        y_ground_truth = np.array([float(ground_truth_function(x)) for x in x_smooth])
        plt.plot(x_smooth, y_ground_truth, 'g--', linewidth=2, label='Ground truth')
    
    # Mark key points with error bars if standard deviations are provided
    if 'x_1' in params:
        x1_value = params['x_1']
        plt.axvline(x=x1_value, color='orange', linestyle=':', alpha=0.7, label=f"x₁ = {x1_value:.3f}")
        
        # Add horizontal error bar for x_1 if std_devs provided
        if std_devs is not None and len(std_devs) >= 4:
            x1_std = std_devs[3]  # x_1 is at index 3
            y_range = plt.ylim()
            plt.errorbar(x1_value, (y_range[0] + y_range[1]) / 2, 
                        xerr=x1_std, fmt='none', color='orange', 
                        capsize=5, capthick=2, linewidth=2, alpha=0.8)
    
    if 'x_2' in params:
        x2_value = params['x_2']
        plt.axvline(x=x2_value, color='purple', linestyle=':', alpha=0.7, label=f"x₂ = {x2_value:.3f}")
        
        # Add horizontal error bar for x_2 if std_devs provided
        if std_devs is not None and len(std_devs) >= 5:
            x2_std = std_devs[4]  # x_2 is at index 4
            y_range = plt.ylim()
            plt.errorbar(x2_value, (y_range[0] + y_range[1]) / 2, 
                        xerr=x2_std, fmt='none', color='purple', 
                        capsize=5, capthick=2, linewidth=2, alpha=0.8)
    
    if 'e_t' in params:
        plt.axvline(x=params['e_t'], color='green', linestyle=':', alpha=0.7, label=f"e_t = {params['e_t']:.3f}")
    
    # Mark ground truth inflection points if provided
    if ground_truth_function is not None:
        ground_truth_params = ground_truth_function.get_params()
        if 'x_1' in ground_truth_params:
            gt_x1 = ground_truth_params['x_1']
            plt.axvline(x=gt_x1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label=f"Ground truth x₁ = {gt_x1:.3f}")
        if 'x_2' in ground_truth_params:
            gt_x2 = ground_truth_params['x_2']
            plt.axvline(x=gt_x2, color='purple', linestyle='--', alpha=0.5, linewidth=1.5, label=f"Ground truth x₂ = {gt_x2:.3f}")
        if 'e_t' in ground_truth_params:
            gt_et = ground_truth_params['e_t']
            plt.axvline(x=gt_et, color='green', linestyle='--', alpha=0.5, linewidth=1.5, label=f"Ground truth e_t = {gt_et:.3f}")
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Fitted Trilinear Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/data_and_fit.png', dpi=300, bbox_inches='tight')

def plot_data_and_fit_blankevoort(x_data, y_data, function, ground_truth_function=None):
    # Use get_params if available, else fallback to attributes
    if hasattr(function, 'get_params'):
        params = function.get_params()
        transition_length, k_1 = params['e_t'], params['k']
    else:
        transition_length, k_1 = function.transition_length, function.k_1
    x_smooth = np.linspace(-0.05, 0.1, 1000)
    y_smooth = np.array([float(function(x)) for x in x_smooth])
    plt.figure(figsize=(12, 8))
    plt.scatter(x_data, y_data, alpha=0.6, label='Data points', s=30)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label='Fitted Blankevoort function')
    if ground_truth_function is not None:
        y_ground_truth = np.array([float(ground_truth_function(x)) for x in x_smooth])
        plt.plot(x_smooth, y_ground_truth, 'g--', linewidth=2, label='Ground truth')
    # Mark transition_length
    plt.axvline(x=transition_length, color='orange', linestyle=':', alpha=0.7, label=f'L = {transition_length:.3f}')
    
    # Mark ground truth transition point if provided
    if ground_truth_function is not None:
        if hasattr(ground_truth_function, 'get_params'):
            gt_params = ground_truth_function.get_params()
            gt_transition_length = gt_params['e_t']
        else:
            gt_transition_length = ground_truth_function.transition_length
        plt.axvline(x=gt_transition_length, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label=f'Ground truth L = {gt_transition_length:.3f}')
    
    plt.xlabel('strain')
    plt.ylabel('force')
    plt.title('Data and Fitted Blankevoort Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/data_and_fit_blankevoort.png', dpi=300, bbox_inches='tight')


def generate_plots(x_data, y_data, function, ground_truth_function=None, std_devs=None):
    # Use get_params to determine function type
    n_params = len(function.get_params()) if hasattr(function, 'get_params') else 0
    if n_params == 2:
        plot_data_and_fit_blankevoort(x_data, y_data, function, ground_truth_function)
        plot_path = './figures/data_and_fit_blankevoort.png'
    else:
        plot_data_and_fit(x_data, y_data, function, ground_truth_function, std_devs)
        plot_path = './figures/data_and_fit.png'
    print("All plots saved as PNG files in ./figures/")
    return {
        'data_and_fit': plot_path,
    }

def plot_hessian(hessian, path='./figures/hessian_heatmap.png'):
    """Basic heatmap visualization of the Hessian matrix."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap with centered colormap
    vmax = max(abs(np.min(hessian)), abs(np.max(hessian)))
    im = plt.imshow(hessian, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Hessian Value')
    
    # Add parameter labels
    param_labels = ['k₁', 'k₂', 'k₃', 'x₀', 'x₁', 'x₂']
    plt.xticks(range(6), param_labels)
    plt.yticks(range(6), param_labels)
    
    # Add text annotations
    for i in range(6):
        for j in range(6):
            plt.text(j, i, f'{hessian[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    # Add title and adjust layout
    plt.title('Hessian Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot inverse Hessian
    plt.figure(figsize=(10, 8))
    inv_hessian = np.linalg.inv(hessian)
    
    # Use YlOrRd colormap for positive values
    im = plt.imshow(inv_hessian, cmap='YlOrRd', aspect='equal')
    plt.colorbar(im, label='Inverse Hessian Value')
    
    # Add parameter labels
    param_labels = ['k₁', 'k₂', 'k₃', 'x₀', 'x₁', 'x₂']
    plt.xticks(range(6), param_labels)
    plt.yticks(range(6), param_labels)
    
    # Add text annotations
    for i in range(6):
        for j in range(6):
            plt.text(j, i, f'{inv_hessian[i, j]:.2f}',
                    ha='center', va='center', fontsize=8)
    
    plt.title('Inverse Hessian Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(path.replace('hessian', 'inverse_hessian'), dpi=300, bbox_inches='tight')
    

def plot_hessian(hessian, path='./figures/hessian_heatmap.png'):
    """Basic heatmap visualization of the Hessian matrix."""
    plt.figure(figsize=(6, 5))
    vmax = max(abs(np.min(hessian)), abs(np.max(hessian)))
    im = plt.imshow(hessian, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label='Hessian Value')

    # Use correct parameter labels
    if hessian.shape == (2, 2):
        param_labels = ['transition_length', 'k_1']
    elif hessian.shape == (6, 6):
        param_labels = ['k₁', 'k₂', 'k₃', 'x₀', 'x₁', 'x₂']
    else:
        param_labels = [f'θ{i+1}' for i in range(hessian.shape[0])]

    plt.xticks(range(hessian.shape[0]), param_labels)
    plt.yticks(range(hessian.shape[1]), param_labels)

    # Add text annotations
    for i in range(hessian.shape[0]):
        for j in range(hessian.shape[1]):
            plt.text(j, i, f'{hessian[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)

    plt.title('Hessian Matrix Heatmap')
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def make_plots(info_dict):
    result = info_dict
    x_data = result['x_data']
    y_data = result['y_data']
    fitted_function = result['fitted_function']
    ground_truth = result['ground_truth']
    distributions = result['distributions']
    param_names = result['param_names']

    posterior_mean = distributions['posterior']['mean']
    posterior_cov_matrix = distributions['posterior']['cov']

    data_mean = distributions['data']['mean']
    data_cov_matrix = distributions['data']['cov']
    data_std = np.sqrt(np.diag(data_cov_matrix))

    prior_mean = distributions['prior']['mean']
    prior_cov_matrix = distributions['prior']['cov']
    prior_std = np.sqrt(np.diag(prior_cov_matrix))

    # Generate plots
    print("\n=== Generating Plots ===")

    print("\nPosterior Distribution:")
    print("-----------------------")
    print(f"Mean:  {posterior_mean}")
    print(f"Std:   {np.sqrt(np.diag(posterior_cov_matrix))}")

    print("\nLikelihood/Data:")
    print("---------------")
    print(f"Covariance Shape: {data_cov_matrix.shape}")
    print(f"MAP Estimate:     {data_mean}")

    print("\nPrior Distribution:")
    print("------------------") 
    print(f"Mean:  {prior_mean}")
    print(f"Std:   {prior_std}")
    
    std = np.sqrt(np.diag(posterior_cov_matrix))
    generate_plots(x_data, y_data, fitted_function, ground_truth, std)
    
    plot_hessian(data_cov_matrix, path='./figures/sampling_covariance_heatmap.png')
        
    from bayesian_plots import plot_bayesian_distributions
    plot_bayesian_distributions(distributions, param_names)