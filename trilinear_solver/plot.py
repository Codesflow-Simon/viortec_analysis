import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from function import TrilinearFunction
from loss import loss
from constraints import constraints
import os

def plot_loss_cross_sections(x_data, y_data, function: TrilinearFunction, n_points=50):
    """
    Plot cross-sections of the loss function for different pairs of variables.
    
    Args:
        x_data: Input data points
        y_data: Target data points  
        function: TrilinearFunction object with optimal parameters
        n_points: Number of points for each variable range
    """
    k_1_opt, k_2_opt, k_3_opt, x_0_opt, x_1_opt, x_2_opt = function.get_params()
    
    # Define ranges for each parameter (centered around optimal values)
    ranges = {
        'k_1': (k_1_opt * 0.5, k_1_opt * 1.5),
        'k_2': (k_2_opt * 0.5, k_2_opt * 1.5), 
        'k_3': (k_3_opt * 0.5, k_3_opt * 1.5),
        'x_0': (x_0_opt * 0.5, x_0_opt * 1.5),
        'x_1': (x_1_opt * 0.5, x_1_opt * 1.5),
        'x_2': (x_2_opt * 0.5, x_2_opt * 1.5)
    }
    
    # Variable pairs to plot
    variable_pairs = [
        ('k_1', 'k_2'),
        ('k_2', 'k_3'), 
        ('x_1', 'x_2'),
        ('k_1', 'x_1'),
        ('k_2', 'x_2')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
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
                # Create parameter vector with current values
                params = list(function.get_params())
                
                # Map variable names to indices
                var_indices = {'k_1': 0, 'k_2': 1, 'k_3': 2, 'x_0': 3, 'x_1': 4, 'x_2': 5}
                params[var_indices[var1]] = X[i, j]
                params[var_indices[var2]] = Y[i, j]
                
                # Calculate loss
                try:
                    Z[i, j] = loss(params, x_data, y_data)
                except:
                    Z[i, j] = np.nan
        
        # Create contour plot
        contour = ax.contour(X, Y, Z, levels=20, colors='black', alpha=0.6)
        ax.clabel(contour, inline=True, fontsize=8)
        
        # Add filled contour
        filled_contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.7)
        
        # Mark optimal point
        opt_var1 = function.get_params()[var_indices[var1]]
        opt_var2 = function.get_params()[var_indices[var2]]
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


def plot_data_and_fit(x_data, y_data, function: TrilinearFunction, ground_truth_function=None):
    """
    Plot the original data points and the fitted trilinear function.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        function: TrilinearFunction object with optimal parameters
        ground_truth_function: Optional ground truth TrilinearFunction for comparison
    """
    k_1_opt, k_2_opt, k_3_opt, x_0_opt, x_1_opt, x_2_opt = function.get_params()
    
    # Generate smooth curve for plotting
    x_smooth = np.linspace(-1, 4, 1000)
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
    
    # Mark key points
    plt.axvline(x=x_0_opt, color='g', linestyle=':', alpha=0.7, label=f'x₀ = {x_0_opt:.3f}')
    plt.axvline(x=x_1_opt, color='orange', linestyle=':', alpha=0.7, label=f'x₁ = {x_1_opt:.3f}')
    plt.axvline(x=x_2_opt, color='purple', linestyle=':', alpha=0.7, label=f'x₂ = {x_2_opt:.3f}')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Fitted Trilinear Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/data_and_fit.png', dpi=300, bbox_inches='tight')

def generate_plots(x_data, y_data, function: TrilinearFunction, ground_truth_function=None):
    """
    Generate all plots for the trilinear fitting results.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        function: TrilinearFunction object with optimal parameters
        ground_truth_function: Optional ground truth TrilinearFunction for comparison
    
    Returns:
        dict: Dictionary containing the generated plot objects
    """
    # Create figures directory if it doesn't exist
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    
    print("Generating plots...")
    
    # Generate all plots
    plot_data_and_fit(x_data, y_data, function, ground_truth_function)
    plot_loss_cross_sections(x_data, y_data, function)
    # plot_3d_loss_surface(x_data, y_data, function, 'x_1', 'k_1')
    
    print("All plots saved as PNG files in ./figures/")
    
    return {
        'data_and_fit': './figures/data_and_fit.png',
        'loss_cross_sections': './figures/loss_cross_sections.png',
        'loss_3d_surface': './figures/loss_3d_surface_k_1_k_2.png'
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
    

def generate_plots_with_uncertainty(x_data, y_data, function: TrilinearFunction, ground_truth_function=None, std=None):
    """
    Generate all plots for the trilinear fitting results.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        function: TrilinearFunction object with optimal parameters
        ground_truth_function: Optional ground truth TrilinearFunction for comparison
    
    Returns:
        dict: Dictionary containing the generated plot objects
    """
    # Create figures directory if it doesn't exist
    if not os.path.exists('./figures'):
        os.makedirs('./figures')
    
    print("Generating plots...")
    
    # Generate all plots
    plot_data_and_fit_with_bars(x_data, y_data, function, ground_truth_function, std)
    # plot_3d_loss_surface(x_data, y_data, function, 'x_1', 'k_1')
    
    print("All plots saved as PNG files in ./figures/")
    
    return {
        'data_and_fit': './figures/data_and_fit.png',
        'loss_cross_sections': './figures/loss_cross_sections.png',
        'loss_3d_surface': './figures/loss_3d_surface_k_1_k_2.png'
    }

def plot_data_and_fit_with_bars(x_data, y_data, function: TrilinearFunction, ground_truth_function=None, std=None):
    """
    Plot the original data points and the fitted trilinear function.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        function: TrilinearFunction object with optimal parameters
        ground_truth_function: Optional ground truth TrilinearFunction for comparison
    """
    k_1_opt, k_2_opt, k_3_opt, x_0_opt, x_1_opt, x_2_opt = function.get_params()
    
    # Generate smooth curve for plotting
    x_smooth = np.linspace(-1, 4, 1000)
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
    
    # Mark key points with error bars
    plt.axvline(x=x_0_opt, color='g', linestyle=':', alpha=0.7, label=f'x₀ = {x_0_opt:.3f} ± {std[3]:.3f}')
    plt.axvline(x=x_1_opt, color='orange', linestyle=':', alpha=0.7, label=f'x₁ = {x_1_opt:.3f} ± {std[4]:.3f}')
    plt.axvline(x=x_2_opt, color='purple', linestyle=':', alpha=0.7, label=f'x₂ = {x_2_opt:.3f} ± {std[5]:.3f}')
    
    # Add horizontal error bars at inflection points
    y_0 = float(function(x_0_opt))
    y_1 = float(function(x_1_opt)) 
    y_2 = float(function(x_2_opt))
    
    plt.errorbar([x_0_opt], [y_0], xerr=[std[3]], color='g', capsize=5, fmt='none')
    plt.errorbar([x_1_opt], [y_1], xerr=[std[4]], color='orange', capsize=5, fmt='none')
    plt.errorbar([x_2_opt], [y_2], xerr=[std[5]], color='purple', capsize=5, fmt='none')
    plt.xlim(-1, 4)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Fitted Trilinear Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('./figures/data_and_fit_with_bars.png', dpi=300, bbox_inches='tight')
