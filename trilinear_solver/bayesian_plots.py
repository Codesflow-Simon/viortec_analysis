import numpy as np
import matplotlib.pyplot as plt

def plot_bayesian_distributions(distributions, param_names, n_samples=1000, n_points=100):
    """
    Plot prior, data likelihood, and posterior distributions.
    
    Args:
        distributions: Dictionary with 'prior', 'data', 'posterior' keys
                      Each containing 'mean' and 'cov' arrays
        param_names: List of parameter names
        n_samples: Number of samples to generate for scatter plots
        n_points: Number of points for cross-section plots
    """
    
    # Create figures directory if it doesn't exist
    import os
    os.makedirs('./figures', exist_ok=True)
    
    # Plot marginal distributions for each parameter
    plot_marginal_distributions(distributions, param_names)
    
    print("Bayesian distribution plots saved in ./figures/")

def plot_marginal_distributions(distributions, param_names):
    """Plot marginal distributions for each parameter."""
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'prior': 'blue', 'data': 'red', 'posterior': 'green'}
    labels = {'prior': 'Prior', 'data': 'Data Likelihood', 'posterior': 'Posterior'}
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        
        # Generate range for this parameter
        means = [dist['mean'][i] for dist in distributions.values()]
        stds = [np.sqrt(dist['cov'][i, i]) for dist in distributions.values()]
        
        x_min = min(means) - 3 * max(stds)
        x_max = max(means) + 3 * max(stds)
        x_range = np.linspace(x_min, x_max, 200)
        
        for dist_name, dist_data in distributions.items():
            mean = dist_data['mean'][i]
            std = np.sqrt(dist_data['cov'][i, i])
            
            # Plot Gaussian PDF
            pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std)**2)
            ax.plot(x_range, pdf, color=colors[dist_name], label=labels[dist_name], linewidth=2)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Probability Density')
        ax.set_title(f'Marginal Distribution: {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplot if needed
    if n_params < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('./figures/marginal_distributions.png', dpi=300, bbox_inches='tight')
    plt.close() 