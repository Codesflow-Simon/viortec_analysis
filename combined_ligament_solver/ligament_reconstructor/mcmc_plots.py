import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde
import pandas as pd
import os
from copy import deepcopy

def create_figures_directory():
    """Create figures directory if it doesn't exist."""
    os.makedirs('./figures', exist_ok=True)

def plot_mcmc_diagnostics(samples, param_names, acceptance_rate=None, save_plots=True):
    """
    Comprehensive MCMC diagnostic plots.
    
    Args:
        samples: MCMC samples array (n_samples, n_params)
        param_names: List of parameter names
        acceptance_rate: MCMC acceptance rate
        save_plots: Whether to save plots to files
    """
    create_figures_directory()
    
    # Convert to pandas DataFrame for easier handling
    df = pd.DataFrame(samples, columns=param_names)
    
    # 1. Trace plots
    # plot_trace_plots(df, param_names, save_plots)
    
    # 2. Marginal distributions with non-Gaussian analysis
    plot_marginal_distributions(df, param_names, save_plots)
    
    # 3. Correlation matrix
    # plot_correlation_matrix(df, param_names, save_plots)
    
    # 4. Pair plots
    plot_pair_plots(df, param_names, save_plots)
    
    # 5. Summary statistics
    # plot_summary_statistics(df, param_names, acceptance_rate, save_plots)
    
    if save_plots:
        print("MCMC diagnostic plots saved in ./figures/")

def plot_trace_plots(df, param_names, save_plots=True):
    """Plot MCMC trace plots for each parameter."""
    n_params = len(param_names)
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 2*n_params))
    
    if n_params == 1:
        axes = [axes]
    
    for i, param_name in enumerate(param_names):
        ax = axes[i]
        ax.plot(df[param_name], alpha=0.7, linewidth=0.5)
        ax.set_ylabel(param_name)
        ax.grid(True, alpha=0.3)
        
        if i == n_params - 1:
            ax.set_xlabel('MCMC Step')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_trace_plots.png', dpi=300, bbox_inches='tight')

def plot_marginal_distributions(df, param_names, save_plots=True):
    """Plot marginal distributions with non-Gaussian analysis."""
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = df[param_name].values
        
        # Histogram with KDE
        ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # KDE estimate
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min(), data.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        # Gaussian fit for comparison
        mu, sigma = np.mean(data), np.std(data)
        gaussian_pdf = stats.norm.pdf(x_range, mu, sigma)
        ax.plot(x_range, gaussian_pdf, 'g--', linewidth=2, label='Gaussian fit')
        
        # Add statistics
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{param_name}\nSkew: {skewness:.3f}, Kurt: {kurtosis:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for i in range(n_params, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_marginal_distributions.png', dpi=300, bbox_inches='tight')

def plot_correlation_matrix(df, param_names, save_plots=True):
    """Plot correlation matrix between parameters."""
    corr_matrix = df.corr()
    
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('./figures/mcmc_correlation_matrix.png', dpi=300, bbox_inches='tight')

def plot_pair_plots(df, param_names, save_plots=True):
    """Plot pairwise scatter plots."""
    # Limit to 6 parameters for readability
    if len(param_names) > 6:
        print(f"Warning: Limiting pair plots to first 6 parameters (out of {len(param_names)})")
        df_subset = df[param_names[:6]]
        param_names_subset = param_names[:6]
    else:
        df_subset = df
        param_names_subset = param_names
    
    sns.pairplot(df_subset, diag_kind='kde', plot_kws={'alpha': 0.6, 's': 10})
    plt.suptitle('Parameter Pairwise Relationships', y=1.02)
    
    if save_plots:
        plt.savefig('./figures/mcmc_pair_plots.png', dpi=300, bbox_inches='tight')

def plot_summary_statistics(df, param_names, acceptance_rate=None, save_plots=True):
    """Plot summary statistics table."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate statistics
    stats_data = []
    for param_name in param_names:
        data = df[param_name].values
        stats_data.append([
            param_name,
            f"{np.mean(data):.4f}",
            f"{np.std(data):.4f}",
            f"{np.percentile(data, 2.5):.4f}",
            f"{np.percentile(data, 97.5):.4f}",
            f"{stats.skew(data):.3f}",
            f"{stats.kurtosis(data):.3f}"
        ])
    
    # Create table
    table = ax.table(cellText=stats_data,
                    colLabels=['Parameter', 'Mean', 'Std', '2.5%', '97.5%', 'Skewness', 'Kurtosis'],
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Add acceptance rate if provided
    if acceptance_rate is not None:
        plt.figtext(0.5, 0.02, f'MCMC Acceptance Rate: {acceptance_rate:.3f}', 
                   ha='center', fontsize=12, fontweight='bold')
    
    plt.title('MCMC Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    if save_plots:
        plt.savefig('./figures/mcmc_summary_statistics.png', dpi=300, bbox_inches='tight')

def plot_uniform_analysis(samples, param_names, save_plots=True):
    """
    Analyze and plot potential uniform distributions in parameters.
    
    Args:
        samples: MCMC samples array
        param_names: List of parameter names
        save_plots: Whether to save plots
    """
    create_figures_directory()
    
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, param_name in enumerate(param_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = samples[:, i]
        
        # Test for uniformity using Kolmogorov-Smirnov test
        data_normalized = (data - data.min()) / (data.max() - data.min())
        ks_stat, p_value = stats.kstest(data_normalized, 'uniform')
        
        # Plot histogram vs uniform
        ax.hist(data, bins=30, density=True, alpha=0.7, color='lightblue', 
                edgecolor='black', label='MCMC samples')
        
        # Uniform distribution over the range
        x_range = np.linspace(data.min(), data.max(), 100)
        uniform_pdf = np.ones_like(x_range) / (data.max() - data.min())
        ax.plot(x_range, uniform_pdf, 'r-', linewidth=2, label='Uniform')
        
        # Gaussian for comparison
        mu, sigma = np.mean(data), np.std(data)
        gaussian_pdf = stats.norm.pdf(x_range, mu, sigma)
        ax.plot(x_range, gaussian_pdf, 'g--', linewidth=2, label='Gaussian')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{param_name}\nKS p-value: {p_value:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for i in range(n_params, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_uniform_analysis.png', dpi=300, bbox_inches='tight')

def plot_convergence_diagnostics(samples, param_names, save_plots=True):
    """
    Plot convergence diagnostics including Gelman-Rubin statistic approximation.
    
    Args:
        samples: MCMC samples array
        param_names: List of parameter names
        save_plots: Whether to save plots
    """
    create_figures_directory()
    
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # 1. Running means
    ax = axes[0]
    for i, param_name in enumerate(param_names):
        running_mean = np.cumsum(samples[:, i]) / np.arange(1, len(samples) + 1)
        ax.plot(running_mean, label=param_name, alpha=0.8)
    ax.set_xlabel('MCMC Step')
    ax.set_ylabel('Running Mean')
    ax.set_title('Parameter Convergence (Running Means)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Running standard deviations
    ax = axes[1]
    for i, param_name in enumerate(param_names):
        running_std = [np.std(samples[:j+1, i]) for j in range(len(samples))]
        ax.plot(running_std, label=param_name, alpha=0.8)
    ax.set_xlabel('MCMC Step')
    ax.set_ylabel('Running Std')
    ax.set_title('Parameter Convergence (Running Std)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Autocorrelation
    ax = axes[2]
    max_lag = min(100, len(samples) // 10)
    for i, param_name in enumerate(param_names):
        autocorr = [1.0]  # lag 0
        for lag in range(1, max_lag):
            if lag < len(samples):
                corr = np.corrcoef(samples[:-lag, i], samples[lag:, i])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
        ax.plot(range(max_lag), autocorr, label=param_name, alpha=0.8)
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Parameter Autocorrelation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Effective sample size estimation
    ax = axes[3]
    ess_estimates = []
    for i, param_name in enumerate(param_names):
        # Simple ESS estimation using autocorrelation
        autocorr = [1.0]
        for lag in range(1, min(50, len(samples) // 10)):
            if lag < len(samples):
                corr = np.corrcoef(samples[:-lag, i], samples[lag:, i])[0, 1]
                autocorr.append(corr if not np.isnan(corr) else 0)
        
        # ESS = N / (1 + 2*sum(autocorr))
        ess = len(samples) / (1 + 2 * sum(autocorr[1:]))
        ess_estimates.append(ess)
    
    bars = ax.bar(param_names, ess_estimates, alpha=0.7)
    ax.set_ylabel('Effective Sample Size')
    ax.set_title('Effective Sample Size Estimation')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, ess in zip(bars, ess_estimates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ess_estimates)*0.01,
                f'{ess:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_convergence_diagnostics.png', dpi=300, bbox_inches='tight')

def plot_credible_intervals(samples, param_names, confidence_levels=[0.68, 0.95, 0.99], save_plots=True):
    """
    Plot credible intervals for each parameter.
    
    Args:
        samples: MCMC samples array
        param_names: List of parameter names
        confidence_levels: List of confidence levels to plot
        save_plots: Whether to save plots
    """
    create_figures_directory()
    
    n_params = len(param_names)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = ['lightblue', 'blue', 'darkblue']
    
    for i, param_name in enumerate(param_names):
        if i >= len(axes):
            break
            
        ax = axes[i]
        data = samples[:, i]
        
        # Plot histogram
        ax.hist(data, bins=50, density=True, alpha=0.7, color='lightgray', edgecolor='black')
        
        # Add credible intervals
        for j, conf_level in enumerate(confidence_levels):
            alpha = (1 - conf_level) / 2
            lower = np.percentile(data, alpha * 100)
            upper = np.percentile(data, (1 - alpha) * 100)
            median = np.median(data)
            
            # Plot interval
            ax.axvline(lower, color=colors[j], linestyle='--', linewidth=2, 
                      label=f'{conf_level*100:.0f}% CI')
            ax.axvline(upper, color=colors[j], linestyle='--', linewidth=2)
            ax.axvline(median, color='red', linestyle='-', linewidth=2, label='Median')
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('Density')
        ax.set_title(f'{param_name} Credible Intervals')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove extra subplots
    for i in range(n_params, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_credible_intervals.png', dpi=300, bbox_inches='tight')

def plot_loss_landscape_with_mcmc(samples, param_names, x_data, y_data, func, 
                                 map_params=None, save_plots=True, n_points=50, epsilon=0.1):
    """
    Plot loss landscape cross-sections with MCMC samples overlaid.
    
    Args:
        samples: MCMC samples array (n_samples, n_params)
        param_names: List of parameter names
        x_data: Input data points
        y_data: Target data points
        func: Function to evaluate
        map_params: MAP estimate (optional, for reference)
        save_plots: Whether to save plots
        n_points: Number of points for landscape evaluation
        epsilon: Tolerance for selecting samples near the slice (as fraction of parameter range)
    """
    create_figures_directory()
    
    n_params = len(param_names)
    
    # Calculate loss function
    def compute_loss(params):
        """Compute MSE loss for given parameters."""
        try:
            func_copy = deepcopy(func)
            func_copy.set_params(params)
            y_pred = np.array([func_copy(x) for x in x_data])
            return np.mean((y_data - y_pred)**2)
        except:
            return np.inf
    
    # Create figure with subplots for each parameter pair
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Get parameter ranges from MCMC samples
    param_ranges = []
    for i in range(n_params):
        param_min = np.percentile(samples[:, i], 1)
        param_max = np.percentile(samples[:, i], 99)
        param_ranges.append((param_min, param_max))
    
    # Calculate mean values for slicing
    mean_params = np.mean(samples, axis=0)
    
    plot_idx = 0
    
    # Plot cross-sections for each parameter pair
    for i in range(n_params):
        for j in range(i+1, n_params):
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Create grid for the two parameters
            param1_range = np.linspace(param_ranges[i][0], param_ranges[i][1], n_points)
            param2_range = np.linspace(param_ranges[j][0], param_ranges[j][1], n_points)
            X, Y = np.meshgrid(param1_range, param2_range)
            
            # Compute loss landscape
            Z = np.zeros_like(X)
            for idx1, p1 in enumerate(param1_range):
                for idx2, p2 in enumerate(param2_range):
                    # Create parameter vector with MAP values for other parameters
                    test_params = np.array([mean_params[k] for k in range(n_params)])
                    test_params[i] = p1
                    test_params[j] = p2
                    Z[idx2, idx1] = compute_loss(test_params)
            
            # Plot contour plot of likelihood (exp(-loss))
            likelihood = np.exp(-Z)
            contour = ax.contourf(X, Y, likelihood, levels=20, cmap='viridis', alpha=0.7)
            ax.contour(X, Y, likelihood, levels=10, colors='black', alpha=0.5, linewidths=0.5)
            
            # Find samples within epsilon of the slice
            slice_samples = []
            for sample in samples:
                # Check if all other parameters (not i or j) are within epsilon of their mean
                within_slice = True
                for k in range(n_params):
                    if k != i and k != j:
                        param_range_k = param_ranges[k][1] - param_ranges[k][0]
                        tolerance_k = epsilon * param_range_k
                        if abs(sample[k] - mean_params[k]) > tolerance_k:
                            within_slice = False
                            break
                
                if within_slice:
                    slice_samples.append(sample)
            
            slice_samples = np.array(slice_samples)
            
            # Overlay MCMC samples within the slice
            if len(slice_samples) > 0:
                ax.scatter(slice_samples[:, i], slice_samples[:, j], c='red', s=15, alpha=0.8, 
                          label=f'MCMC samples (slice, n={len(slice_samples)})', 
                          edgecolors='white', linewidth=0.5)
            else:
                ax.text(0.5, 0.5, 'No samples in slice', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Add MAP point if provided
            if map_params is not None:
                if isinstance(map_params, dict):
                    map_vals = list(map_params.values())
                else:
                    map_vals = map_params
                ax.scatter(map_vals[i], map_vals[j], c='yellow', s=200, 
                          marker='*', label='MAP estimate', edgecolors='black', linewidth=1)
            
            # Add mean of MCMC samples
            ax.scatter(mean_params[i], mean_params[j], c='cyan', s=150, 
                      marker='s', label='MCMC mean', edgecolors='black', linewidth=1)
            
            ax.set_xlabel(param_names[i])
            ax.set_ylabel(param_names[j])
            ax.set_title(f'Loss Landscape: {param_names[i]} vs {param_names[j]}\nSlice at mean values (ε={epsilon})')
            ax.legend()
            
            # Add colorbar
            cbar = plt.colorbar(contour, ax=ax, shrink=0.8)
            cbar.set_label('MSE Loss')
            
            plot_idx += 1
    
    # Remove extra subplots
    for i in range(plot_idx, len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_loss_landscape.png', dpi=300, bbox_inches='tight')
    
    # Create additional plot showing loss vs individual parameters
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
    axes2 = axes2.flatten()
    
    for i, param_name in enumerate(param_names):
        ax = axes2[i]
        
        # Create parameter range
        param_range = np.linspace(param_ranges[i][0], param_ranges[i][1], n_points)
        losses = []
        
        for p in param_range:
            test_params = np.array([np.mean(samples[:, k]) for k in range(n_params)])
            test_params[i] = p
            losses.append(compute_loss(test_params))
        
        # Plot loss landscape
        ax.plot(param_range, losses, 'b-', linewidth=2, label='Loss landscape')
        
        # Overlay MCMC samples
        sample_losses = []
        for sample in samples:
            sample_losses.append(compute_loss(sample))
        
        ax.scatter(samples[:, i], sample_losses, c='red', s=20, alpha=0.6, 
                  label='MCMC samples')
        
        # Add MAP point if provided
        if map_params is not None:
            if isinstance(map_params, dict):
                map_vals = list(map_params.values())
            else:
                map_vals = map_params
            map_loss = compute_loss(map_vals)
            ax.scatter(map_vals[i], map_loss, c='yellow', s=200, 
                      marker='*', label='MAP estimate', edgecolors='black', linewidth=1)
        
        # Add mean of MCMC samples
        mean_params = np.mean(samples, axis=0)
        mean_loss = compute_loss(mean_params)
        ax.scatter(mean_params[i], mean_loss, c='cyan', s=150, 
                  marker='s', label='MCMC mean', edgecolors='black', linewidth=1)
        
        ax.set_xlabel(param_name)
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'Loss vs {param_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_plots:
        plt.savefig('./figures/mcmc_loss_vs_parameters.png', dpi=300, bbox_inches='tight')
    
    print("Loss landscape plots saved in ./figures/")

def plot_all_mcmc_diagnostics(samples, param_names, acceptance_rate=None, save_plots=True, constraint_manager=None):
    """
    Generate all MCMC diagnostic plots.
    
    Args:
        samples: MCMC samples array (n_samples, n_params)
        param_names: List of parameter names
        acceptance_rate: MCMC acceptance rate
        save_plots: Whether to save plots to files
        constraint_manager: Optional ConstraintManager to filter valid samples
    """
    print("Generating comprehensive MCMC diagnostic plots...")
    
    # Filter samples that violate constraints if constraint_manager is provided
    if constraint_manager is not None:
        # Analyze constraint violations
        violation_analysis = analyze_constraint_violations(samples, constraint_manager)
        print("\nConstraint violation analysis:")
        for constraint_name, analysis in violation_analysis.items():
            if analysis['percentage'] > 0:
                print(f"  {analysis['description']}: {analysis['percentage']:.1f}% violations")
        

    # Main diagnostics
    plot_mcmc_diagnostics(samples, param_names, acceptance_rate, save_plots)
    
    # Uniform analysis
    plot_uniform_analysis(samples, param_names, save_plots)
    
    # Convergence diagnostics
    plot_convergence_diagnostics(samples, param_names, save_plots)
    
    # Credible intervals
    plot_credible_intervals(samples, param_names, save_plots=False)
    
    print("All MCMC diagnostic plots completed!")

def filter_valid_samples(samples, constraint_manager):
    """
    Filter out samples that violate constraints.
    
    Args:
        samples: MCMC samples array
        constraint_manager: ConstraintManager instance
        
    Returns:
        valid_samples: Array of samples that satisfy all constraints
    """
    if constraint_manager is None:
        return samples
    
    # Convert to numpy array if it's a list
    if isinstance(samples, list):
        samples = np.array(samples)
    
    constraints = constraint_manager.get_constraints()
    valid_indices = []
    
    for i, sample in enumerate(samples):
        valid = True
        for constraint in constraints:
            try:
                if constraint['fun'](sample) <= 0:
                    valid = False
                    break
            except Exception:
                valid = False
                break
        
        if valid:
            valid_indices.append(i)
    
    return samples[valid_indices]

def analyze_constraint_violations(samples, constraint_manager):
    """
    Analyze which constraints are violated most frequently.
    
    Args:
        samples: MCMC samples array
        constraint_manager: ConstraintManager instance
        
    Returns:
        violation_analysis: Dictionary with constraint violation statistics
    """
    if constraint_manager is None:
        return {}
    
    constraints = constraint_manager.get_constraints()
    n_samples = len(samples)
    constraint_violations = {i: 0 for i in range(len(constraints))}
    
    for sample in samples:
        for j, constraint in enumerate(constraints):
            try:
                if constraint['fun'](sample) <= 0:
                    constraint_violations[j] += 1
            except Exception:
                constraint_violations[j] += 1
    
    # Convert to percentages and create analysis
    violation_analysis = {}
    for j, count in constraint_violations.items():
        percentage = (count / n_samples) * 100
        violation_analysis[f"constraint_{j}"] = {
            'violations': count,
            'percentage': percentage,
            'description': f"Constraint {j+1}"
        }
    
    # Sort by violation percentage
    violation_analysis = dict(sorted(violation_analysis.items(), 
                                   key=lambda x: x[1]['percentage'], reverse=True))
    
    return violation_analysis 