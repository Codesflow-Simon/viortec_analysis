"""
Script to visualize results from the summary CSV file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from matplotlib.gridspec import GridSpec

def load_csv(csv_path, ligament_type='MCL'):
    """Load the summary CSV and filter by ligament type"""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} total results from {csv_path}")
    
    # Filter for ligament type if column exists
    if 'ligament_type' in df.columns:
        df = df[df['ligament_type'] == ligament_type].copy()
        print(f"Filtered for ligament type: {ligament_type}")
        print(f"Results after filtering: {len(df)}")
    else:
        print("Warning: 'ligament_type' column not found. Using all data.")
    
    print(f"Columns: {list(df.columns)}")
    return df

def plot_parameter_error_heatmaps(df, output_dir=None):
    """Create heatmaps showing parameter recovery error"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        # Calculate relative error
        mean_col = f'{param}_mean'
        gt_col = f'{param}_gt'
        
        df['error'] = np.abs(df[mean_col] - df[gt_col]) / np.abs(df[gt_col]) * 100
        
        # Pivot for heatmap
        pivot = df.pivot(index='lcl_strain', columns='mcl_strain', values='error')
        
        # Plot
        im = sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax, 
                        cbar_kws={'label': 'Relative Error (%)'})
        ax.set_title(f'{param} - Relative Error (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('MCL Strain', fontsize=10)
        ax.set_ylabel('LCL Strain', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_errors.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter error heatmaps")
    
    return fig

def plot_loss_heatmaps(df, output_dir=None):
    """Create heatmaps for initial and final loss"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Initial loss
    pivot_initial = df.pivot(index='lcl_strain', columns='mcl_strain', values='initial_loss')
    sns.heatmap(pivot_initial, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': 'Loss'})
    axes[0].set_title('Initial Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('MCL Strain')
    axes[0].set_ylabel('LCL Strain')
    
    # Final loss
    pivot_final = df.pivot(index='lcl_strain', columns='mcl_strain', values='final_loss')
    sns.heatmap(pivot_final, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'Loss'})
    axes[1].set_title('Final Loss (After Optimization)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('MCL Strain')
    axes[1].set_ylabel('LCL Strain')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loss_values.png'), dpi=300, bbox_inches='tight')
        print(f"Saved loss heatmaps")
    
    return fig

def plot_parameter_std_heatmaps(df, output_dir=None):
    """Create heatmaps showing parameter standard deviations"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        std_col = f'{param}_std'
        
        # Pivot for heatmap
        pivot = df.pivot(index='lcl_strain', columns='mcl_strain', values=std_col)
        
        # Plot
        im = sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, 
                        cbar_kws={'label': 'Standard Deviation'})
        ax.set_title(f'{param} - Standard Deviation', fontsize=12, fontweight='bold')
        ax.set_xlabel('MCL Strain', fontsize=10)
        ax.set_ylabel('LCL Strain', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_std.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter standard deviation heatmaps")
    
    return fig

def plot_acceptance_rate(df, output_dir=None):
    """Plot acceptance rate heatmap"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    pivot = df.pivot(index='lcl_strain', columns='mcl_strain', values='acceptance_rate')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax,
                cbar_kws={'label': 'Acceptance Rate'})
    ax.set_title('MCMC Acceptance Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('MCL Strain', fontsize=11)
    ax.set_ylabel('LCL Strain', fontsize=11)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'acceptance_rate.png'), dpi=300, bbox_inches='tight')
        print(f"Saved acceptance rate heatmap")
    
    return fig

def plot_covariance_metrics(df, output_dir=None):
    """Plot overall variance metrics from covariance matrix"""
    
    # Compute covariance metrics for each row
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    n_params = len(param_names)
    
    cov_determinants = []
    cov_traces = []
    total_variances = []
    normalized_uncertainties = []  # Scale-invariant metric
    
    for idx, row in df.iterrows():
        # Reconstruct covariance matrix from standard deviations (diagonal approximation)
        # Note: We only have std values, so we approximate with diagonal covariance
        variances = np.array([row[f'{param}_std']**2 for param in param_names])
        
        # Determinant (product of variances for diagonal matrix)
        det = np.prod(variances)
        cov_determinants.append(det)
        
        # Trace (sum of variances)
        trace = np.sum(variances)
        cov_traces.append(trace)
        
        # Total variance (same as trace for this purpose)
        total_variances.append(trace)
        
        # Normalized uncertainty: sum of squared coefficients of variation
        # CV = std/mean, so CV² = var/mean²
        # Use ground truth values as reference
        cv_squared_sum = 0
        for param in param_names:
            std_val = row[f'{param}_std']
            gt_val = row[f'{param}_gt']
            if gt_val != 0:
                cv_squared = (std_val / gt_val) ** 2
                cv_squared_sum += cv_squared
        normalized_uncertainties.append(cv_squared_sum)
    
    df['cov_determinant'] = cov_determinants
    df['cov_trace'] = cov_traces
    df['normalized_uncertainty'] = normalized_uncertainties
    
    # Create figure with four plots
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Plot 1: Covariance Determinant (log scale)
    pivot_det = df.pivot(index='lcl_strain', columns='mcl_strain', values='cov_determinant')
    # Use log scale for better visualization
    pivot_det_log = np.log10(pivot_det + 1e-20)  # Add small value to avoid log(0)
    sns.heatmap(pivot_det_log, annot=False, cmap='YlOrRd', ax=axes[0],
                cbar_kws={'label': 'log₁₀(Determinant)'})
    axes[0].set_title('Covariance Determinant (log scale)\nLower = Better Constrained', 
                     fontsize=12, fontweight='bold')
    axes[0].set_xlabel('MCL Strain', fontsize=11)
    axes[0].set_ylabel('LCL Strain', fontsize=11)
    
    # Plot 2: Covariance Trace
    pivot_trace = df.pivot(index='lcl_strain', columns='mcl_strain', values='cov_trace')
    sns.heatmap(pivot_trace, annot=True, fmt='.2e', cmap='YlOrRd', ax=axes[1],
                cbar_kws={'label': 'Trace (Sum of Variances)'})
    axes[1].set_title('Covariance Trace\nLower = Less Uncertainty', 
                     fontsize=12, fontweight='bold')
    axes[1].set_xlabel('MCL Strain', fontsize=11)
    axes[1].set_ylabel('LCL Strain', fontsize=11)
    
    # Plot 3: Total Standard Deviation (sum of std values)
    total_std = df[[f'{param}_std' for param in param_names]].sum(axis=1)
    df['total_std'] = total_std
    pivot_std = df.pivot(index='lcl_strain', columns='mcl_strain', values='total_std')
    sns.heatmap(pivot_std, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[2],
                cbar_kws={'label': 'Total Std Dev'})
    axes[2].set_title('Sum of Standard Deviations\nLower = Better Precision', 
                     fontsize=12, fontweight='bold')
    axes[2].set_xlabel('MCL Strain', fontsize=11)
    axes[2].set_ylabel('LCL Strain', fontsize=11)
    
    # Plot 4: Normalized Uncertainty (SCALE-INVARIANT)
    pivot_norm = df.pivot(index='lcl_strain', columns='mcl_strain', values='normalized_uncertainty')
    sns.heatmap(pivot_norm, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[3],
                cbar_kws={'label': 'Sum of CV²'})
    axes[3].set_title('Normalized Uncertainty (Scale-Invariant)\nSum of (σ/μ)² - Lower = Better', 
                     fontsize=12, fontweight='bold')
    axes[3].set_xlabel('MCL Strain', fontsize=11)
    axes[3].set_ylabel('LCL Strain', fontsize=11)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'covariance_metrics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved covariance metrics heatmaps")
    
    # Create a separate, larger plot for just the normalized uncertainty
    fig2, ax = plt.subplots(figsize=(10, 8))
    pivot_norm = df.pivot(index='lcl_strain', columns='mcl_strain', values='normalized_uncertainty')
    sns.heatmap(pivot_norm, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=ax,
                cbar_kws={'label': 'Sum of (σ/μ)²'}, annot_kws={'size': 11})
    ax.set_title('Normalized Uncertainty (Scale-Invariant)\nSum of Squared Coefficients of Variation\nLower = Better Constrained Parameters', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('MCL Strain', fontsize=12)
    ax.set_ylabel('LCL Strain', fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'normalized_uncertainty.png'), dpi=300, bbox_inches='tight')
        print(f"Saved normalized uncertainty heatmap")
    
    return fig

def plot_gt_log_likelihood(df, output_dir=None):
    """Plot ground truth log-likelihood under KDE and posterior"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # KDE plot
    if 'gt_log_likelihood_kde' in df.columns:
        pivot_kde = df.pivot(index='lcl_strain', columns='mcl_strain', values='gt_log_likelihood_kde')
        sns.heatmap(pivot_kde, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[0],
                    cbar_kws={'label': 'Log-Likelihood'})
        axes[0].set_title('Ground Truth Log-Likelihood (KDE)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('MCL Strain', fontsize=11)
        axes[0].set_ylabel('LCL Strain', fontsize=11)
    else:
        axes[0].text(0.5, 0.5, 'KDE data not available', ha='center', va='center', 
                    transform=axes[0].transAxes, fontsize=14)
        axes[0].set_title('Ground Truth Log-Likelihood (KDE)', fontsize=14, fontweight='bold')
    
    # Posterior plot
    if 'gt_log_likelihood_posterior' in df.columns:
        pivot_post = df.pivot(index='lcl_strain', columns='mcl_strain', values='gt_log_likelihood_posterior')
        sns.heatmap(pivot_post, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1],
                    cbar_kws={'label': 'Log-Probability'})
        axes[1].set_title('Ground Truth Log-Probability (Posterior)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('MCL Strain', fontsize=11)
        axes[1].set_ylabel('LCL Strain', fontsize=11)
    else:
        axes[1].text(0.5, 0.5, 'Posterior data not available', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=14)
        axes[1].set_title('Ground Truth Log-Probability (Posterior)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'gt_log_likelihood_comparison.png'), dpi=300, bbox_inches='tight')
        print(f"Saved ground truth log-likelihood comparison heatmaps")
    
    return fig

def plot_theta_range(df, output_dir=None):
    """Plot theta statistics"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['max_theta_deg', 'min_theta_deg', 'lowest_force_theta_deg']
    titles = ['Maximum Theta (degrees)', 'Minimum Theta (degrees)', 'Theta at Lowest Force (degrees)']
    
    for ax, metric, title in zip(axes, metrics, titles):
        pivot = df.pivot(index='lcl_strain', columns='mcl_strain', values=metric)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    cbar_kws={'label': 'Degrees'})
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('MCL Strain', fontsize=10)
        ax.set_ylabel('LCL Strain', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'theta_statistics.png'), dpi=300, bbox_inches='tight')
        print(f"Saved theta statistics heatmaps")
    
    return fig

def plot_parameter_estimates(df, output_dir=None):
    """Plot parameter estimates with error bars"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        mean_col = f'{param}_mean'
        std_col = f'{param}_std'
        gt_col = f'{param}_gt'
        
        x = range(len(df))
        
        # Plot mean estimates with error bars
        ax.errorbar(x, df[mean_col], yerr=df[std_col], fmt='o', capsize=3, 
                   label='Estimated', alpha=0.6)
        
        # Plot ground truth
        ax.plot(x, df[gt_col], 'r--', label='Ground Truth', linewidth=2)
        
        ax.set_xlabel('Configuration Index', fontsize=10)
        ax.set_ylabel(f'{param} Value', fontsize=10)
        ax.set_title(f'{param} Estimates vs Ground Truth', fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_estimates.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter estimates plot")
    
    return fig

def plot_loss_reduction(df, output_dir=None):
    """Plot loss reduction from optimization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = range(len(df))
    width = 0.35
    
    ax.bar([i - width/2 for i in x], df['initial_loss'], width, label='Initial Loss', alpha=0.7)
    ax.bar([i + width/2 for i in x], df['final_loss'], width, label='Final Loss', alpha=0.7)
    
    ax.set_xlabel('Configuration Index', fontsize=11)
    ax.set_ylabel('Loss Value', fontsize=11)
    ax.set_title('Optimization Loss Reduction', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loss_reduction.png'), dpi=300, bbox_inches='tight')
        print(f"Saved loss reduction plot")
    
    return fig

def create_summary_statistics(df, output_path=None):
    """Create a text summary of statistics"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    summary = []
    summary.append("=" * 80)
    summary.append("SUMMARY STATISTICS")
    summary.append("=" * 80)
    summary.append(f"\nTotal configurations: {len(df)}")
    summary.append(f"LCL strains: {sorted(df['lcl_strain'].unique())}")
    summary.append(f"MCL strains: {sorted(df['mcl_strain'].unique())}")
    
    summary.append("\n" + "=" * 80)
    summary.append("PARAMETER ERRORS (Relative %)")
    summary.append("=" * 80)
    
    for param in param_names:
        mean_col = f'{param}_mean'
        gt_col = f'{param}_gt'
        errors = np.abs(df[mean_col] - df[gt_col]) / np.abs(df[gt_col]) * 100
        
        summary.append(f"\n{param.upper()}:")
        summary.append(f"  Mean error:   {errors.mean():.2f}%")
        summary.append(f"  Median error: {errors.median():.2f}%")
        summary.append(f"  Std error:    {errors.std():.2f}%")
        summary.append(f"  Min error:    {errors.min():.2f}%")
        summary.append(f"  Max error:    {errors.max():.2f}%")
    
    summary.append("\n" + "=" * 80)
    summary.append("MCMC ACCEPTANCE RATE")
    summary.append("=" * 80)
    summary.append(f"  Mean:   {df['acceptance_rate'].mean():.4f}")
    summary.append(f"  Median: {df['acceptance_rate'].median():.4f}")
    summary.append(f"  Min:    {df['acceptance_rate'].min():.4f}")
    summary.append(f"  Max:    {df['acceptance_rate'].max():.4f}")
    
    summary.append("\n" + "=" * 80)
    summary.append("OPTIMIZATION PERFORMANCE")
    summary.append("=" * 80)
    summary.append(f"  Mean initial loss: {df['initial_loss'].mean():.2f}")
    summary.append(f"  Mean final loss:   {df['final_loss'].mean():.2f}")
    summary.append(f"  Mean reduction:    {(df['initial_loss'] - df['final_loss']).mean():.2f}")
    summary.append(f"  Mean reduction %:  {((df['initial_loss'] - df['final_loss']) / df['initial_loss'] * 100).mean():.1f}%")
    
    summary.append("\n" + "=" * 80)
    summary.append("COVARIANCE METRICS (OVERALL UNCERTAINTY)")
    summary.append("=" * 80)
    
    # Compute metrics if not already computed
    if 'cov_determinant' not in df.columns:
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        cov_determinants = []
        cov_traces = []
        for idx, row in df.iterrows():
            variances = np.array([row[f'{param}_std']**2 for param in param_names])
            cov_determinants.append(np.prod(variances))
            cov_traces.append(np.sum(variances))
        df['cov_determinant'] = cov_determinants
        df['cov_trace'] = cov_traces
    
    summary.append(f"  Covariance Determinant (geometric mean): {np.exp(np.mean(np.log(df['cov_determinant'] + 1e-20))):.3e}")
    summary.append(f"  Covariance Trace (mean):                 {df['cov_trace'].mean():.3e}")
    summary.append(f"  Covariance Trace (median):               {df['cov_trace'].median():.3e}")
    summary.append(f"  Covariance Trace (min):                  {df['cov_trace'].min():.3e}")
    summary.append(f"  Covariance Trace (max):                  {df['cov_trace'].max():.3e}")
    summary.append(f"\n  Normalized Uncertainty (SCALE-INVARIANT):")
    summary.append(f"    Mean:   {df['normalized_uncertainty'].mean():.4f}")
    summary.append(f"    Median: {df['normalized_uncertainty'].median():.4f}")
    summary.append(f"    Min:    {df['normalized_uncertainty'].min():.4f}")
    summary.append(f"    Max:    {df['normalized_uncertainty'].max():.4f}")
    
    summary.append("\n" + "=" * 80)
    summary.append("GROUND TRUTH LOG-LIKELIHOOD (KDE)")
    summary.append("=" * 80)
    if 'gt_log_likelihood_kde' in df.columns:
        summary.append(f"  Mean:   {df['gt_log_likelihood_kde'].mean():.3f}")
        summary.append(f"  Median: {df['gt_log_likelihood_kde'].median():.3f}")
        summary.append(f"  Min:    {df['gt_log_likelihood_kde'].min():.3f}")
        summary.append(f"  Max:    {df['gt_log_likelihood_kde'].max():.3f}")
    else:
        summary.append("  Not available in data")
    
    summary.append("\n" + "=" * 80)
    summary.append("GROUND TRUTH LOG-PROBABILITY (POSTERIOR)")
    summary.append("=" * 80)
    if 'gt_log_likelihood_posterior' in df.columns:
        summary.append(f"  Mean:   {df['gt_log_likelihood_posterior'].mean():.3f}")
        summary.append(f"  Median: {df['gt_log_likelihood_posterior'].median():.3f}")
        summary.append(f"  Min:    {df['gt_log_likelihood_posterior'].min():.3f}")
        summary.append(f"  Max:    {df['gt_log_likelihood_posterior'].max():.3f}")
    else:
        summary.append("  Not available in data")
    
    summary_text = "\n".join(summary)
    print(summary_text)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(summary_text)
        print(f"\nSaved summary to {output_path}")
    
    return summary_text

def main(csv_path, ligament_type='MCL'):
    """Main visualization function"""
    
    # Load data
    df = load_csv(csv_path, ligament_type)
    
    # Create output directory for plots
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    plots_dir = os.path.join(csv_dir, f'{csv_name}_{ligament_type.lower()}_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations in {plots_dir}...")
    
    # Generate all plots
    plot_parameter_error_heatmaps(df, plots_dir)
    plot_parameter_std_heatmaps(df, plots_dir)
    plot_covariance_metrics(df, plots_dir)
    plot_loss_heatmaps(df, plots_dir)
    plot_acceptance_rate(df, plots_dir)
    if 'gt_log_likelihood_kde' in df.columns or 'gt_log_likelihood_posterior' in df.columns:
        plot_gt_log_likelihood(df, plots_dir)
    plot_theta_range(df, plots_dir)
    plot_parameter_estimates(df, plots_dir)
    plot_loss_reduction(df, plots_dir)
    
    # Create summary statistics
    summary_path = os.path.join(plots_dir, 'summary_statistics.txt')
    create_summary_statistics(df, summary_path)
    
    print(f"\n✓ All visualizations saved to: {plots_dir}")

if __name__ == "__main__":
    # Parse arguments
    ligament_type = 'MCL'  # Default
    csv_path = None
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    if len(sys.argv) > 2:
        ligament_type = sys.argv[2].upper()
        if ligament_type not in ['MCL', 'LCL']:
            print(f"ERROR: Invalid ligament type '{sys.argv[2]}'. Must be 'MCL' or 'LCL'.")
            sys.exit(1)
    
    if csv_path is None:
        # Find most recent summary CSV
        results_dir = 'results'
        if os.path.exists(results_dir):
            csv_files = [f for f in os.listdir(results_dir) if f.startswith('summary_') and f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(results_dir, sorted(csv_files)[-1])
                print(f"Using most recent CSV: {csv_path}")
            else:
                print("No summary CSV files found in results/")
                print("Usage: python visualize_csv.py [path_to_summary.csv] [ligament_type]")
                print("       ligament_type: MCL or LCL (default: MCL)")
                sys.exit(1)
        else:
            print("No results directory found.")
            print("Usage: python visualize_csv.py [path_to_summary.csv] [ligament_type]")
            print("       ligament_type: MCL or LCL (default: MCL)")
            sys.exit(1)
    
    main(csv_path, ligament_type)

