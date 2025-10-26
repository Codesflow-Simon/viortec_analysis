"""
Script to visualize results where MCL strain = LCL strain (diagonal cases only)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def load_and_filter_csv(csv_path, ligament_type='MCL'):
    """Load CSV and filter for cases where MCL strain = LCL strain and specified ligament type"""
    df = pd.read_csv(csv_path)
    
    # Filter for ligament type
    if 'ligament_type' in df.columns:
        df = df[df['ligament_type'] == ligament_type].copy()
        print(f"Filtered for ligament type: {ligament_type}")
    
    # Filter for diagonal cases
    df_filtered = df[df['lcl_strain'] == df['mcl_strain']].copy()
    
    print(f"Loaded {len(df)} total results from {csv_path}")
    print(f"Filtered to {len(df_filtered)} diagonal cases (MCL strain = LCL strain)")
    print(f"Strain values: {sorted(df_filtered['mcl_strain'].unique())}")
    
    return df_filtered

def plot_parameter_estimates(df, output_dir=None):
    """Plot parameter estimates with error bars for diagonal cases"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    strains = df['mcl_strain'].values
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        mean_col = f'{param}_mean'
        std_col = f'{param}_std'
        gt_col = f'{param}_gt'
        
        # Plot mean estimates with error bars
        ax.errorbar(strains, df[mean_col], yerr=df[std_col], fmt='o-', capsize=5, 
                   label='Estimated (mean ± std)', alpha=0.7, linewidth=2, markersize=8)
        
        # Plot ground truth
        ax.plot(strains, df[gt_col], 's--', label='Ground Truth', linewidth=2, 
               markersize=8, color='red')
        
        ax.set_xlabel('Strain (MCL = LCL)', fontsize=12)
        ax.set_ylabel(f'{param} Value', fontsize=12)
        ax.set_title(f'{param} Estimates vs Ground Truth', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_estimates_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter estimates plot")
    
    return fig

def plot_parameter_errors(df, output_dir=None):
    """Plot relative errors for each parameter"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    strains = df['mcl_strain'].values
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        mean_col = f'{param}_mean'
        gt_col = f'{param}_gt'
        
        # Calculate relative error in percentage
        relative_error = np.abs(df[mean_col] - df[gt_col]) / np.abs(df[gt_col]) * 100
        
        # Plot error
        ax.plot(strains, relative_error, 'o-', linewidth=2, markersize=8, color='darkred')
        
        ax.set_xlabel('Strain (MCL = LCL)', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title(f'{param} - Relative Error', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 0
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_errors_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter errors plot")
    
    return fig

def plot_parameter_std(df, output_dir=None):
    """Plot parameter uncertainties (standard deviations)"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    strains = df['mcl_strain'].values
    
    for idx, param in enumerate(param_names):
        ax = axes[idx]
        
        std_col = f'{param}_std'
        
        # Plot standard deviation
        ax.plot(strains, df[std_col], 'o-', linewidth=2, markersize=8, color='steelblue')
        
        ax.set_xlabel('Strain (MCL = LCL)', fontsize=12)
        ax.set_ylabel('Standard Deviation', fontsize=12)
        ax.set_title(f'{param} - Uncertainty (Std Dev)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'parameter_uncertainties_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved parameter uncertainties plot")
    
    return fig

def plot_loss_comparison(df, output_dir=None):
    """Plot initial vs final loss"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    strains = df['mcl_strain'].values
    
    # Plot 1: Initial vs Final Loss
    ax1.plot(strains, df['initial_loss'], 'o-', label='Initial Loss', 
            linewidth=2, markersize=8, color='orange')
    ax1.plot(strains, df['final_loss'], 's-', label='Final Loss', 
            linewidth=2, markersize=8, color='green')
    ax1.set_xlabel('Strain (MCL = LCL)', fontsize=12)
    ax1.set_ylabel('Loss Value', fontsize=12)
    ax1.set_title('Optimization Loss Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss Reduction
    loss_reduction = df['initial_loss'] - df['final_loss']
    reduction_pct = (loss_reduction / df['initial_loss']) * 100
    
    ax2.bar(strains, reduction_pct, width=0.015, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Strain (MCL = LCL)', fontsize=12)
    ax2.set_ylabel('Loss Reduction (%)', fontsize=12)
    ax2.set_title('Optimization Loss Reduction', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'loss_comparison_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved loss comparison plot")
    
    return fig

def plot_mcmc_metrics(df, output_dir=None):
    """Plot MCMC acceptance rate and ground truth log-likelihood"""
    fig = plt.figure(figsize=(18, 5))
    
    strains = df['mcl_strain'].values
    
    # Plot 1: Acceptance Rate
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(strains, df['acceptance_rate'], 'o-', linewidth=2, markersize=8, color='purple')
    ax1.set_xlabel('Strain (MCL = LCL)', fontsize=12)
    ax1.set_ylabel('Acceptance Rate', fontsize=12)
    ax1.set_title('MCMC Acceptance Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Add horizontal line at ideal acceptance rate (0.234)
    ax1.axhline(y=0.234, color='red', linestyle='--', alpha=0.5, label='Ideal (0.234)')
    ax1.legend(fontsize=10)
    
    # Plot 2: Ground Truth Log-Likelihood (KDE)
    ax2 = plt.subplot(1, 3, 2)
    if 'gt_log_likelihood_kde' in df.columns:
        ax2.plot(strains, df['gt_log_likelihood_kde'], 's-', linewidth=2, markersize=8, color='darkgreen')
        ax2.set_xlabel('Strain (MCL = LCL)', fontsize=12)
        ax2.set_ylabel('Log-Likelihood', fontsize=12)
        ax2.set_title('Ground Truth Log-Likelihood (KDE)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'KDE data not available', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Ground Truth Log-Likelihood (KDE)', fontsize=13, fontweight='bold')
    
    # Plot 3: Ground Truth Log-Probability (Posterior)
    ax3 = plt.subplot(1, 3, 3)
    if 'gt_log_likelihood_posterior' in df.columns:
        ax3.plot(strains, df['gt_log_likelihood_posterior'], '^-', linewidth=2, markersize=8, color='darkblue')
        ax3.set_xlabel('Strain (MCL = LCL)', fontsize=12)
        ax3.set_ylabel('Log-Probability', fontsize=12)
        ax3.set_title('Ground Truth Log-Probability (Posterior)', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Posterior data not available', ha='center', va='center',
                transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Ground Truth Log-Probability (Posterior)', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'mcmc_metrics_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved MCMC metrics plot")
    
    return fig

def plot_covariance_metrics_diagonal(df, output_dir=None):
    """Plot overall variance metrics"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    cov_determinants = []
    cov_traces = []
    total_stds = []
    normalized_uncertainties = []
    
    for idx, row in df.iterrows():
        variances = np.array([row[f'{param}_std']**2 for param in param_names])
        cov_determinants.append(np.prod(variances))
        cov_traces.append(np.sum(variances))
        total_stds.append(np.sum([row[f'{param}_std'] for param in param_names]))
        
        # Normalized uncertainty (scale-invariant)
        cv_squared_sum = 0
        for param in param_names:
            std_val = row[f'{param}_std']
            gt_val = row[f'{param}_gt']
            if gt_val != 0:
                cv_squared = (std_val / gt_val) ** 2
                cv_squared_sum += cv_squared
        normalized_uncertainties.append(cv_squared_sum)
    
    strains = df['mcl_strain'].values
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    
    # Plot 1: Covariance Determinant (log scale)
    axes[0].semilogy(strains, cov_determinants, 'o-', linewidth=2, markersize=8, color='darkred')
    axes[0].set_xlabel('Strain (MCL = LCL)', fontsize=12)
    axes[0].set_ylabel('Determinant (log scale)', fontsize=12)
    axes[0].set_title('Covariance Determinant\nLower = Better Constrained', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Covariance Trace
    axes[1].plot(strains, cov_traces, 's-', linewidth=2, markersize=8, color='steelblue')
    axes[1].set_xlabel('Strain (MCL = LCL)', fontsize=12)
    axes[1].set_ylabel('Trace (Sum of Variances)', fontsize=12)
    axes[1].set_title('Covariance Trace\nLower = Less Uncertainty', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Total Standard Deviation
    axes[2].plot(strains, total_stds, '^-', linewidth=2, markersize=8, color='darkgreen')
    axes[2].set_xlabel('Strain (MCL = LCL)', fontsize=12)
    axes[2].set_ylabel('Sum of Std Deviations', fontsize=12)
    axes[2].set_title('Total Standard Deviation\nLower = Better Precision', 
                     fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Normalized Uncertainty (SCALE-INVARIANT)
    axes[3].plot(strains, normalized_uncertainties, 'd-', linewidth=2, markersize=8, color='purple')
    axes[3].set_xlabel('Strain (MCL = LCL)', fontsize=12)
    axes[3].set_ylabel('Sum of (σ/μ)²', fontsize=12)
    axes[3].set_title('Normalized Uncertainty (Scale-Invariant)\nLower = Better Constrained', 
                     fontsize=13, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'covariance_metrics_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved covariance metrics plot")
    
    return fig

def plot_theta_statistics(df, output_dir=None):
    """Plot theta range statistics"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    strains = df['mcl_strain'].values
    
    ax.plot(strains, df['max_theta_deg'], 'o-', label='Max Theta', 
           linewidth=2, markersize=8, color='red')
    ax.plot(strains, df['min_theta_deg'], 's-', label='Min Theta', 
           linewidth=2, markersize=8, color='blue')
    ax.plot(strains, df['lowest_force_theta_deg'], '^-', label='Theta at Lowest Force', 
           linewidth=2, markersize=8, color='green')
    
    ax.set_xlabel('Strain (MCL = LCL)', fontsize=12)
    ax.set_ylabel('Theta (degrees)', fontsize=12)
    ax.set_title('Theta Range Statistics', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'theta_statistics_diagonal.png'), dpi=300, bbox_inches='tight')
        print(f"Saved theta statistics plot")
    
    return fig

def create_summary_statistics(df, output_path=None):
    """Create a text summary of statistics"""
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    
    summary = []
    summary.append("=" * 80)
    summary.append("SUMMARY STATISTICS (Diagonal Cases: MCL Strain = LCL Strain)")
    summary.append("=" * 80)
    summary.append(f"\nTotal configurations: {len(df)}")
    summary.append(f"Strain values: {sorted(df['mcl_strain'].unique())}")
    
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
    
    # Compute metrics
    param_names = ['k', 'alpha', 'l_0', 'f_ref']
    cov_determinants = []
    cov_traces = []
    normalized_uncertainties = []
    for idx, row in df.iterrows():
        variances = np.array([row[f'{param}_std']**2 for param in param_names])
        cov_determinants.append(np.prod(variances))
        cov_traces.append(np.sum(variances))
        
        # Normalized uncertainty
        cv_squared_sum = 0
        for param in param_names:
            std_val = row[f'{param}_std']
            gt_val = row[f'{param}_gt']
            if gt_val != 0:
                cv_squared = (std_val / gt_val) ** 2
                cv_squared_sum += cv_squared
        normalized_uncertainties.append(cv_squared_sum)
    
    summary.append(f"  Covariance Determinant (geometric mean): {np.exp(np.mean(np.log(np.array(cov_determinants) + 1e-20))):.3e}")
    summary.append(f"  Covariance Trace (mean):                 {np.mean(cov_traces):.3e}")
    summary.append(f"  Covariance Trace (median):               {np.median(cov_traces):.3e}")
    summary.append(f"  Covariance Trace (min):                  {np.min(cov_traces):.3e}")
    summary.append(f"  Covariance Trace (max):                  {np.max(cov_traces):.3e}")
    summary.append(f"\n  Normalized Uncertainty (SCALE-INVARIANT):")
    summary.append(f"    Mean:   {np.mean(normalized_uncertainties):.4f}")
    summary.append(f"    Median: {np.median(normalized_uncertainties):.4f}")
    summary.append(f"    Min:    {np.min(normalized_uncertainties):.4f}")
    summary.append(f"    Max:    {np.max(normalized_uncertainties):.4f}")
    
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
    
    # Load and filter data
    df = load_and_filter_csv(csv_path, ligament_type)
    
    if len(df) == 0:
        print("ERROR: No diagonal cases found in the data!")
        return
    
    # Create output directory for plots
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    plots_dir = os.path.join(csv_dir, f'{csv_name}_{ligament_type.lower()}_diagonal_plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"\nGenerating visualizations in {plots_dir}...")
    
    # Generate all plots
    plot_parameter_estimates(df, plots_dir)
    plot_parameter_errors(df, plots_dir)
    plot_parameter_std(df, plots_dir)
    plot_covariance_metrics_diagonal(df, plots_dir)
    plot_loss_comparison(df, plots_dir)
    plot_mcmc_metrics(df, plots_dir)
    plot_theta_statistics(df, plots_dir)
    
    # Create summary statistics
    summary_path = os.path.join(plots_dir, 'summary_statistics_diagonal.txt')
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
                print("Usage: python visualize_diagonal.py [path_to_summary.csv] [ligament_type]")
                print("       ligament_type: MCL or LCL (default: MCL)")
                sys.exit(1)
        else:
            print("No results directory found.")
            print("Usage: python visualize_diagonal.py [path_to_summary.csv] [ligament_type]")
            print("       ligament_type: MCL or LCL (default: MCL)")
            sys.exit(1)
    
    main(csv_path, ligament_type)

