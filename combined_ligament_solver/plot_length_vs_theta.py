"""
Script to plot ligament length vs theta angle
Shows how ligament lengths change with knee rotation
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.statics_solver.models.statics_model import KneeModel
from src.ligament_models.blankevoort import BlankevoortFunction
import sys

def plot_length_vs_theta(config, output_path=None):
    """
    Generate and plot ligament length vs theta angle
    
    Args:
        config: Configuration dictionary
        output_path: Optional path to save the plot
    """
    
    # Create ligament functions
    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    
    # Storage for data
    length_lcl = []  # LCL lengths
    length_mcl = []  # MCL lengths
    theta_list = []
    
    theta = 0
    moment_limit = 12_000  # In N(mm)
    
    print("Collecting data for positive theta (flexion)...")
    # Positive theta direction
    while True:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()
        
        moment = float(solutions['applied_force'].get_moment().norm())
        if moment > moment_limit:
            print(f"Moment limit reached at theta: {np.degrees(theta):.2f}°")
            break
        
        length_lcl.append(float(solutions['lig_springA_length']))
        length_mcl.append(float(solutions['lig_springB_length']))
        theta_list.append(theta)
        
        theta += 1/3 * np.pi/180
    
    print("Collecting data for negative theta (extension)...")
    # Negative theta direction
    theta = 0
    while True:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()
        
        moment = float(solutions['applied_force'].get_moment().norm())
        if moment > moment_limit:
            print(f"Moment limit reached at theta: {np.degrees(theta):.2f}°")
            break
        
        length_lcl.append(float(solutions['lig_springA_length']))
        length_mcl.append(float(solutions['lig_springB_length']))
        theta_list.append(theta)
        
        theta -= 1/3 * np.pi/180
    
    # Convert to numpy arrays and sort by theta
    theta_array = np.array(theta_list)
    length_lcl_array = np.array(length_lcl)
    length_mcl_array = np.array(length_mcl)
    
    sort_idx = np.argsort(theta_array)
    theta_sorted = theta_array[sort_idx]
    length_lcl_sorted = length_lcl_array[sort_idx]
    length_mcl_sorted = length_mcl_array[sort_idx]
    
    # Convert theta to degrees for plotting
    theta_deg = np.degrees(theta_sorted)
    
    print(f"\nTheta range: [{theta_deg.min():.2f}°, {theta_deg.max():.2f}°]")
    print(f"LCL length range: [{length_lcl_sorted.min():.2f}, {length_lcl_sorted.max():.2f}] mm")
    print(f"MCL length range: [{length_mcl_sorted.min():.2f}, {length_mcl_sorted.max():.2f}] mm")
    
    # Get reference lengths (l_0)
    l0_lcl = config['blankevoort_lcl']['l_0']
    l0_mcl = config['blankevoort_mcl']['l_0']
    
    # Calculate strain (relative to l_0)
    strain_lcl = (length_lcl_sorted - l0_lcl) / l0_lcl * 100  # Percentage
    strain_mcl = (length_mcl_sorted - l0_mcl) / l0_mcl * 100
    
    # Create figure with two subplots
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 12))
    
    # ============ TOP PLOT: Length with twin y-axes ============
    color_lcl = 'tab:blue'
    ax_top.set_xlabel('Knee Angle θ (degrees)', fontsize=14)
    ax_top.set_ylabel('LCL Length (mm)', fontsize=14, color=color_lcl)
    line1 = ax_top.plot(theta_deg, length_lcl_sorted, color=color_lcl, linewidth=3, 
                     label='LCL (Lateral)', marker='o', markersize=4, markevery=10)
    ax_top.tick_params(axis='y', labelcolor=color_lcl)
    ax_top.axhline(y=l0_lcl, color=color_lcl, linestyle='--', alpha=0.3, linewidth=1.5)
    ax_top.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax_top.grid(True, alpha=0.3)
    
    # Create second y-axis for MCL
    ax_top_twin = ax_top.twinx()
    color_mcl = 'tab:red'
    ax_top_twin.set_ylabel('MCL Length (mm)', fontsize=14, color=color_mcl)
    line2 = ax_top_twin.plot(theta_deg, length_mcl_sorted, color=color_mcl, linewidth=3, 
                     label='MCL (Medial)', marker='s', markersize=4, markevery=10)
    ax_top_twin.tick_params(axis='y', labelcolor=color_mcl)
    ax_top_twin.axhline(y=l0_mcl, color=color_mcl, linestyle='--', alpha=0.3, linewidth=1.5)
    
    # Add title and legend
    ax_top.set_title('Ligament Length vs Knee Angle', fontsize=16, fontweight='bold')
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax_top.legend(lines, labels, loc='upper center', fontsize=12)
    
    # ============ BOTTOM PLOT: Strain ============
    ax_bottom.plot(theta_deg, strain_lcl, color=color_lcl, linewidth=3, 
                   label='LCL (Lateral)', marker='o', markersize=4, markevery=10)
    ax_bottom.plot(theta_deg, strain_mcl, color=color_mcl, linewidth=3, 
                   label='MCL (Medial)', marker='s', markersize=4, markevery=10)
    ax_bottom.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax_bottom.axvline(x=0, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax_bottom.set_xlabel('Knee Angle θ (degrees)', fontsize=14)
    ax_bottom.set_ylabel('Ligament Strain (%)', fontsize=14)
    ax_bottom.set_title('Ligament Strain vs Knee Angle', fontsize=16, fontweight='bold')
    ax_bottom.legend(loc='upper center', fontsize=12)
    ax_bottom.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    return fig, {'theta': theta_sorted, 'length_lcl': length_lcl_sorted, 'length_mcl': length_mcl_sorted}

def main():
    """Main function"""
    
    # Parse command line arguments
    config_file = 'config.yaml'
    output_file = 'results/length_vs_theta.png'
    
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    # Load configuration
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print configuration info
    print("\nConfiguration:")
    print(f"  LCL l_0: {config['blankevoort_lcl']['l_0']:.2f} mm")
    print(f"  MCL l_0: {config['blankevoort_mcl']['l_0']:.2f} mm")
    print(f"  LCL k: {config['blankevoort_lcl']['k']:.2f}")
    print(f"  MCL k: {config['blankevoort_mcl']['k']:.2f}")
    print(f"  Left ligament length: {config['mechanics']['left_length']:.2f} mm")
    print(f"  Right ligament length: {config['mechanics']['right_length']:.2f} mm")
    
    # Generate plot
    fig, data = plot_length_vs_theta(config, output_file)
    
    print("\nShowing plot...")
    plt.show()

if __name__ == "__main__":
    main()

