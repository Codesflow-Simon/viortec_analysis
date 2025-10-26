"""
Script to plot likelihood comparisons (KDE vs Posterior) from saved results.

Usage:
    python plot_likelihood_comparison.py results/results_TIMESTAMP.pkl
"""

import pickle
import sys
import yaml
from main import plot_likelihood_comparison

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_likelihood_comparison.py <path_to_results.pkl>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load results
    print(f"Loading results from {results_file}...")
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} result sets")
    
    # Plot for each result
    for idx, result in enumerate(results):
        lcl_strain = result['lcl_strain']
        mcl_strain = result['mcl_strain']
        
        print(f"\n{'='*60}")
        print(f"Result {idx+1}/{len(results)}: LCL strain={lcl_strain:.2f}, MCL strain={mcl_strain:.2f}")
        print(f"{'='*60}")
        
        # Plot LCL
        lcl_data = result['result']['LCL']
        if lcl_data['sampler'] is not None:
            print(f"\nPlotting LCL likelihood comparison...")
            plot_likelihood_comparison(
                lcl_data, 
                lcl_data['sampler'], 
                config, 
                ligament_type=f"LCL (strain={lcl_strain:.2f})"
            )
        else:
            print(f"Skipping LCL (no valid sampler)")
        
        # Plot MCL
        mcl_data = result['result']['MCL']
        if mcl_data['sampler'] is not None:
            print(f"\nPlotting MCL likelihood comparison...")
            plot_likelihood_comparison(
                mcl_data, 
                mcl_data['sampler'], 
                config, 
                ligament_type=f"MCL (strain={mcl_strain:.2f})"
            )
        else:
            print(f"Skipping MCL (no valid sampler)")

if __name__ == "__main__":
    main()




