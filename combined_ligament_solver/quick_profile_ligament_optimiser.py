#!/usr/bin/env python3
"""
Quick profiling script for ligament optimiser functions.
This script provides a fast analysis of the current performance.
"""

import cProfile
import pstats
import io
import time
import numpy as np
import yaml
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament, least_squares_optimize_complete_model
from src.ligament_models.constraints import ConstraintManager
from src.ligament_models.blankevoort import BlankevoortFunction
from src.statics_solver.models.statics_model import KneeModel

def load_configs():
    """Load configuration files."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)
    
    return config, constraints_config

def generate_test_data():
    """Generate synthetic test data for profiling."""
    x_data = np.linspace(50, 100, 20)  # Reduced from 50 to 20 points
    y_data = 50 * (x_data - 60)**2 + 100
    noise = np.random.normal(0, 10, len(y_data))
    y_data += noise
    return x_data, y_data

def generate_knee_test_data(config):
    """Generate test data for complete knee model optimization."""
    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    
    data_lists = {
        'applied_force': [], 'applied_moment': [],
        'length_known_a': [], 'force_known_a': [], 'moment_known_a': [],
        'length_known_b': [], 'force_known_b': [], 'moment_known_b': [],
        'thetas': []
    }
    
    # Generate data for a smaller range for faster profiling
    thetas = np.linspace(-0.3, 0.3, 5)  # Only 5 points for quick profiling
    
    knee_model = KneeModel(config['mechanics'], log=False)
    knee_model.build_geometry()
    knee_model.build_ligament_forces(lig_left, lig_right)
    
    for theta in thetas:
        try:
            knee_model.assemble_equations(theta)
            solutions = knee_model.solve()
            
            data_lists['applied_force'].append(float(solutions['applied_force'].get_force().norm()))
            data_lists['applied_moment'].append(float(solutions['applied_force'].get_moment().norm()))
            data_lists['length_known_a'].append(float(solutions['lig_springA_length']))
            data_lists['force_known_a'].append(float(solutions['lig_springA_force'].get_force().norm()))
            data_lists['length_known_b'].append(float(solutions['lig_springB_length']))
            data_lists['force_known_b'].append(float(solutions['lig_springB_force'].get_force().norm()))
            data_lists['thetas'].append(theta)
        except Exception as e:
            print(f"Warning: Could not collect data at theta={theta}: {e}")
            continue
    
    return data_lists

def profile_individual_ligament():
    """Profile individual ligament reconstruction."""
    print("=" * 50)
    print("PROFILING INDIVIDUAL LIGAMENT RECONSTRUCTION")
    print("=" * 50)
    
    config, constraints_config = load_configs()
    constraint_manager = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    x_data, y_data = generate_test_data()
    
    print(f"Test data: {len(x_data)} points")
    
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = reconstruct_ligament(x_data, y_data, constraint_manager)
    end_time = time.time()
    
    pr.disable()
    
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"Final loss: {result['loss']:.6f}")
    print(f"Optimization success: {result['opt_result'].success}")
    print(f"Iterations: {result['opt_result'].nit}")
    print(f"Function evaluations: {result['opt_result'].nfev}")
    
    # Show top functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("\nTop 10 functions by cumulative time:")
    print(s.getvalue())
    
    return result, pr

def profile_complete_model():
    """Profile complete model optimization."""
    print("\n" + "=" * 50)
    print("PROFILING COMPLETE MODEL OPTIMIZATION")
    print("=" * 50)
    
    config, constraints_config = load_configs()
    constraint_manager_mcl = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    constraint_manager_lcl = ConstraintManager(constraints_config=constraints_config['blankevoort_lcl'])
    constraint_manager = (constraint_manager_mcl, constraint_manager_lcl)
    
    data = generate_knee_test_data(config)
    
    print(f"Test data: {len(data['thetas'])} points")
    
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = least_squares_optimize_complete_model(
        data['thetas'], 
        data['applied_force'], 
        data['length_known_a'], 
        data['length_known_b'],
        constraint_manager, 
        config['mechanics'], 
        sigma_noise=1e3
    )
    end_time = time.time()
    
    pr.disable()
    
    print(f"Execution time: {end_time - start_time:.4f} seconds")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE: {result['mae']:.2f}")
    print(f"Optimization success: {result['optimization_result'].success}")
    print(f"Iterations: {result['optimization_result'].nit}")
    print(f"Function evaluations: {result['optimization_result'].nfev}")
    
    # Show top functions
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)
    print("\nTop 10 functions by cumulative time:")
    print(s.getvalue())
    
    return result, pr

def profile_loss_functions():
    """Profile individual loss functions."""
    print("\n" + "=" * 50)
    print("PROFILING LOSS FUNCTIONS")
    print("=" * 50)
    
    from src.ligament_reconstructor.ligament_optimiser import loss, loss_jac, loss_hess
    
    config, constraints_config = load_configs()
    constraint_manager = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    x_data, y_data = generate_test_data()
    
    initial_guess_list = [40, 0.06, 90.0, 0.0]
    function = BlankevoortFunction(initial_guess_list)
    params = np.array(initial_guess_list)
    
    print(f"Testing with {len(x_data)} data points")
    
    # Profile loss function
    start_time = time.time()
    for _ in range(50):
        loss_val = loss(params, x_data, y_data, function, include_reg=True)
    loss_time = (time.time() - start_time) / 50
    
    # Profile Jacobian
    start_time = time.time()
    for _ in range(50):
        jac_val = loss_jac(params, x_data, y_data, function, include_reg=True)
    jac_time = (time.time() - start_time) / 50
    
    # Profile Hessian
    start_time = time.time()
    for _ in range(25):
        hess_val = loss_hess(params, x_data, y_data, function, include_reg=True)
    hess_time = (time.time() - start_time) / 25
    
    print(f"Loss function: {loss_time*1000:.2f} ms per call")
    print(f"Jacobian: {jac_time*1000:.2f} ms per call")
    print(f"Hessian: {hess_time*1000:.2f} ms per call")
    print(f"Loss value: {loss_val:.6f}")
    print(f"Gradient norm: {np.linalg.norm(jac_val):.6f}")

def main():
    """Main profiling function."""
    print("QUICK LIGAMENT OPTIMISER PROFILING")
    print("=" * 50)
    
    try:
        # Profile individual ligament reconstruction
        individual_result = profile_individual_ligament()
        
        # Profile complete model optimization
        complete_result = profile_complete_model()
        
        # Profile loss functions
        profile_loss_functions()
        
        # Save profiling results
        individual_result[1].dump_stats('individual_ligament_quick.prof')
        complete_result[1].dump_stats('complete_model_quick.prof')
        
        print("\n" + "=" * 50)
        print("PROFILING COMPLETED")
        print("=" * 50)
        print("Profiling files saved:")
        print("  - individual_ligament_quick.prof")
        print("  - complete_model_quick.prof")
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

