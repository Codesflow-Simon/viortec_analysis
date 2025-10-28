#!/usr/bin/env python3
"""
Profiling script for ligament optimiser functions.
This script profiles both individual ligament reconstruction and complete model optimization.
"""

import cProfile
import pstats
import io
import time
import numpy as np
import yaml
import matplotlib.pyplot as plt
from contextlib import contextmanager
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ligament_reconstructor.ligament_optimiser import reconstruct_ligament, least_squares_optimize_complete_model
from src.ligament_models.constraints import ConstraintManager
from src.ligament_models.blankevoort import BlankevoortFunction
from src.statics_solver.models.statics_model import KneeModel

@contextmanager
def profile_context():
    """Context manager for profiling code blocks."""
    pr = cProfile.Profile()
    pr.enable()
    yield pr
    pr.disable()

def load_configs():
    """Load configuration files."""
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)
    
    return config, constraints_config

def generate_test_data():
    """Generate synthetic test data for profiling."""
    # Generate synthetic ligament data
    x_data = np.linspace(50, 100, 50)  # Length data
    y_data = 50 * (x_data - 60)**2 + 100  # Force data (quadratic relationship)
    
    # Add some noise
    noise = np.random.normal(0, 10, len(y_data))
    y_data += noise
    
    return x_data, y_data

def generate_knee_test_data(config):
    """Generate test data for complete knee model optimization."""
    # Use the same data generation as in main.py but with fewer points for faster profiling
    lig_left = BlankevoortFunction(config['blankevoort_lcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    
    data_lists = {
        'applied_force': [], 'applied_moment': [],
        'length_known_a': [], 'force_known_a': [], 'moment_known_a': [],
        'length_known_b': [], 'force_known_b': [], 'moment_known_b': [],
        'thetas': []
    }
    
    # Generate data for a smaller range for faster profiling
    thetas = np.linspace(-0.5, 0.5, 10)  # 10 points instead of many
    
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

def profile_individual_ligament_reconstruction():
    """Profile the individual ligament reconstruction function."""
    print("=" * 60)
    print("PROFILING INDIVIDUAL LIGAMENT RECONSTRUCTION")
    print("=" * 60)
    
    # Load configurations
    config, constraints_config = load_configs()
    constraint_manager = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    
    # Generate test data
    x_data, y_data = generate_test_data()
    
    print(f"Test data: {len(x_data)} points")
    print(f"X range: {x_data.min():.2f} to {x_data.max():.2f}")
    print(f"Y range: {y_data.min():.2f} to {y_data.max():.2f}")
    
    # Profile the reconstruction function
    with profile_context() as pr:
        start_time = time.time()
        
        result = reconstruct_ligament(x_data, y_data, constraint_manager)
        
        end_time = time.time()
        execution_time = end_time - start_time
    
    print(f"\nExecution time: {execution_time:.4f} seconds")
    print(f"Final loss: {result['loss']:.6f}")
    print(f"Parameters: {result['params']}")
    print(f"Optimization success: {result['opt_result'].success}")
    print(f"Number of iterations: {result['opt_result'].nit}")
    print(f"Number of function evaluations: {result['opt_result'].nfev}")
    print(f"Number of gradient evaluations: {result['opt_result'].njev}")
    print(f"Number of Hessian evaluations: {result['opt_result'].nhev}")
    
    # Analyze profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\n" + "=" * 40)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 40)
    print(s.getvalue())
    
    return result, pr

def profile_complete_model_optimization():
    """Profile the complete model optimization function."""
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE MODEL OPTIMIZATION")
    print("=" * 60)
    
    # Load configurations
    config, constraints_config = load_configs()
    constraint_manager_mcl = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    constraint_manager_lcl = ConstraintManager(constraints_config=constraints_config['blankevoort_lcl'])
    constraint_manager = (constraint_manager_mcl, constraint_manager_lcl)
    
    # Generate test data
    data = generate_knee_test_data(config)
    
    print(f"Test data: {len(data['thetas'])} points")
    print(f"Theta range: {np.degrees(min(data['thetas'])):.2f}° to {np.degrees(max(data['thetas'])):.2f}°")
    print(f"Force range: {min(data['applied_force']):.2f} to {max(data['applied_force']):.2f}")
    
    # Profile the complete model optimization
    with profile_context() as pr:
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
        execution_time = end_time - start_time
    
    print(f"\nExecution time: {execution_time:.4f} seconds")
    print(f"RMSE: {result['rmse']:.2f}")
    print(f"MAE: {result['mae']:.2f}")
    print(f"MCL parameters: {result['mcl_params']}")
    print(f"LCL parameters: {result['lcl_params']}")
    print(f"Optimization success: {result['optimization_result'].success}")
    print(f"Number of iterations: {result['optimization_result'].nit}")
    print(f"Number of function evaluations: {result['optimization_result'].nfev}")
    
    # Analyze profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\n" + "=" * 40)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("=" * 40)
    print(s.getvalue())
    
    return result, pr

def profile_loss_functions():
    """Profile the individual loss functions."""
    print("\n" + "=" * 60)
    print("PROFILING LOSS FUNCTIONS")
    print("=" * 60)
    
    from src.ligament_reconstructor.ligament_optimiser import loss, loss_jac, loss_hess
    
    # Load configurations
    config, constraints_config = load_configs()
    constraint_manager = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
    
    # Generate test data
    x_data, y_data = generate_test_data()
    
    # Create function and parameters
    initial_guess_list = [40, 0.06, 90.0, 0.0]
    function = BlankevoortFunction(initial_guess_list)
    params = np.array(initial_guess_list)
    
    print(f"Testing with {len(x_data)} data points and {len(params)} parameters")
    
    # Profile loss function
    with profile_context() as pr:
        start_time = time.time()
        
        for _ in range(100):  # Run multiple times for better timing
            loss_val = loss(params, x_data, y_data, function, include_reg=True)
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 100
    
    print(f"\nLoss function:")
    print(f"  Average execution time: {execution_time*1000:.4f} ms per call")
    print(f"  Loss value: {loss_val:.6f}")
    
    # Profile Jacobian
    with profile_context() as pr:
        start_time = time.time()
        
        for _ in range(100):
            jac_val = loss_jac(params, x_data, y_data, function, include_reg=True)
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 100
    
    print(f"\nJacobian function:")
    print(f"  Average execution time: {execution_time*1000:.4f} ms per call")
    print(f"  Gradient norm: {np.linalg.norm(jac_val):.6f}")
    
    # Profile Hessian
    with profile_context() as pr:
        start_time = time.time()
        
        for _ in range(50):  # Hessian is more expensive
            hess_val = loss_hess(params, x_data, y_data, function, include_reg=True)
        
        end_time = time.time()
        execution_time = (end_time - start_time) / 50
    
    print(f"\nHessian function:")
    print(f"  Average execution time: {execution_time*1000:.4f} ms per call")
    print(f"  Hessian condition number: {np.linalg.cond(hess_val):.2e}")
    
    return pr

def analyze_memory_usage():
    """Analyze memory usage during optimization."""
    print("\n" + "=" * 60)
    print("MEMORY USAGE ANALYSIS")
    print("=" * 60)
    
    try:
        import psutil
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Load configurations
        config, constraints_config = load_configs()
        constraint_manager = ConstraintManager(constraints_config=constraints_config['blankevoort_mcl'])
        
        # Generate test data
        x_data, y_data = generate_test_data()
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run reconstruction
        result = reconstruct_ligament(x_data, y_data, constraint_manager)
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Peak memory (tracemalloc): {peak / 1024 / 1024:.2f} MB")
        
    except ImportError:
        print("psutil or tracemalloc not available for memory analysis")

def create_performance_summary(individual_result, complete_result, loss_pr):
    """Create a summary of performance metrics."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    # Individual ligament reconstruction metrics
    individual_time = individual_result[1].getstats()[0].totaltime
    individual_calls = individual_result[1].getstats()[0].percall
    
    print(f"Individual Ligament Reconstruction:")
    print(f"  Total time: {individual_time:.4f} seconds")
    print(f"  Time per call: {individual_calls:.6f} seconds")
    print(f"  Final loss: {individual_result[0]['loss']:.6f}")
    print(f"  Iterations: {individual_result[0]['opt_result'].nit}")
    print(f"  Function evaluations: {individual_result[0]['opt_result'].nfev}")
    
    # Complete model optimization metrics
    complete_time = complete_result[1].getstats()[0].totaltime
    complete_calls = complete_result[1].getstats()[0].percall
    
    print(f"\nComplete Model Optimization:")
    print(f"  Total time: {complete_time:.4f} seconds")
    print(f"  Time per call: {complete_calls:.6f} seconds")
    print(f"  RMSE: {complete_result[0]['rmse']:.2f}")
    print(f"  MAE: {complete_result[0]['mae']:.2f}")
    print(f"  Iterations: {complete_result[0]['optimization_result'].nit}")
    print(f"  Function evaluations: {complete_result[0]['optimization_result'].nfev}")
    
    # Save detailed profiling results
    print(f"\nSaving detailed profiling results...")
    
    # Save individual reconstruction profile
    individual_result[1].dump_stats('individual_ligament_profile.prof')
    
    # Save complete model profile
    complete_result[1].dump_stats('complete_model_profile.prof')
    
    # Save loss functions profile
    loss_pr.dump_stats('loss_functions_profile.prof')
    
    print("Profiling results saved to:")
    print("  - individual_ligament_profile.prof")
    print("  - complete_model_profile.prof")
    print("  - loss_functions_profile.prof")
    
    print("\nTo analyze these files later, use:")
    print("  python -m pstats individual_ligament_profile.prof")

def main():
    """Main profiling function."""
    print("LIGAMENT OPTIMISER PROFILING")
    print("=" * 60)
    
    try:
        # Profile individual ligament reconstruction
        individual_result = profile_individual_ligament_reconstruction()
        
        # Profile complete model optimization
        complete_result = profile_complete_model_optimization()
        
        # Profile loss functions
        loss_pr = profile_loss_functions()
        
        # Analyze memory usage
        analyze_memory_usage()
        
        # Create performance summary
        create_performance_summary(individual_result, complete_result, loss_pr)
        
        print("\n" + "=" * 60)
        print("PROFILING COMPLETED SUCCESSFULLY")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during profiling: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

