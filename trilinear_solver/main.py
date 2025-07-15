import numpy as np
from function import TrilinearFunction
from scipy.optimize import minimize
from plot import generate_plots
from loss import loss, loss_jac, loss_hess
from constraints import constraints



def solve(x_data, y_data, initial_guess):
    loss_func = lambda params: loss(params, x_data, y_data)
    jac_func = lambda params: loss_jac(params, x_data, y_data)
    hess_func = lambda params: loss_hess(params, x_data, y_data)

    constraints_list = constraints

    # Available optimizers in scipy.optimize.minimize:
    # 'Nelder-Mead' - Simplex algorithm, derivative-free, works well with discontinuities
    # 'Powell' - Derivative-free, good for discontinuous functions
    # 'CG' - Conjugate gradient, requires continuity
    # 'BFGS' - Quasi-Newton method, requires continuity
    # 'Newton-CG' - Newton's method, requires continuity and twice differentiable
    # 'L-BFGS-B' - Limited memory BFGS with bounds, requires continuity
    # 'TNC' - Truncated Newton, requires continuity
    # 'COBYLA' - Constrained optimization by linear approximation, handles discontinuities
    # 'SLSQP' - Sequential least squares programming, requires continuity
    # 'trust-constr' - Trust region, requires continuity
    
    # For discontinuous functions, best choices are:
    # - Nelder-Mead (but doesn't handle constraints well)
    # - Powell (but doesn't handle constraints well) 
    # - COBYLA (handles constraints)
    
    # Using SLSQP since we have constraints and derivatives
    result = minimize(loss_func, initial_guess, method='Newton-CG', 
                     jac=jac_func,
                     hess=hess_func,
                     constraints=constraints_list)
    
    print("Optimization result:")
    print(f"Success: {result.success}")
    print(f"Optimal parameters: k_1={result.x[0]:.2f}, k_2={result.x[1]:.2f}, k_3={result.x[2]:.2f}, x_0={result.x[3]:.3f}, x_1={result.x[4]:.3f}, x_2={result.x[5]:.3f}")
    print(f"Final loss: {result.fun}")
    
    return result

def enforce_constraints(initial_guess):
    # Enforce constraints on initial guess using max operations
    # k_3 > k_2 > k_1
    initial_guess[2] = max(initial_guess[2], initial_guess[1] + 0.1)  # k_3 > k_2
    initial_guess[1] = max(initial_guess[1], initial_guess[0] + 0.1)  # k_2 > k_1
    
    # x_2 > x_1 > x_0 >= 0
    initial_guess[3] = max(0, initial_guess[3])  # x_0 >= 0
    initial_guess[4] = max(initial_guess[4], initial_guess[3] + 0.1)  # x_1 > x_0
    initial_guess[5] = max(initial_guess[5], initial_guess[4] + 0.1)  # x_2 > x_1
    
    return initial_guess

def get_initial_guess(x_data, y_data):
    initial_guess = [k_1, k_2, k_3, x_0, x_1-0.5, x_2-1.2]
    # Add 10% noise to initial guess
    noise = np.random.normal(0, 0.1, len(initial_guess))  # 10% noise
    initial_guess = [val * (1 + noise[i]) for i, val in enumerate(initial_guess)]
    initial_guess = enforce_constraints(initial_guess)
    return initial_guess

if __name__ == "__main__":
    # Problem parameters
    k_1 = 1
    k_2 = 2
    k_3 = 3
    x_0 = 0
    x_1 = 1
    x_2 = 2
    ground_truth = TrilinearFunction(k_1, k_2, k_3, 0.0, 0.5, 0.6)

    # Generate noise-free data points
    x_data = np.linspace(-1, 4, 100)  # Sample points from before x_0 to after x_2
    y_data = np.array([float(ground_truth(x)) for x in x_data])
    x_noise = np.random.normal(0, 0.05, len(x_data))
    y_noise = np.random.normal(0, 0.05, len(y_data))
    x_data = x_data + x_noise
    y_data = y_data + y_noise

    # Solve the optimization problem
    initial_guess = get_initial_guess(x_data, y_data)
    result = solve(x_data, y_data, initial_guess)
    
    # Create TrilinearFunction object from optimization result
    print(result.x)
    fitted_function = TrilinearFunction(*result.x)
    
    # Generate plots``
    print("\n=== Generating Plots ===")
    plot_files = generate_plots(x_data, y_data, fitted_function, ground_truth)
    
    print(f"\nPlot files saved:")
    for plot_name, file_path in plot_files.items():
        print(f"  {plot_name}: {file_path}")