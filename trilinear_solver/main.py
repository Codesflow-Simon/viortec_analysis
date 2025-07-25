import numpy as np
from function import TrilinearFunction, BlankevoortFunction, blankevoort_function, blankevoort_function_dx, blankevoort_function_jac
from scipy.optimize import minimize
from plot import generate_plots, plot_hessian, generate_plots_with_uncertainty
from loss import loss, loss_jac, loss_hess
from constraints import constraints
from kkt_covariance import compute_constrained_covariance, compute_lagrangian_hessian, print_covariance_analysis
import matplotlib.pyplot as plt
import os

def solve(x_data, y_data, initial_guess):
    loss_func = lambda params: loss(params, x_data, y_data)
    jac_func = lambda params: loss_jac(params, x_data, y_data)
    # hess_func = lambda params: loss_hess(params, x_data, y_data)

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
    result = minimize(loss_func, initial_guess, method='trust-constr', 
                     jac=jac_func,
                    #  hess=hess_func,
                     constraints=constraints_list)
    
    print("Optimization result:")
    print(f"Success: {result.success}")
    print(f"Optimal parameters: k_1={result.x[0]:.2f}, k_2={result.x[1]:.2f}, k_3={result.x[2]:.2f}, x_0={result.x[3]:.3f}, x_1={result.x[4]:.3f}, x_2={result.x[5]:.3f}")
    print("\nLagrange multipliers:")
    for i, multiplier in enumerate(result.v):
        print(f"λ_{i+1}: {multiplier[0]:.5f}")
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

def get_initial_guess(params):
    initial_guess = params
    # Add 10% noise to initial guess
    noise = np.random.normal(0, 0.2, len(initial_guess))  # 10% noise
    initial_guess = [val * (1 + noise[i]) for i, val in enumerate(initial_guess)]
    # initial_guess = enforce_constraints(initial_guess)
    return initial_guess

if __name__ == "__main__":
    # Problem parameters
    # k_1 = 1
    # k_2 = 2
    # k_3 = 3
    # x_0 = 0
    # x_1 = 1
    # x_2 = 2
    transition_length = 0.06
    k_1 = 2000
    x0 = 0
    # ground_truth = TrilinearFunction(k_1, k_2, k_3, x_0, x_1, x_2)
    ground_truth = BlankevoortFunction(transition_length, k_1, x0)

    # Generate data points
    x_data = np.linspace(-1, 1.5, 100)  # Sample points from before x_0 to after x_2
    y_data = np.array([float(ground_truth(x)) for x in x_data])
    x_noise = np.random.normal(0, 5e-2, len(x_data))
    y_noise = np.random.normal(0, 5e-2, len(y_data))
    x_data = x_data + x_noise
    y_data = y_data + y_noise

    # Solve the optimization problem
    # initial_guess = get_initial_guess([k_1, k_2, k_3, x_0, x_1, x_2])
    initial_guess = get_initial_guess([transition_length, k_1, x0])
    result = solve(x_data, y_data, initial_guess)

    # Can we get the lagragian value, lagrangian deriaative or hessian at the solution?
    # Or perhaps the Lagrange multipliers to allow us to find these?
    
    # Create TrilinearFunction object from optimization result
    # fitted_function = TrilinearFunction(*result.x)
    fitted_function = BlankevoortFunction(*result.x)

    # Compute Lagrangian Hessian and constrained covariance
    
    # Compute Lagrangian Hessian
    hessian = compute_lagrangian_hessian(result.x, x_data, y_data, result.v)
    inverse_hessian = np.linalg.inv(hessian)
    diag_inverse_hessian = np.diag(inverse_hessian)
    std = np.sqrt(diag_inverse_hessian)

    # Print parameters and check if their variance is less than 100
    param_names = ['k_1', 'k_2', 'k_3', 'x_0', 'x_1', 'x_2']
    for i, (name, param, variance) in enumerate(zip(param_names, result.x, diag_inverse_hessian)):
        print(f"{name}: {param:.4f} (variance: {variance:.4f}, Observable: {'Yes' if variance < 1000 else 'No'})")

    # sigma_theta, std_constrained, K_inv, M = compute_constrained_covariance(result, x_data, y_data)
    # print(f"Sigma_theta: {sigma_theta}")
    # print(f"Standard deviations: {std_constrained}")

    # Generate plots
    print("\n=== Generating Plots ===")
    generate_plots(x_data, y_data, fitted_function, ground_truth)
    plot_hessian(hessian)
    generate_plots_with_uncertainty(x_data, y_data, fitted_function, ground_truth, std)
    plt.close('all')
