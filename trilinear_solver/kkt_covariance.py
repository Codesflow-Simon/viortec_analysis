import numpy as np
from loss import loss_hess
from constraints import constraints
from function import trilinear_function

def compute_constraint_jacobian(params):
    """
    Compute the constraint Jacobian matrix A ∈ ℝ^{m×p}
    where m is the number of constraints and p is the number of parameters.
    
    Args:
        params: Parameter vector [k_1, k_2, k_3, x_0, x_1, x_2]
    
    Returns:
        A: Constraint Jacobian matrix of shape (m, p)
    """
    p = len(params)
    m = len(constraints)
    A = np.zeros((m, p))
    
    # Finite difference to compute constraint gradients
    eps = 1e-8
    for i, constraint in enumerate(constraints):
        constraint_func = constraint['fun']
        base_value = constraint_func(params)
        
        for j in range(p):
            params_plus = params.copy()
            params_plus[j] += eps
            value_plus = constraint_func(params_plus)
            
            A[i, j] = (value_plus - base_value) / eps
    
    return A

def compute_constraint_hessians(params):
    """
    Compute the Hessians of each constraint function.
    Since all constraints are linear, their Hessians are zero matrices.
    
    Args:
        params: Parameter vector [k_1, k_2, k_3, x_0, x_1, x_2]
    
    Returns:
        C_list: List of m zero matrices, each of shape (p, p)
    """
    p = len(params)
    m = len(constraints)
    
    # All constraints are linear, so their Hessians are zero
    C_list = [np.zeros((p, p)) for _ in range(m)]
    
    return C_list

def compute_lagrangian_hessian(params, x_data, y_data, lagrange_multipliers):
    """
    Compute the Lagrangian Hessian H_L = H_data + sum(λ_j * C_j)
    
    Args:
        params: Optimal parameter vector
        x_data: Input data points
        y_data: Target data points
        lagrange_multipliers: Vector of Lagrange multipliers λ
    
    Returns:
        H_L: Lagrangian Hessian matrix of shape (p, p)
    """
    # Compute data Hessian
    H_data = loss_hess(params, x_data, y_data, include_reg=True)
    
    # Compute constraint Hessians
    C_list = compute_constraint_hessians(params)
    
    # Build Lagrangian Hessian
    H_L = H_data.copy()
    for j, lambda_j in enumerate(lagrange_multipliers):
        H_L += lambda_j * C_list[j]
    
    return H_L

def compute_kkt_matrix_inverse(params, x_data, y_data, lagrange_multipliers):
    """
    Compute the inverse of the KKT matrix and extract the parameter covariance.
    
    Args:
        params: Optimal parameter vector Θ*
        x_data: Input data points
        y_data: Target data points
        lagrange_multipliers: Vector of Lagrange multipliers λ
    
    Returns:
        Sigma_Theta: Estimated covariance matrix of Θ* (p×p)
        K_inv: Full KKT matrix inverse
        M: Upper-left p×p block of K_inv
    """
    p = len(params)
    m = len(constraints)
    n = len(x_data)
    
    # 1. Compute data Hessian
    H_data = loss_hess(params, x_data, y_data, include_reg=True)
    
    # 2. Compute constraint Jacobian
    A = compute_constraint_jacobian(params)
    
    # 3. Load Lagrange multipliers
    lambda_vec = lagrange_multipliers
    
    # 4. Compute constraint Hessians (all zero for linear constraints)
    C_list = compute_constraint_hessians(params)
    
    # 5. Build Lagrangian Hessian H_L
    H_L = H_data.copy()
    for j, lambda_j in enumerate(lambda_vec):
        H_L += lambda_j * C_list[j]
    
    # 6. Assemble the KKT matrix K of size (p+m)×(p+m):
    # K = [ H_L      A^T   ]
    #     [ A        0_m×m ]
    K = np.block([[H_L, A.T],
                  [A, np.zeros((m, m))]])
    
    # 7. Factor or invert K
    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        # If K is singular, use pseudo-inverse
        K_inv = np.linalg.pinv(K)
    
    # 8. Extract upper-left p×p block M from K_inv
    M = K_inv[:p, :p]
    
    # 9. Estimate noise variance
    y_pred = np.array([trilinear_function(x, *params) for x in x_data])
    residuals = y_pred - y_data
    sigma2_noise = np.sum(residuals**2) / (n - p + m)  # degrees-of-freedom correction
    
    # 10. Form parameter covariance
    Sigma_Theta = sigma2_noise * M
    
    return Sigma_Theta, K_inv, M

def compute_constrained_covariance(result, x_data, y_data):
    """
    Main function to compute the constrained covariance matrix.
    
    Args:
        result: Optimization result from scipy.optimize.minimize
        x_data: Input data points
        y_data: Target data points
    
    Returns:
        Sigma_Theta: Estimated covariance matrix of optimal parameters
        std_params: Standard deviations of parameters
        K_inv: Full KKT matrix inverse
        M: Upper-left block of K_inv
    """
    params = result.x
    lagrange_multipliers = result.v
    
    # Compute constrained covariance
    Sigma_Theta, K_inv, M = compute_kkt_matrix_inverse(
        params, x_data, y_data, lagrange_multipliers
    )
    
    # Extract standard deviations
    std_params = np.sqrt(np.diag(np.linalg.inv(Sigma_Theta)))
    
    return Sigma_Theta, std_params, K_inv, M

def print_covariance_analysis(Sigma_Theta, std_params, result):
    """
    Print detailed covariance analysis.
    
    Args:
        Sigma_Theta: Parameter covariance matrix
        std_params: Parameter standard deviations
        result: Optimization result
    """
    param_names = ['k_1', 'k_2', 'k_3', 'x_0', 'x_1', 'x_2']
    
    print("\n=== Constrained Covariance Analysis ===")
    print(f"Parameter estimates: {dict(zip(param_names, result.x))}")
    print(f"Standard deviations: {dict(zip(param_names, std_params))}")
    print(f"Lagrange multipliers: {result.v}")
    
    print("\nCorrelation matrix:")
    corr_matrix = np.zeros_like(Sigma_Theta)
    for i in range(len(Sigma_Theta)):
        for j in range(len(Sigma_Theta)):
            if std_params[i] > 0 and std_params[j] > 0:
                corr_matrix[i, j] = Sigma_Theta[i, j] / (std_params[i] * std_params[j])
            else:
                corr_matrix[i, j] = 0
    
    for i, name_i in enumerate(param_names):
        for j, name_j in enumerate(param_names):
            if i <= j:
                print(f"Corr({name_i}, {name_j}): {corr_matrix[i, j]:.4f}")
    
    print(f"\nCondition number of KKT matrix: {np.linalg.cond(Sigma_Theta):.2e}")
    print(f"Determinant of covariance matrix: {np.linalg.det(Sigma_Theta):.2e}") 