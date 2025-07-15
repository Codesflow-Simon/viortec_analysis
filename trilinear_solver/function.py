import warnings
import numpy as np

def trilinear_function(x, k_1, k_2, k_3, x_0, x_1, x_2):
    if x < x_0:
        return 0
    elif x < x_1:
        return k_1 * (x - x_0)
    elif x < x_2:
        return k_1 * (x_1 - x_0) + k_2 * (x - x_1)
    else:
        return k_1 * (x_1 - x_0) + k_2 * (x_2 - x_1) + k_3 * (x - x_2)

def trilinear_function_dx(x, k_1, k_2, k_3, x_0, x_1, x_2):
    if x < x_0:
        return 0
    elif x < x_1:
        return k_1
    elif x < x_2:
        return k_2
    else:
        return k_3

def trilinear_function_jac(x, k_1, k_2, k_3, x_0, x_1, x_2):
    if x < x_0:
        return np.array([0,0,0,0,0,0])
    elif x < x_1:
        return np.array([x-x_0,
                        0,
                        0,
                        -k_1,
                        0,
                        0])
    elif x < x_2:
        return np.array([x_1 - x_0,
                        x - x_1, 
                        0,
                        -k_1,
                        k_1-k_2,
                        0])
    else:
        return np.array([x_1 - x_0,
                        x_2 - x_1,
                        x - x_2,
                        -k_1,
                        k_1-k_2,
                        k_2-k_3])

def trilinear_function_hess(x, k_1, k_2, k_3, x_0, x_1, x_2):
    """
    Returns the Hessian matrix (6x6) of second partial derivatives.
    Order of parameters: [k_1, k_2, k_3, x_0, x_1, x_2]
    """
    if x < x_0:
        # All second derivatives are zero when x < x_0
        return np.array([[0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
    elif x < x_1:
        # Region: x_0 <= x < x_1, f(x) = k_1 * (x - x_0)
        return np.array([[0, 0, 0, -1, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [-1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0]])
    elif x < x_2:
        # Region: x_1 <= x < x_2, f(x) = k_1 * (x_1 - x_0) + k_2 * (x - x_1)
        return np.array([[ 0,  0, 0, -1,  1, 0],
                        [ 0,  0, 0,  0, -1, 0],
                        [ 0,  0, 0,  0,  0, 0],
                        [-1,  0, 0,  0,  0, 0],
                        [ 1, -1, 0,  0,  0, 0],
                        [ 0,  0, 0,  0,  0, 0]])
    else:
        # Region: x >= x_2, f(x) = k_1 * (x_1 - x_0) + k_2 * (x_2 - x_1) + k_3 * (x - x_2)
        return np.array([[ 0,  0,  0, -1,  1,  0],
                        [ 0,  0,  0,  0, -1,  1],
                        [ 0,  0,  0,  0,  0, -1],
                        [-1,  0,  0,  0,  0,  0],
                        [ 1, -1,  0,  0,  0,  0],
                        [ 0,  1, -1,  0,  0,  0]])


class TrilinearFunction:
    def __init__(self, k_1: float, k_2: float, k_3: float, x_0: float, x_1: float, x_2: float):
        self.k_1 = k_1
        self.k_2 = k_2  
        self.k_3 = k_3
        self.x_1 = x_1
        self.x_2 = x_2
        self.x_0 = x_0
        self.check_param_constraints()

    def __call__(self, x):
        return trilinear_function(x, self.k_1, self.k_2, self.k_3, self.x_0, self.x_1, self.x_2)

    def get_params(self):
        return self.k_1, self.k_2, self.k_3, self.x_0, self.x_1, self.x_2

    def check_param_constraints(self):
        errors = []
        tol = 1e-10  # Small tolerance for numerical stability
        if self.k_1 < -tol:
            errors.append(f"k_1 must be non-negative (got {self.k_1})")
        if self.k_2 < -tol:
            errors.append(f"k_2 must be non-negative (got {self.k_2})")
        if self.k_3 < -tol:
            errors.append(f"k_3 must be non-negative (got {self.k_3})")
        if self.x_0 < -tol:
            errors.append(f"x_0 must be non-negative (got {self.x_0})")
        if self.x_1 < -tol:
            errors.append(f"x_1 must be non-negative (got {self.x_1})")
        if self.x_2 < -tol:
            errors.append(f"x_2 must be non-negative (got {self.x_2})")
        if self.x_0 > self.x_1 + tol:
            errors.append(f"x_0 must be less than x_1 (got x_0={self.x_0}, x_1={self.x_1})")
        if self.x_1 > self.x_2 + tol:
            errors.append(f"x_1 must be less than x_2 (got x_1={self.x_1}, x_2={self.x_2})")
        if self.k_1 > self.k_2 + tol:
            errors.append(f"k_1 must be less than k_2 (got k_1={self.k_1}, k_2={self.k_2})")
        if self.k_2 > self.k_3 + tol:
            errors.append(f"k_2 must be less than k_3 (got k_2={self.k_2}, k_3={self.k_3})")
            
        if errors:
            warnings.warn("\n".join(errors))
            raise ValueError("\n".join(errors))