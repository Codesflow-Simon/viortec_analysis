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
        tol = 1e-6  # Small tolerance for numerical stability
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

def blankevoort_function(x, transition_length, k_1, x0):
    """
    A spring that has a transition length, and a different spring constant for each transition length.
    Consider the spring strain, x, which is zero at rest length x0.
    When x < x0, the spring force is 0.
    When x0 < x < x0 + transition_length, the spring force is k_1 * x^2 / (2 * transition_length).
    When x > x0 + transition_length, the spring force is k_1 * (x - x0 - transition_length/2).
    """
    if x < x0:
        return 0
    elif x < x0 + transition_length:
        return k_1 * x**2 / (2 * transition_length)
    else:
        return k_1 * (x - x0 - transition_length/2)

def blankevoort_function_dx(x, transition_length, k_1, x0):
    if x < x0:
        return 0
    elif x < x0 + transition_length:
        return k_1 * x / transition_length
    else:
        return k_1

def blankevoort_function_jac(x, transition_length, k_1, x0):
    """ Jacobian of blankevoort_function
    
    """
    if x < x0:
        return np.array([0, 0, 0])
    elif x < x0 + transition_length:
        d_transition_length = -k_1 * x**2 / (2 * transition_length**2)
        d_k1 = x**2 / (2 * transition_length)
        d_x0 = 0
        return np.array([d_transition_length, d_k1, d_x0])
    else:
        d_transition_length = -k_1 / 2
        d_k1 = x - x0 - transition_length / 2
        d_x0 = -k_1
        return np.array([d_transition_length, d_k1, d_x0])

class BlankevoortFunction:
    def __init__(self, transition_length, k_1, x0):
        self.transition_length = transition_length
        self.k_1 = k_1
        self.x0 = x0

    def __call__(self, x):
        return blankevoort_function(x, self.transition_length, self.k_1, self.x0)
        
        