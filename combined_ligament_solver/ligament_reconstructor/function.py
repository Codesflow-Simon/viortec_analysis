import warnings
import numpy as np

def trilinear_function(x, k_1, k_2, k_3, l_0, a_1, a_2):
    x_1 = a_1 * l_0
    x_2 = a_2 * l_0

    if x < l_0:
        return 0
    elif x < x_1:
        return k_1 * (x - l_0)
    elif x < x_2:
        return k_1 * x_1 + k_2 * (x - x_1)
    else:
        return k_1 * x_1 + k_2 * (x_2 - x_1) + k_3 * (x - x_2)

def trilinear_function_vectorized(x, k_1, k_2, k_3, l_0, a_1, a_2):
    """
    Vectorized version of trilinear_function that works with numpy arrays.
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    
    x_1 = a_1 * l_0
    x_2 = a_2 * l_0
    
    # Region 1: x < l_0
    mask1 = x < l_0
    result[mask1] = 0
    
    # Region 2: l_0 <= x < x_1
    mask2 = (x >= l_0) & (x < x_1)
    result[mask2] = k_1 * (x[mask2] - l_0)
    
    # Region 3: x_1 <= x < x_2
    mask3 = (x >= x_1) & (x < x_2)
    result[mask3] = k_1 * x_1 + k_2 * (x[mask3] - x_1)
    
    # Region 4: x >= x_2
    mask4 = x >= x_2
    result[mask4] = k_1 * x_1 + k_2 * (x_2 - x_1) + k_3 * (x[mask4] - x_2)
    
    return result

def trilinear_function_jac(x, k_1, k_2, k_3, l_0, a_1, a_2):
    x_1 = a_1 * l_0
    x_2 = a_2 * l_0

    if x < l_0:
        return np.array([0,0,0,0,0])
    elif x < x_1:
        return np.array([l_0*(),
                        0,
                        0,
                        0,
                        0])
    elif x < x_2:
        return np.array([x_1,
                        x - x_1, 
                        0,
                        k_1-k_2,
                        0])
    else:
        return np.array([x_1,
                        x_2 - x_1,
                        x - x_2,
                        k_1-k_2,
                        k_2-k_3])

def trilinear_function_jac_vectorized(x, k_1, k_2, k_3, x_1, x_2):
    """
    Vectorized version of trilinear_function_jac that works with numpy arrays.
    Returns a 2D array where each row is the Jacobian for the corresponding x value.
    """
    x = np.asarray(x)
    n = x.size
    result = np.zeros((n, 5))
    
    # Region 1: x < 0
    mask1 = x < 0
    result[mask1] = [0, 0, 0, 0, 0]
    
    # Region 2: 0 <= x < x_1
    mask2 = (x >= 0) & (x < x_1)
    result[mask2] = np.column_stack([x[mask2], np.zeros(np.sum(mask2)), np.zeros(np.sum(mask2)), 
                                   np.zeros(np.sum(mask2)), np.zeros(np.sum(mask2))])
    
    # Region 3: x_1 <= x < x_2
    mask3 = (x >= x_1) & (x < x_2)
    result[mask3] = np.column_stack([np.full(np.sum(mask3), x_1), 
                                   x[mask3] - x_1, 
                                   np.zeros(np.sum(mask3)), 
                                   np.full(np.sum(mask3), k_1 - k_2), 
                                   np.zeros(np.sum(mask3))])
    
    # Region 4: x >= x_2
    mask4 = x >= x_2
    result[mask4] = np.column_stack([np.full(np.sum(mask4), x_1), 
                                   np.full(np.sum(mask4), x_2 - x_1), 
                                   x[mask4] - x_2, 
                                   np.full(np.sum(mask4), k_1 - k_2), 
                                   np.full(np.sum(mask4), k_2 - k_3)])
    
    return result

def trilinear_function_hess(x, k_1, k_2, k_3, x_1, x_2):
    """
    Returns the Hessian matrix (5x5) of second partial derivatives.
    Order of parameters: [k_1, k_2, k_3, x_1, x_2]
    """
    if x < 0:
        # All second derivatives are zero when x < 0
        return np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
    elif x < x_1:
        # Region: 0 <= x < x_1, f(x) = k_1 * x
        return np.array([[0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0]])
    elif x < x_2:
        # Region: x_1 <= x < x_2, f(x) = k_1 * x_1 + k_2 * (x - x_1)
        return np.array([[ 0,  0, 0,  1, 0],
                        [ 0,  0, 0, -1, 0],
                        [ 0,  0, 0,  0, 0],
                        [ 1, -1, 0,  0, 0],
                        [ 0,  0, 0,  0, 0]])
    else:
        # Region: x >= x_2, f(x) = k_1 * x_1 + k_2 * (x_2 - x_1) + k_3 * (x - x_2)
        return np.array([[ 0,  0,  0,  1,  0],
                        [ 0,  0,  0,  -1,  1],
                        [ 0,  0,  0,   0, -1],
                        [ 1, -1,  0,   0,  0],
                        [ 0,  1, -1,   0,  0]])

def trilinear_function_hess_vectorized(x, k_1, k_2, k_3, x_1, x_2):
    """
    Vectorized version of trilinear_function_hess that works with numpy arrays.
    Returns a 3D array where each slice is the Hessian for the corresponding x value.
    """
    x = np.asarray(x)
    n = x.size
    result = np.zeros((n, 5, 5))
    
    # Region 1: x < 0
    mask1 = x < 0
    # Hessian is already zero for all regions
    
    # Region 2: 0 <= x < x_1
    mask2 = (x >= 0) & (x < x_1)
    # Hessian is zero for this region
    
    # Region 3: x_1 <= x < x_2
    mask3 = (x >= x_1) & (x < x_2)
    hess3 = np.array([[ 0,  0, 0,  1, 0],
                     [ 0,  0, 0, -1, 0],
                     [ 0,  0, 0,  0, 0],
                     [ 1, -1, 0,  0, 0],
                     [ 0,  0, 0,  0, 0]])
    for i in np.where(mask3)[0]:
        result[i] = hess3
    
    # Region 4: x >= x_2
    mask4 = x >= x_2
    hess4 = np.array([[ 0,  0,  0,  1,  0],
                     [ 0,  0,  0,  -1,  1],
                     [ 0,  0,  0,   0, -1],
                     [ 1, -1,  0,   0,  0],
                     [ 0,  1, -1,   0,  0]])
    for i in np.where(mask4)[0]:
        result[i] = hess4
    
    return result


class TrilinearFunction:
    def __init__(self, k_1: float, k_2: float, k_3: float, x_1: float, x_2: float):
        self.k_1 = k_1
        self.k_2 = k_2  
        self.k_3 = k_3
        self.x_1 = x_1
        self.x_2 = x_2
        # self.check_param_constraints()

    def __call__(self, x):
        return trilinear_function(x, self.k_1, self.k_2, self.k_3, self.x_1, self.x_2)

    def __call_vectorized__(self, x):
        """Vectorized version of __call__ that works with numpy arrays."""
        return trilinear_function_vectorized(x, self.k_1, self.k_2, self.k_3, self.x_1, self.x_2)

    def get_params(self):
        return {"k_1": self.k_1, 
                "k_2": self.k_2, 
                "k_3": self.k_3, 
                "x_1": self.x_1, 
                "x_2": self.x_2}

    def check_param_constraints(self):
        # Not used for now
        errors = []
        tol = 1e-2  # Small tolerance for numerical stability
        if self.k_1 < -tol:
            errors.append(f"k_1 must be non-negative (got {self.k_1})")
        if self.k_2 < -tol:
            errors.append(f"k_2 must be non-negative (got {self.k_2})")
        if self.k_3 < -tol:
            errors.append(f"k_3 must be non-negative (got {self.k_3})")
        if self.x_1 < -tol:
            errors.append(f"x_1 must be non-negative (got {self.x_1})")
        if self.x_2 < -tol:
            errors.append(f"x_2 must be non-negative (got {self.x_2})")
        if self.x_1 > self.x_2 + tol:
            errors.append(f"x_1 must be less than x_2 (got x_1={self.x_1}, x_2={self.x_2})")
        if self.k_1 > self.k_2 + tol:
            errors.append(f"k_1 must be less than k_2 (got k_1={self.k_1}, k_2={self.k_2})")
        if self.k_2 > self.k_3 + tol:
            errors.append(f"k_2 must be less than k_3 (got k_2={self.k_2}, k_3={self.k_3})")
            
        if errors:
            warnings.warn("\n".join(errors))
            raise ValueError("\n".join(errors))

def blankevoort_function(epsilon, epsilon_t, k):
    """
    Piecewise spring force function:
        F_spring(epsilon) = 0,                        if epsilon < 0
                         = k * epsilon^2 / (2*epsilon_t),  if 0 <= epsilon <= epsilon_t
                         = k * (epsilon - epsilon_t/2),    if epsilon > epsilon_t
    """
    if epsilon < 0:
        return 0
    elif epsilon <= epsilon_t:
        return k * epsilon**2 / (2 * epsilon_t)
    else:
        return k * (epsilon - epsilon_t / 2)

def blankevoort_function_vectorized(epsilon:np.ndarray, epsilon_t:float, k:float):
    """
    Vectorized version of blankevoort_function that works with numpy arrays.
    """
    print(f"epsilon: {epsilon}")
    print(f"epsilon_t: {epsilon_t}")
    print(f"k: {k}")
    epsilon = np.asarray(epsilon)
    result = np.zeros_like(epsilon, dtype=float)
    
    # Region 1: epsilon < 0
    mask1 = epsilon < 0
    result[mask1] = 0
    
    # Region 2: 0 <= epsilon <= epsilon_t
    mask2 = (epsilon >= 0) & (epsilon <= epsilon_t)
    result[mask2] = k * epsilon[mask2]**2 / (2 * epsilon_t)
    
    # Region 3: epsilon > epsilon_t
    mask3 = epsilon > epsilon_t
    result[mask3] = k * (epsilon[mask3] - epsilon_t / 2)
    
    return result

def blankevoort_function_dx(epsilon, epsilon_t, k):
    if epsilon < 0:
        return 0
    elif epsilon < epsilon_t:
        return k * epsilon / epsilon_t
    else:
        return k

def blankevoort_function_dx_vectorized(epsilon:np.ndarray, epsilon_t:float, k:float):
    """
    Vectorized version of blankevoort_function_dx that works with numpy arrays.
    """

    epsilon = np.asarray(epsilon)
    result = np.zeros_like(epsilon, dtype=float)
    
    # Region 1: epsilon < 0
    mask1 = epsilon < 0
    result[mask1] = 0
    
    # Region 2: 0 <= epsilon < epsilon_t
    mask2 = (epsilon >= 0) & (epsilon < epsilon_t)
    result[mask2] = k * epsilon[mask2] / epsilon_t
    
    # Region 3: epsilon >= epsilon_t
    mask3 = epsilon >= epsilon_t
    result[mask3] = k
    
    return result

def blankevoort_function_jac(epsilon:np.ndarray, epsilon_t:float, k:float):
    """Jacobian of blankevoort_spring_force with respect to [epsilon_t, k] (no epsilon)."""
    if epsilon < 0:
        return np.array([0, 0])
    elif epsilon < epsilon_t:
        d_epsilon_t = -k * epsilon**2 / (2 * epsilon_t**2)
        d_k = epsilon**2 / (2 * epsilon_t)
        return np.array([d_epsilon_t, d_k])
    else:
        d_epsilon_t = -k / 2
        d_k = epsilon - epsilon_t / 2
        return np.array([d_epsilon_t, d_k])

def blankevoort_function_jac_vectorized(epsilon, epsilon_t, k):
    """
    Vectorized version of blankevoort_function_jac that works with numpy arrays.
    Returns a 2D array where each row is the Jacobian for the corresponding epsilon value.
    """
    epsilon = np.asarray(epsilon)
    n = epsilon.size
    result = np.zeros((n, 2))
    
    # Region 1: epsilon < 0
    mask1 = epsilon < 0
    result[mask1] = [0, 0]
    
    # Region 2: 0 <= epsilon < epsilon_t
    mask2 = (epsilon >= 0) & (epsilon < epsilon_t)
    d_epsilon_t = -k * epsilon[mask2]**2 / (2 * epsilon_t**2)
    d_k = epsilon[mask2]**2 / (2 * epsilon_t)
    result[mask2] = np.column_stack([d_epsilon_t, d_k])
    
    # Region 3: epsilon >= epsilon_t
    mask3 = epsilon >= epsilon_t
    result[mask3] = np.column_stack([np.full(np.sum(mask3), -k / 2), 
                                   epsilon[mask3] - epsilon_t / 2])
    
    return result

def blankevoort_function_hess(epsilon, epsilon_t, k):
    if epsilon < 0:
        return np.array([[0, 0], [0, 0]])
    elif epsilon < epsilon_t:
        return np.array([[k*epsilon**2/(epsilon_t**3), -epsilon**2/(2*epsilon_t**2)], 
                        [-epsilon**2/(2*epsilon_t**2), 0]])
    else:
        return np.array([[0, -1/2], [-1/2, 0]])

def blankevoort_function_hess_vectorized(epsilon, epsilon_t, k):
    """
    Vectorized version of blankevoort_function_hess that works with numpy arrays.
    Returns a 3D array where each slice is the Hessian for the corresponding epsilon value.
    """
    epsilon = np.asarray(epsilon)
    n = epsilon.size
    result = np.zeros((n, 2, 2))
    
    # Region 1: epsilon < 0
    mask1 = epsilon < 0
    # Hessian is already zero
    
    # Region 2: 0 <= epsilon < epsilon_t
    mask2 = (epsilon >= 0) & (epsilon < epsilon_t)
    for i in np.where(mask2)[0]:
        eps = epsilon[i]
        result[i] = np.array([[k*eps**2/(epsilon_t**3), -eps**2/(2*epsilon_t**2)], 
                             [-eps**2/(2*epsilon_t**2), 0]])
    
    # Region 3: epsilon >= epsilon_t
    mask3 = epsilon >= epsilon_t
    hess3 = np.array([[0, -1/2], [-1/2, 0]])
    for i in np.where(mask3)[0]:
        result[i] = hess3
    
    return result

class BlankevoortFunction:
    def __init__(self, epsilon_t, k):
        self.epsilon_t = epsilon_t
        self.k = k

    def __call__(self, epsilon):
        return blankevoort_function(epsilon, self.epsilon_t, self.k)


    def __call_vectorized__(self, epsilon):
        """Vectorized version of __call__ that works with numpy arrays."""
        return blankevoort_function_vectorized(epsilon, self.epsilon_t, self.k)

    def get_params(self):
        return {"e_t": self.epsilon_t, 
                "k": self.k}

    def check_param_constraints(self):
        errors = []
        tol = 1e-6  # Small tolerance for numerical stability
        if self.epsilon_t < -tol:
            errors.append(f"epsilon_t must be non-negative (got {self.epsilon_t})")
        if self.k < -tol:
            errors.append(f"k must be non-negative (got {self.k})")
        
        if errors:
            warnings.warn("\n".join(errors))
            raise ValueError("\n".join(errors))
        
        