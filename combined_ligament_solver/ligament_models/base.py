import warnings
import numpy as np
from sympy import symbols, Piecewise, diff, Matrix, lambdify

class LigamentFunction:
    def __init__(self, params: np.ndarray):
        self.set_params(params)
        # Cache for symbolic expressions and compiled functions
        self._cached_expr = None
        self._cached_dx_func = None
        self._cached_d2x2_func = None
        self._cached_jac_funcs = None
        self._cached_hess_funcs = None
        self._cached_func = None
        
        # Pre-compute all symbolic expressions and compile functions
        self._setup_cached_functions()

    def _setup_cached_functions(self):
        """
        Pre-compute all symbolic expressions and compile them to functions.
        This is done once during initialization for maximum performance.
        """
        # Get the symbolic expression
        expr = self.sympy_implementation()
        self._cached_expr = expr
        
        # Get parameter symbols
        param_symbols = self.get_param_symbols()
        x_sym = symbols('x')
        
        # Compile the main function
        self._cached_func = lambdify([x_sym] + list(param_symbols), expr, 'numpy')
        
        # Compile first derivative with respect to x
        dx_expr = diff(expr, x_sym)
        self._cached_dx_func = lambdify([x_sym] + list(param_symbols), dx_expr, 'numpy')
        
        # Compile second derivative with respect to x
        d2x_expr = diff(expr, x_sym, 2)
        self._cached_d2x2_func = lambdify([x_sym] + list(param_symbols), d2x_expr, 'numpy')
        
        # Compile Jacobian functions (partial derivatives with respect to parameters)
        jac_exprs = [diff(expr, param) for param in param_symbols]
        self._cached_jac_funcs = [lambdify([x_sym] + list(param_symbols), jac_expr, 'numpy') 
                                 for jac_expr in jac_exprs]
        
        # Compile Hessian functions (second partial derivatives)
        hess_exprs = []
        for param1 in param_symbols:
            for param2 in param_symbols:
                hess_exprs.append(diff(expr, param1, param2))
        self._cached_hess_funcs = [lambdify([x_sym] + list(param_symbols), hess_expr, 'numpy') 
                                  for hess_expr in hess_exprs]

    def get_params(self):
        return self.params

    def set_params(self, params):
        if isinstance(params, dict):
            self.params = np.array(list(params.values()))
        else:
            self.params = params

    def __call__(self, x: np.ndarray, use_cached_function: bool = True):
        if use_cached_function:
            result = self.function(x, self.params)
        else:
            result = self.sympy_implementation()
            result = result.subs(x, x)
        return result

    def sympy_implementation(self):
        """
        Abstract method to be implemented by subclasses.
        Should return the symbolic expression for the function.
        """
        raise NotImplementedError("Subclasses must implement sympy_implementation")

    def dx(self, x: np.ndarray):
        """
        Returns the derivative with respect to x using cached function.
        """
        return self._cached_dx_func(x, *self.params)

    def d2x2(self, x: np.ndarray):
        """
        Returns the second derivative with respect to x using cached function.
        """
        return self._cached_d2x2_func(x, *self.params)

    def jac(self, x: np.ndarray):
        """
        Returns the Jacobian with respect to parameters using cached functions.
        Shape: (n_params, n_points) for array input, (n_params,) for scalar input
        """
        if np.isscalar(x):
            return np.array([func(x, *self.params) for func in self._cached_jac_funcs])
        else:
            # Ensure all functions return arrays of the same shape
            jac_results = []
            for func in self._cached_jac_funcs:
                result = func(x, *self.params)
                # Convert scalar to array if necessary
                if np.isscalar(result):
                    result = np.full_like(x, result)
                # Ensure result is 1D array and has the same length as x
                result = np.asarray(result).flatten()
                if len(result) != len(x):
                    # If the result has different length, broadcast it
                    if len(result) == 1:
                        result = np.full(len(x), result[0])
                    else:
                        # This shouldn't happen, but handle it gracefully
                        raise ValueError(f"Jacobian result has unexpected length: {len(result)}, expected {len(x)}")
                jac_results.append(result)
            # Return shape (n_params, n_points)
            return np.array(jac_results)

    def hess(self, x: np.ndarray):
        """
        Returns the Hessian with respect to parameters using cached functions.
        """
        param_symbols = self.get_param_symbols()
        n_params = len(param_symbols)
        
        if np.isscalar(x):
            hess_values = [func(x, *self.params) for func in self._cached_hess_funcs]
            return np.array(hess_values).reshape(n_params, n_params)
        else:
            # Ensure all functions return arrays of the same shape
            hess_values = []
            for func in self._cached_hess_funcs:
                result = func(x, *self.params)
                # Convert scalar to array if necessary
                if np.isscalar(result):
                    result = np.full_like(x, result)
                # Ensure result is 1D array and has the same length as x
                result = np.asarray(result).flatten()
                if len(result) != len(x):
                    # If the result has different length, broadcast it
                    if len(result) == 1:
                        result = np.full(len(x), result[0])
                    else:
                        # This shouldn't happen, but handle it gracefully
                        raise ValueError(f"Hessian result has unexpected length: {len(result)}, expected {len(x)}")
                hess_values.append(result)
            # Return shape (n_points, n_params, n_params)
            return np.array(hess_values).reshape(n_params, n_params, len(x)).transpose(2, 0, 1)

    def get_param_symbols(self):
        """
        Abstract method to be implemented by subclasses.
        Should return the symbolic parameter names.
        """
        raise NotImplementedError("Subclasses must implement get_param_symbols")

    def function(self, x: np.ndarray, params: np.ndarray):
        """
        Evaluates the function using cached compiled function.
        """
        return self._cached_func(x, *params)

    # Vectorized methods for optimization
    def vectorized_function(self, x: np.ndarray, params_array: np.ndarray):
        """
        Vectorized function evaluation for multiple parameter sets.
        
        Args:
            x: Input points, shape (n_points,)
            params_array: Parameter sets, shape (n_param_sets, n_params)
            
        Returns:
            Function values, shape (n_param_sets, n_points)
        """
        # Use numpy's apply_along_axis for better performance than Python loops

        def eval_single_params(params):
            return self._cached_func(x, *params)
        
        return np.apply_along_axis(eval_single_params, 1, params_array)

    def vectorized_jacobian(self, x: np.ndarray, params_array: np.ndarray):
        """
        Vectorized Jacobian computation for multiple parameter sets.
        
        Args:
            x: Input points, shape (n_points,)
            params_array: Parameter sets, shape (n_param_sets, n_params)
            
        Returns:
            Jacobians, shape (n_param_sets, n_params, n_points) 
        """
        n_param_sets = params_array.shape[0]
        n_params = params_array.shape[1]
        n_points = len(x)
        results = np.zeros((n_param_sets, n_params, n_points))
        
        # Use numpy's apply_along_axis for better performance
        def eval_jac_single_params(params):
            jac_row = np.zeros((n_params, n_points))
            for j, func in enumerate(self._cached_jac_funcs):
                result = func(x, *params)
                if np.isscalar(result):
                    result = np.full(n_points, result)
                result = np.asarray(result).flatten()
                if len(result) != n_points:
                    if len(result) == 1:
                        result = np.full(n_points, result[0])
                    else:
                        raise ValueError(f"Jacobian result has unexpected length: {len(result)}, expected {n_points}")
                jac_row[j] = result
            return jac_row
        
        # Apply to each parameter set
        for i in range(n_param_sets):
            results[i] = eval_jac_single_params(params_array[i])
                
        return results

    def vectorized_hessian(self, x: np.ndarray, params_array: np.ndarray):
        """
        Vectorized Hessian computation for multiple parameter sets.
        
        Args:
            x: Input points, shape (n_points,)
            params_array: Parameter sets, shape (n_param_sets, n_params)
            
        Returns:
            Hessians, shape (n_param_sets, n_points, n_params, n_params)
        """
        n_param_sets = params_array.shape[0]
        n_params = params_array.shape[1]
        n_points = len(x)
        results = np.zeros((n_param_sets, n_points, n_params, n_params))
        
        # Use numpy's apply_along_axis for better performance
        def eval_hess_single_params(params):
            hess_values = []
            for func in self._cached_hess_funcs:
                result = func(x, *params)
                if np.isscalar(result):
                    result = np.full(n_points, result)
                result = np.asarray(result).flatten()
                if len(result) != n_points:
                    if len(result) == 1:
                        result = np.full(n_points, result[0])
                    else:
                        raise ValueError(f"Hessian result has unexpected length: {len(result)}, expected {n_points}")
                hess_values.append(result)
            
            # Reshape to (n_points, n_params, n_params)
            hess_matrix = np.array(hess_values).reshape(n_params, n_params, n_points).transpose(2, 0, 1)
            return hess_matrix
        
        # Apply to each parameter set
        for i in range(n_param_sets):
            results[i] = eval_hess_single_params(params_array[i])
            
        return results