from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import constraint_transform, inverse_constraint_transform
import emcee
from copy import deepcopy
from ligament_models.transformations import inverse_constraint_transform, constraint_transform
from scipy import integrate

class IntegralProbability():
    def __init__(self, constraint_manager: ConstraintManager):
        self.constraint_manager = constraint_manager
        # Simple cache for normalization coefficient
        self._last_params = None
        self._last_integral_value = None

    def log_likelihood(self, params: np.ndarray, x_data: np.ndarray, 
                y_data: np.ndarray, func: Any, sigma_noise: float = 100) -> float:
        """
        Compute log-likelihood function.
        
        Args:
            params: Parameter vector (in unconstrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            log_likelihood: Log-likelihood value
        """
        func = deepcopy(func)

        # Transform parameters back to constrained space for function evaluation
        params = inverse_constraint_transform(params, self.constraint_manager)
        
        func.set_params(params)
        y_pred = np.array([func(x) for x in x_data])
        
        # Compute residuals
        residuals = y_data - y_pred
        
        # Log-likelihood (assuming Gaussian noise)
        log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
        
        return log_like

    def clear_cache(self):
        """Clear the normalization coefficient cache."""
        self._last_params = None
        self._last_integral_value = None

    def integrate(self, params: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, func: Any, sigma_noise: float):
        """
        Integrate the log-likelihood function using SciPy's efficient numerical integration.
        
        This method computes the normalization coefficient by integrating over l_0 and f_ref
        while keeping k and alpha fixed. Uses SciPy's dblquad for 2D integration.
        
        Args:
            params: Parameter vector [k, alpha, l_0, f_ref] (k and alpha are kept fixed)
            x_data: Input data points
            y_data: Target data points  
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            normalization_coefficient: The computed normalization coefficient
        """
        # Keep the parameters k [0] and alpha [1] fixed and integrate over l_0 [2] and f_ref [3]
        # Use the constraint manager to get the bounds for l_0 and f_ref
        constraints_list = self.constraint_manager.get_constraints_list()
        l_0_bounds = constraints_list[2]  # l_0 bounds
        f_ref_bounds = constraints_list[3]  # f_ref bounds
        
        # Extract fixed parameters (only k and alpha matter for caching)
        k_fixed = params[0]
        alpha_fixed = params[1]
        
        # Check if we can use cached result
        if (self._last_params is not None and 
            np.array_equal(self._last_params, np.array([k_fixed, alpha_fixed]))):
            return self._last_integral_value
        
        # Define the integrand function (likelihood function)
        def integrand(f_ref, l_0):
            """
            Integrand function for 2D integration.
            Note: dblquad expects the order (y, x) for the integrand function.
            """
            params_val = np.array([k_fixed, alpha_fixed, l_0, f_ref])
            params_unconstrained = constraint_transform(params_val, self.constraint_manager)
            log_like = self.log_likelihood(params_unconstrained, x_data, y_data, func, sigma_noise)
            return np.exp(log_like)
        
        # Use SciPy's dblquad for efficient 2D integration
        integral_value, error_estimate = integrate.dblquad(
            integrand, 
            l_0_bounds[0], l_0_bounds[1],  # x bounds (l_0)
            lambda x: f_ref_bounds[0], lambda x: f_ref_bounds[1]  # y bounds (f_ref)
        )
        
        # Cache the result
        self._last_params = np.array([k_fixed, alpha_fixed])
        self._last_integral_value = integral_value
        
        return integral_value

    def probability_density(self, params: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, func: Any, sigma_noise: float):
        """
        Compute the probability density function.
        """
        normalization_coefficient = self.integrate(params, x_data, y_data, func, sigma_noise)
        return np.exp(self.log_likelihood(params, x_data, y_data, func, sigma_noise)) / normalization_coefficient