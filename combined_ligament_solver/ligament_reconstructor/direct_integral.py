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
        # Transform parameters back to constrained space for function evaluation
        params = inverse_constraint_transform(params, self.constraint_manager)
        
        # Use vectorized function evaluation
        if params.ndim == 1:
            params_array = params.reshape(1, -1)
        else:
            params_array = params
            
        y_pred_matrix = func.vectorized_function(x_data, params_array)
        
        # For single parameter set, extract the result
        if params.ndim == 1:
            y_pred = y_pred_matrix[0]
        else:
            y_pred = y_pred_matrix
        
        # Compute residuals
        residuals = y_data - y_pred
        
        # Log-likelihood (assuming Gaussian noise)
        log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
        
        return log_like

    def clear_cache(self):
        """Clear the normalization coefficient cache."""
        self._last_params = None
        self._last_integral_value = None

    def integrate(self, params: np.ndarray, x_data: np.ndarray, y_data: np.ndarray, func: Any, sigma_noise: float, grid_size: int = 50):
        """
        Integrate the log-likelihood function using grid-based integration for better performance.
        
        This method computes the normalization coefficient by integrating over l_0 and f_ref
        while keeping k and alpha fixed. Uses a pre-computed grid for vectorized evaluation.
        
        Args:
            params: Parameter vector [k, alpha, l_0, f_ref] (k and alpha are kept fixed)
            x_data: Input data points
            y_data: Target data points  
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            grid_size: Number of grid points for each dimension
            
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
        
        # Create grid for l_0 and f_ref
        l_0_grid = np.linspace(l_0_bounds[0], l_0_bounds[1], grid_size)
        f_ref_grid = np.linspace(f_ref_bounds[0], f_ref_bounds[1], grid_size)
        
        # Create meshgrid for all combinations
        l_0_mesh, f_ref_mesh = np.meshgrid(l_0_grid, f_ref_grid, indexing='ij')
        
        # Flatten the grids for vectorized computation
        l_0_flat = l_0_mesh.flatten()
        f_ref_flat = f_ref_mesh.flatten()
        
        # Create parameter array for all grid points
        params_grid = np.column_stack([
            np.full(len(l_0_flat), k_fixed),
            np.full(len(l_0_flat), alpha_fixed),
            l_0_flat,
            f_ref_flat
        ])
        
        # Transform all parameters to unconstrained space
        params_unconstrained_grid = np.array([
            constraint_transform(params_grid[i], self.constraint_manager) 
            for i in range(len(params_grid))
        ])
        
        # Compute log-likelihood for all grid points using vectorized evaluation
        log_likes = np.array([
            self.log_likelihood(params_unconstrained_grid[i], x_data, y_data, func, sigma_noise)
            for i in range(len(params_unconstrained_grid))
        ])
        
        # Convert to likelihood values
        likelihoods = np.exp(log_likes)
        
        # Reshape back to grid
        likelihood_grid = likelihoods.reshape(grid_size, grid_size)
        
        # Compute integral using trapezoidal rule
        # Calculate grid spacing
        dl_0 = (l_0_bounds[1] - l_0_bounds[0]) / (grid_size - 1)
        df_ref = (f_ref_bounds[1] - f_ref_bounds[0]) / (grid_size - 1)
        
        # Use scipy's simpson for more accurate 2D integration
        integral_value = integrate.simpson(
            integrate.simpson(likelihood_grid, f_ref_grid, axis=1), 
            l_0_grid, axis=0
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

    def calculate_marginal_probability(self, params: np.ndarray, marginal_param: int, marginal_param_value: float, x_data: np.ndarray, y_data: np.ndarray, func: Any, sigma_noise: float, grid_size: int = 20):
        """
        Calculate the marginal probability of a parameter using grid-based integration.
        """
        # Get bounds for all parameters
        constraints_list = self.constraint_manager.get_constraints_list()
        
        # Keep the parameter of interest fixed
        fixed_param_val = marginal_param_value
        
        # Get bounds for parameters to integrate over (excluding the marginal parameter)
        integration_bounds = []
        for i in range(len(params)):
            if i != marginal_param:
                integration_bounds.append(constraints_list[i])
        
        # Create grids for each parameter to integrate over
        param_grids = []
        for bounds in integration_bounds:
            param_grids.append(np.linspace(bounds[0], bounds[1], grid_size))
        
        # Create meshgrid for all combinations
        if len(param_grids) == 1:
            # 1D case
            param_meshes = [param_grids[0]]
            param_flat = param_meshes[0]
        elif len(param_grids) == 2:
            # 2D case
            param_meshes = np.meshgrid(param_grids[0], param_grids[1], indexing='ij')
            param_flat = np.column_stack([param_meshes[0].flatten(), param_meshes[1].flatten()])
        elif len(param_grids) == 3:
            # 3D case
            param_meshes = np.meshgrid(param_grids[0], param_grids[1], param_grids[2], indexing='ij')
            param_flat = np.column_stack([
                param_meshes[0].flatten(), 
                param_meshes[1].flatten(), 
                param_meshes[2].flatten()
            ])
        else:
            raise ValueError(f"Integration over {len(param_grids)} dimensions not implemented")
        
        # Create full parameter arrays for all grid points
        params_grid = []
        for i in range(len(param_flat)):
            param_vals = []
            param_idx = 0
            for j in range(len(params)):
                if j == marginal_param:
                    param_vals.append(fixed_param_val)
                else:
                    param_vals.append(param_flat[i, param_idx])
                    param_idx += 1
            params_grid.append(np.array(param_vals))
        
        # Transform all parameters to unconstrained space
        params_unconstrained_grid = np.array([
            constraint_transform(params_grid[i], self.constraint_manager) 
            for i in range(len(params_grid))
        ])

        # Compute log-likelihood for all grid points using vectorized evaluation
        log_likes = np.array([
            self.log_likelihood(params_unconstrained_grid[i], x_data, y_data, func, sigma_noise)
            for i in range(len(params_unconstrained_grid))
        ])
        
        # Convert to likelihood values
        likelihoods = np.exp(log_likes)
        
        # Reshape back to grid for integration
        if len(param_grids) == 1:
            likelihood_grid = likelihoods
        elif len(param_grids) == 2:
            likelihood_grid = likelihoods.reshape(grid_size, grid_size)
        elif len(param_grids) == 3:
            likelihood_grid = likelihoods.reshape(grid_size, grid_size, grid_size)
        
        # Compute integral using Simpson's rule for higher accuracy
        if len(param_grids) == 1:
            integral_value = integrate.simpson(likelihood_grid, param_grids[0])
        elif len(param_grids) == 2:
            integral_value = integrate.simpson(
                integrate.simpson(likelihood_grid, param_grids[1], axis=1), 
                param_grids[0], axis=0
            )
        elif len(param_grids) == 3:
            integral_value = integrate.simpson(
                integrate.simpson(
                    integrate.simpson(likelihood_grid, param_grids[2], axis=2), 
                    param_grids[1], axis=1
                ), 
                param_grids[0], axis=0
            )
        
        return integral_value