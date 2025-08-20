import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv, cholesky
from typing import Dict, Tuple, Any
from .base_sampler import BaseSampler


class LaplaceSampler(BaseSampler):
    """
    Laplace Approximation sampler for Bayesian inference.
    
    This method approximates the posterior distribution with a Gaussian
    centered at the MAP estimate, using the Hessian of the log-posterior
    to determine the covariance structure.
    """
    
    def __init__(self, constraint_manager=None):
        """
        Initialize Laplace sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
        """
        super().__init__(constraint_manager)
    
    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, 
               n_samples=1000, random_state=None, loss_hessian=None, **kwargs):
        """
        Generate samples using Laplace approximation.
        
        Args:
            map_params: MAP estimate (optimal parameters in constrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            n_samples: Number of samples to generate
            random_state: Random state for reproducibility
            loss_hessian: Pre-computed loss Hessian (optional, will compute if not provided)
            **kwargs: Additional parameters
            
        Returns:
            cov_matrix: Covariance matrix (in constrained space)
            std_params: Standard deviations of parameters (in constrained space)
            samples: Parameter samples (in constrained space)
            acceptance_rate: Always 1.0 for Laplace approximation
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Convert MAP parameters to array
        map_params_array = np.array(list(map_params.values()), dtype=float)
        
        # Transform to unconstrained space for optimization
        if self.constraint_manager is not None:
            from ligament_models.transformations import constraint_transform
            map_params_unconstrained = constraint_transform(map_params_array, self.constraint_manager)
        else:
            map_params_unconstrained = map_params_array
        
        # Compute Hessian at MAP estimate
        if loss_hessian is not None:
            # Use pre-computed loss Hessian
            # The log-likelihood Hessian is -1/(sigma_noise^2) times the loss Hessian
            hessian = -loss_hessian / (sigma_noise**2)
        else:
            # Fall back to finite differences computation
            hessian = self._compute_hessian(map_params_unconstrained, x_data, y_data, func, sigma_noise)
        
        # Compute covariance matrix (inverse of negative Hessian)
        try:
            cov_matrix_unconstrained = inv(-hessian)
        except np.linalg.LinAlgError:
            # If Hessian is not invertible, add small regularization
            cov_matrix_unconstrained = inv(-hessian + 1e-6 * np.eye(hessian.shape[0]))
        
        # Generate samples in unconstrained space
        samples_unconstrained = np.random.multivariate_normal(
            map_params_unconstrained, 
            cov_matrix_unconstrained, 
            size=n_samples
        )
        
        # Transform samples back to constrained space
        if self.constraint_manager is not None:
            from ligament_models.transformations import inverse_constraint_transform
            samples = np.array([inverse_constraint_transform(s, self.constraint_manager) 
                               for s in samples_unconstrained])
        else:
            samples = samples_unconstrained
        
        # Compute statistics in constrained space
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = 1.0  # Laplace approximation always accepts
        
        # Store convergence metrics
        self.convergence_metrics = {
            'hessian_condition_number': np.linalg.cond(hessian),
            'n_samples': n_samples,
            'method': 'laplace_approximation',
            'hessian_source': 'pre_computed' if loss_hessian is not None else 'finite_differences'
        }
        
        return cov_matrix, std_params, samples, 1.0
    
    def _compute_hessian(self, params, x_data, y_data, func, sigma_noise, epsilon=1e-6):
        """
        Compute Hessian matrix of log-posterior using finite differences.
        
        Args:
            params: Parameter vector (in unconstrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            epsilon: Step size for finite differences
            
        Returns:
            hessian: Hessian matrix
        """
        n_params = len(params)
        hessian = np.zeros((n_params, n_params))
        
        # Compute Hessian using finite differences
        for i in range(n_params):
            for j in range(n_params):
                # Forward differences
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()
                
                params_pp[i] += epsilon
                params_pp[j] += epsilon
                
                params_pm[i] += epsilon
                params_pm[j] -= epsilon
                
                params_mp[i] -= epsilon
                params_mp[j] += epsilon
                
                params_mm[i] -= epsilon
                params_mm[j] -= epsilon
                
                # Compute function values
                f_pp = self.log_probability(params_pp, x_data, y_data, func, sigma_noise)
                f_pm = self.log_probability(params_pm, x_data, y_data, func, sigma_noise)
                f_mp = self.log_probability(params_mp, x_data, y_data, func, sigma_noise)
                f_mm = self.log_probability(params_mm, x_data, y_data, func, sigma_noise)
                
                # Second derivative
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * epsilon**2)
        
        # Ensure symmetry
        hessian = 0.5 * (hessian + hessian.T)
        
        return hessian
