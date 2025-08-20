from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
from ligament_models.constraints import ConstraintManager


class BaseSampler(ABC):
    """
    Abstract base class for Bayesian sampling methods.
    
    This class provides a common interface for different Bayesian inference
    algorithms including MCMC, Variational Inference, Laplace Approximation,
    and others.
    """
    
    def __init__(self, constraint_manager: Optional[ConstraintManager] = None):
        """
        Initialize the sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
        """
        self.constraint_manager = constraint_manager
        self.samples = None
        self.covariance_matrix = None
        self.parameter_std = None
        self.acceptance_rate = None
        self.convergence_metrics = {}
    
    @abstractmethod
    def sample(self, 
               map_params: Dict[str, float],
               x_data: np.ndarray,
               y_data: np.ndarray,
               func: Any,
               sigma_noise: float = 1e-3,
               **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Generate samples from the posterior distribution.
        
        Args:
            map_params: MAP estimate (optimal parameters in constrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            **kwargs: Additional method-specific parameters
            
        Returns:
            cov_matrix: Covariance matrix (in constrained space)
            std_params: Standard deviations of parameters (in constrained space)
            samples: Parameter samples (in constrained space)
            acceptance_rate: Acceptance rate (if applicable)
        """
        pass
    
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
        from copy import deepcopy
        func = deepcopy(func)

        # Transform parameters back to constrained space for function evaluation
        from ligament_models.transformations import inverse_constraint_transform
        params = inverse_constraint_transform(params, self.constraint_manager)
        
        func.set_params(params)
        y_pred = np.array([func(x) for x in x_data])
        
        # Compute residuals
        residuals = y_data - y_pred
        
        # Log-likelihood (assuming Gaussian noise)
        log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
        
        return log_like
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute log-prior function in unconstrained space.
        
        Args:
            params: Parameter vector (in unconstrained space)
            
        Returns:
            log_prior: Log-prior value
        """
        # Basic check: parameters should be finite
        for param in params:
            if not np.isfinite(param):
                return -np.inf
        
        # Isotropic normal prior with mean 0 and std 100
        # Log of N(0, 100^2) = -0.5 * (x^2 / 100^2) - log(100 * sqrt(2*pi))
        prior_std = 1
        log_prior_value = -0.5 * np.sum(params**2) / (prior_std**2) - len(params) * np.log(prior_std * np.sqrt(2 * np.pi))
        
        return log_prior_value
    
    def log_probability(self, params: np.ndarray, x_data: np.ndarray, 
                       y_data: np.ndarray, func: Any, sigma_noise: float) -> float:
        """
        Compute log-probability function (prior + likelihood).
        
        Args:
            params: Parameter vector (in unconstrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            log_prob: Log-probability value
        """
        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params, x_data, y_data, func, sigma_noise)
        return lp + ll
    
    def get_samples(self) -> Optional[np.ndarray]:
        """Get the generated samples."""
        return self.samples
    
    def get_covariance(self) -> Optional[np.ndarray]:
        """Get the covariance matrix."""
        return self.covariance_matrix
    
    def get_parameter_std(self) -> Optional[np.ndarray]:
        """Get the parameter standard deviations."""
        return self.parameter_std
    
    def get_acceptance_rate(self) -> Optional[float]:
        """Get the acceptance rate (if applicable)."""
        return self.acceptance_rate
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get convergence metrics for the sampling method."""
        return self.convergence_metrics
    
    def estimate_noise_level(self, x_data: np.ndarray, y_data: np.ndarray, 
                           map_params: Dict[str, float], func: Any) -> float:
        """
        Estimate noise level from residuals at MAP estimate.
        
        Args:
            x_data: Input data points
            y_data: Target data points
            map_params: MAP estimate
            func: Function to evaluate
            
        Returns:
            sigma_noise: Estimated noise standard deviation
        """
        y_pred = np.array([func(x) for x in x_data])
        residuals = y_data - y_pred
        return np.std(residuals)
