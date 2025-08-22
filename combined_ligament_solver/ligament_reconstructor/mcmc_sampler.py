from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import constraint_transform, inverse_constraint_transform
import emcee
from copy import deepcopy
from ligament_models.transformations import inverse_constraint_transform, constraint_transform



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
        # prior_std = 10
        # log_prior_value = -0.5 * np.sum(params**2) / (prior_std**2) - len(params) * np.log(prior_std * np.sqrt(2 * np.pi))
        log_prior_value = 1

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

class MCMCSampler(BaseSampler):
    """
    MCMC sampler using emcee for Bayesian inference.
    """
    
    def __init__(self, constraint_manager=None, n_walkers=16, n_steps=200, n_burnin=20):
        """
        Initialize MCMC sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
            n_walkers: Number of MCMC walkers
            n_steps: Number of MCMC steps
            n_burnin: Number of burn-in steps to discard
        """
        super().__init__(constraint_manager)
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burnin = n_burnin  
    
    def initial_walkers(self, map_params, n_walkers, n_params, constraint_manager=None):
        # """
        # Initialize walkers in unconstrained space around the transformed MAP estimate.
        # """
        # constraints_list = constraint_manager.get_constraints_list()

        # if isinstance(map_params, dict):
        #     map_params = np.array(list(map_params.values()))
        
        # # Create initial positions array with shape (n_walkers, n_params)
        # initial_positions = np.zeros((n_walkers, n_params))
        # # First parameter: MAP estimate + small noise
        # initial_positions[:, 0] = map_params[0] * (1 + np.random.normal(0, 0.01, n_walkers))
        # # Clip to constraints
        # initial_positions[:, 0] = np.clip(
        #     initial_positions[:, 0],
        #     constraints_list[0][0],
        #     constraints_list[0][1]
        # )

        # # Second parameter: uniform distribution
        # initial_positions[:, 1] = np.random.uniform(
        #     constraints_list[1][0],
        #     constraints_list[1][1],
        #     n_walkers
        # )

        # # Third parameter: Gaussian distribution centered at MAP estimate
        # initial_positions[:, 2] = map_params[2] * (1 + np.random.normal(0, 0.01, n_walkers))
        # # Clip to constraints
        # initial_positions[:, 2] = np.clip(
        #     initial_positions[:, 2],
        #     constraints_list[2][0],
        #     constraints_list[2][1]
        # )

        # # Fourth parameter: Gaussian distribution centered at MAP estimate
        # initial_positions[:, 3] = map_params[3] * (1 + np.random.normal(0, 0.01, n_walkers))
        # # Clip to constraints
        # initial_positions[:, 3] = np.clip(
        #     initial_positions[:, 3],
        #     constraints_list[3][0],
        #     constraints_list[3][1]
        # )
        # # Transform to unconstrained space
        # initial_positions_transformed = np.array([
        #     constraint_transform(pos, constraint_manager) for pos in initial_positions
        # ])

        # Initialize walkers with zero-centered Gaussian distribution
        initial_positions = np.random.normal(0, 1, size=(n_walkers, n_params))
        return initial_positions
    
    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, 
               random_state=None, **kwargs):
        """
        Generate samples using MCMC.
        
        Args:
            map_params: MAP estimate (optimal parameters in constrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            random_state: Random state for reproducibility
            **kwargs: Additional parameters (can override n_walkers, n_steps, n_burnin)
            
        Returns:
            cov_matrix: MCMC covariance matrix (in constrained space)
            std_params: Standard deviations of parameters (in constrained space)
            mcmc_samples: All MCMC samples (in constrained space)
            acceptance_fraction: Acceptance fraction of the sampler
        """
        # Override default parameters if provided
        n_walkers = kwargs.get('n_walkers', self.n_walkers)
        n_steps = kwargs.get('n_steps', self.n_steps)
        n_burnin = kwargs.get('n_burnin', self.n_burnin)
        
        n_params = len(map_params)

        # eigenvectors = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]]
        # eigenvectors = np.array(eigenvectors)
        # eigenvalues = [1, 1, 1, 1]
        # covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # covariance_matrix = covariance_matrix * 1e+1
        covariance_matrix = np.eye(n_params)

        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params, self.log_probability,
            args=(x_data, y_data, func, sigma_noise),
            moves=emcee.moves.DEMove()
        )
        
        # Initialize walkers in unconstrained space
        initial_positions = self.initial_walkers(map_params, n_walkers, n_params, self.constraint_manager)
    
        if random_state is not None:
            np.random.seed(random_state)
        
        # Run MCMC
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        # Get samples after burn-in (still in unconstrained space)
        samples_unconstrained = sampler.get_chain(discard=n_burnin, flat=True)
        
        # Transform samples back to constrained space
        if self.constraint_manager is not None:
            samples = np.array([inverse_constraint_transform(s, self.constraint_manager) for s in samples_unconstrained])
        else:
            samples = samples_unconstrained
        
        # Ensure samples is a numpy array
        samples = np.array(samples)
        
        # Compute statistics in constrained space
        mcmc_means = np.mean(samples, axis=0)
        print(mcmc_means)
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = np.mean(sampler.acceptance_fraction)
        
        return cov_matrix, std_params, samples, self.acceptance_rate
    