from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
import emcee
from copy import deepcopy
from src.statics_model import KneeModel
from src.ligament_optimiser import parse_constraints

def assert_parameter_format(params: np.ndarray):
    """
    Assert that the parameters are in the correct format.
    """
    if not isinstance(params, np.ndarray):
        raise ValueError(f"Parameters must be a numpy array, got  {type(params)}")
    if not params.ndim == 1:
        raise ValueError(f"Parameters must be a 1D array, got {params.ndim}")
    if params.dtype != np.float64:
        raise ValueError(f"Parameters must be a float64 array, got {params.dtype}")
    

class BaseSampler(ABC):
    """
    Abstract base class for Bayesian sampling methods.
    
    This class provides a common interface for different Bayesian inference
    algorithms including MCMC, Variational Inference, Laplace Approximation,
    and others.
    """
    
    def __init__(self):
        """
        Initialize the sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
        """
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
               sigma_noise: float,
               **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        pass
    
    def log_likelihood(self, params: np.ndarray, x_data: np.ndarray, 
                      y_data: np.ndarray, func: Any, sigma_noise: float = 100) -> float:
        """
        Compute log-likelihood function assuming Gaussian noise.
        
        Args:
            params: Parameter vector (in unconstrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            log_likelihood: Log-likelihood value
        """
        assert_parameter_format(params)

        try:
            y_pred = func.vectorized_function(x_data, params)
            if not np.all(np.isfinite(y_pred)):
                return -np.inf
            
            # Compute residuals
            residuals = y_data - y_pred.flatten()
            
            # Gaussian log-likelihood: -½∑(residuals²/σ²) - N*log(σ√(2π))
            log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
            
            return log_like if np.isfinite(log_like) else -np.inf
            
        except:
            return -np.inf
    
    def log_prior(self, params: np.ndarray) -> float:
        """
        Compute log-prior probability for parameters.
        
        Uses uniform priors within parameter constraints.
        
        Args:
            params: Parameter vector (in unconstrained space)
            
        Returns:
            log_prior: Log-prior probability value
        """
        # Check parameters are finite
        assert_parameter_format(params)
        if not np.all(np.isfinite(params)):
            return -np.inf
        
        # Check all parameters are within bounds
        constraints_list = self.constraint_manager.get_constraints_list()
        for i, (lower, upper) in enumerate(constraints_list):
            if not (lower <= params[i] <= upper):
                return -np.inf
        
        # Uniform prior: log(1/volume) = -log(volume)
        # For uniform priors, this is just a constant
        return 0.0
    
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
        assert_parameter_format(params)

        lp = self.log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        
        ll = self.log_likelihood(params, x_data, y_data, func, sigma_noise)
        return lp + ll

class CompleteMCMCSampler(BaseSampler):
    """
    Complete MCMC sampler using emcee for Bayesian inference.
    """
    
    def __init__(self, knee_config, constraints_config, n_walkers=32, n_steps=200, n_burnin=150, num_samples=32*50):
        super().__init__()
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.n_burnin = n_burnin
        self.num_samples = num_samples
        self.knee_config = knee_config  # Store pre-built knee model
        self.constraints_config = constraints_config  # Store constraints config
        
        # Parse constraints to get bounds
        self.bounds = parse_constraints(constraints_config)
        
        # Create and cache a KneeModel instance to reuse across likelihood evaluations
        # This avoids the expensive rebuild cost on every call
        
        # Build model once with dummy parameters
        self.knee_model = KneeModel(self.knee_config, log=False)
        self.knee_model.build_geometry()

    def log_probability(self, params: np.ndarray, thetas: np.ndarray, 
                       applied_forces: np.ndarray, sigma_noise: float) -> float:
        """Compute log-probability (prior + likelihood)."""
        assert_parameter_format(params)
        
        # Check both constraint bounds (MCL and LCL)
        mcl_params = params[:3]
        lcl_params = params[3:]
        
        # Check MCL constraints
        mcl_lp = self._log_prior_single(mcl_params, self.bounds['blankevoort_mcl'])
        if not np.isfinite(mcl_lp):
            return -np.inf
        
        # Check LCL constraints
        lcl_lp = self._log_prior_single(lcl_params, self.bounds['blankevoort_lcl'])
        if not np.isfinite(lcl_lp):
            return -np.inf
        
        ll = self.log_likelihood(params, thetas, applied_forces, sigma_noise)
        
        total_log_prob = mcl_lp + lcl_lp + ll
        
        return total_log_prob

    def _log_prior_single(self, params: np.ndarray, bounds: list) -> float:
        """
        Compute log-prior probability for a single set of parameters using bounds.
        
        Args:
            params: Parameter vector
            bounds: List of (lower, upper) bounds for each parameter
            
        Returns:
            log_prior: Log-prior probability value
        """
        # Check parameters are finite
        if not np.all(np.isfinite(params)):
            return -np.inf
        
        # Check all parameters are within bounds
        for i, (lower, upper) in enumerate(bounds):
            if not (lower <= params[i] <= upper):
                return -np.inf
        
        # Uniform prior: log(1/volume) = -log(volume)
        # For uniform priors, this is just a constant
        return 0.0

    def initial_walkers(self, n_walkers, std=0.1, ls_result=None):
        """Initialize walkers uniformly within parameter bounds."""
        mcl_bounds = self.bounds['blankevoort_mcl']
        lcl_bounds = self.bounds['blankevoort_lcl']
        
        # Total parameters: 3 for MCL + 3 for LCL = 6
        n_mcl_params = len(mcl_bounds)
        n_lcl_params = len(lcl_bounds)
        total_params = n_mcl_params + n_lcl_params
        
        initial_positions = np.zeros((n_walkers, total_params))

        print("Using uniform initialization within parameter bounds")
        
        # Initialize walkers for MCL parameters uniformly within bounds
        for i in range(n_mcl_params):
            lower, upper = mcl_bounds[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i] = lower
            else:
                # Variable parameter - uniform within bounds
                initial_positions[:, i] = np.random.uniform(lower, upper, n_walkers)
        
        # Initialize walkers for LCL parameters uniformly within bounds
        for i in range(n_lcl_params):
            lower, upper = lcl_bounds[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i + n_mcl_params] = lower
            else:
                # Variable parameter - uniform within bounds
                initial_positions[:, i + n_mcl_params] = np.random.uniform(lower, upper, n_walkers)
            
        return initial_positions

    def sample(self, thetas, applied_forces, sigma_noise, random_state=None, ls_result=None, **kwargs):
        """

        """

        # Ensure burn-in is less than total steps
        if self.n_burnin >= self.n_steps:
            raise ValueError(f"Burn-in {self.n_burnin} steps must be less than total steps {self.n_steps}")
        n_params = 6

        # Build per-parameter Gaussian step scales (~5% of each parameter range)
        mcl_bounds = self.bounds['blankevoort_mcl']
        lcl_bounds = self.bounds['blankevoort_lcl']
        per_param_scale = []
        for low, up in mcl_bounds + lcl_bounds:
            rng = max(up - low, 1e-9)
            per_param_scale.append(0.2 * rng)
        per_param_scale = np.array(per_param_scale, dtype=np.float64)

        moves = [emcee.moves.StretchMove()]
        
        sampler = emcee.EnsembleSampler(
            self.n_walkers, n_params, self.log_probability,
            args=(thetas, applied_forces, sigma_noise),
            moves=moves
        )

        # Initialize walkers uniformly within bounds for maximum diversity
        initial_positions = self.initial_walkers(
            self.n_walkers, std=0.25, ls_result=ls_result
        )
        sampler.run_mcmc(initial_positions, self.n_steps, progress=True)
        samples = sampler.get_chain(discard=self.n_burnin, flat=True)
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))

        if len(samples) > self.num_samples:
            # Randomly select num_samples indices without replacement
            sample_indices = np.random.choice(len(samples), size=self.num_samples, replace=False)
            samples = samples[sample_indices]

        return cov_matrix, std_params, samples, acceptance_rate

    def log_likelihood(self, params: np.ndarray, thetas: np.ndarray, 
                      applied_forces: np.ndarray, sigma_noise: float = 1e-3) -> float:
        """
        Compute log-likelihood function assuming Gaussian noise.
        
        Uses the same approach as ligament_optimiser.py but presents as log likelihood.
        """
        assert_parameter_format(params)
        
        # Split parameters into MCL (first 3) and LCL (last 3) parameters
        mcl_params = params[:3]  # [k, alpha, l_0]
        lcl_params = params[3:]  # [k, alpha, l_0]
        
        # Validate parameter arrays
        if not np.all(np.isfinite(mcl_params)) or not np.all(np.isfinite(lcl_params)):
            return -np.inf
        
        try:
            # Use the same approach as ligament_optimiser.py
            estimated_applied_forces = self.knee_model.solve_applied(thetas, mcl_params, lcl_params)['applied_forces']
            estimated_applied_forces = np.array(estimated_applied_forces).reshape(-1)
            
            # Convert squared residuals to log likelihood
            # ligament_optimiser uses: np.sum((applied_forces - estimated_applied_forces)**2 / len(thetas))
            # We convert this to log likelihood: -0.5 * sum(residuals^2) / sigma^2 - N*log(sigma*sqrt(2*pi))
            residuals = applied_forces - estimated_applied_forces
            n_data = len(thetas)
            
            # Gaussian log-likelihood: -½∑(residuals²/σ²) - N*log(σ√(2π))
            log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - n_data * np.log(sigma_noise * np.sqrt(2 * np.pi))
            
            return log_like if np.isfinite(log_like) else -np.inf
            
        except:
            return -np.inf


    def sample_independent(self, thetas, applied_forces, sigma_noise, n_candidates=5000, top_k=500):
        """
        Independent sampling: draw candidates uniformly within bounds and select
        the top_k by posterior log-probability. This is not a Markov chain, but
        provides a quick approximation and broad exploration across the bounds.

        Returns (cov_matrix, std_params, samples, acceptance_rate)
        where acceptance_rate is NaN for this method.
        """
        mcl_bounds = self.bounds['blankevoort_mcl']
        lcl_bounds = self.bounds['blankevoort_lcl']
        total_params = len(mcl_bounds) + len(lcl_bounds)

        candidates = np.zeros((n_candidates, total_params), dtype=np.float64)

        # Uniform sampling within bounds
        for i, (low, up) in enumerate(mcl_bounds + lcl_bounds):
            if up == low:
                candidates[:, i] = low
            else:
                candidates[:, i] = np.random.uniform(low, up, size=n_candidates)

        # Evaluate log-probability
        log_probs = np.full(n_candidates, -np.inf, dtype=np.float64)
        for idx in range(n_candidates):
            log_probs[idx] = self.log_probability(candidates[idx], thetas, applied_forces, sigma_noise)

        # Select top_k by log probability
        finite_mask = np.isfinite(log_probs)
        if not np.any(finite_mask):
            # Fallback: return empty result
            return np.empty((0, total_params)), np.array([]), np.empty((0, total_params)), np.nan

        finite_idx = np.where(finite_mask)[0]
        order = np.argsort(log_probs[finite_idx])[::-1]
        top_idx = finite_idx[order[:min(top_k, len(finite_idx))]]
        samples = candidates[top_idx]

        # Stats
        if samples.shape[0] >= 2:
            cov_matrix = np.cov(samples, rowvar=False)
            std_params = np.sqrt(np.diag(cov_matrix))
        else:
            cov_matrix = np.zeros((total_params, total_params))
            std_params = np.zeros(total_params)

        return cov_matrix, std_params, samples, np.nan


