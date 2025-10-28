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
               sigma_noise: float = 1e-3,
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
    
    def __init__(self, knee_config, constraints_config, n_walkers=64, n_steps=2000, n_burnin=1500, num_samples=50):
        super().__init__()
        self.n_walkers = n_walkers
        self.n_steps = n_steps
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
        mcl_params = params[:4]
        lcl_params = params[4:]
        
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
        """Initialize walkers using scaled Gaussian around least squares results."""
        mcl_bounds = self.bounds['blankevoort_mcl']
        lcl_bounds = self.bounds['blankevoort_lcl']
        
        # Total parameters: 4 for MCL + 4 for LCL = 8
        n_mcl_params = len(mcl_bounds)
        n_lcl_params = len(lcl_bounds)
        total_params = n_mcl_params + n_lcl_params
        
        initial_positions = np.zeros((n_walkers, total_params))

        # Use least squares results if available, otherwise use default values
        if ls_result is not None:
            mcl_start_values = ls_result['mcl_params']
            lcl_start_values = ls_result['lcl_params']
            print(f"Using least squares results for MCMC initialization:")
            print(f"  MCL: {mcl_start_values}")
            print(f"  LCL: {lcl_start_values}")
        else:
            mcl_start_values = [33.5, 0.06, 90.0, 0.0]  # From config.yaml
            lcl_start_values = [42.8, 0.06, 60.0, 0.0]  # From config.yaml
            print("Using default values for MCMC initialization")
        
        # Initialize walkers for MCL parameters using scaled Gaussian
        for i in range(n_mcl_params):
            lower, upper = mcl_bounds[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i] = lower
            else:
                # Variable parameter - Gaussian around starting value
                start_val = mcl_start_values[i]
                # Scale standard deviation by parameter range
                param_range = upper - lower
                scaled_std = std * param_range
                # Generate Gaussian samples
                initial_positions[:, i] = np.random.normal(start_val, scaled_std, n_walkers)
                # Clip to bounds
                initial_positions[:, i] = np.clip(initial_positions[:, i], lower, upper)
        
        # Initialize walkers for LCL parameters using scaled Gaussian
        for i in range(n_lcl_params):
            lower, upper = lcl_bounds[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i + n_mcl_params] = lower
            else:
                # Variable parameter - Gaussian around starting value
                start_val = lcl_start_values[i]
                # Scale standard deviation by parameter range
                param_range = upper - lower
                scaled_std = std * param_range
                # Generate Gaussian samples
                initial_positions[:, i + n_mcl_params] = np.random.normal(start_val, scaled_std, n_walkers)
                # Clip to bounds
                initial_positions[:, i + n_mcl_params] = np.clip(initial_positions[:, i + n_mcl_params], lower, upper)
            
        return initial_positions

    def _initial_walkers_uniform_topk(self, n_walkers, thetas, applied_forces, sigma_noise,
                                      oversample_factor: int = 10, top_frac: float = 0.1,
                                      ls_result=None):
        """Uniformly sample candidate walkers within bounds, pick top-K by log_prob, and
        perturb/clip to produce n_walkers diverse initial positions.

        Falls back to Gaussian-around-LS if no finite candidates are found.
        """
        mcl_bounds = self.bounds['blankevoort_mcl']
        lcl_bounds = self.bounds['blankevoort_lcl']

        n_mcl_params = len(mcl_bounds)
        n_lcl_params = len(lcl_bounds)
        total_params = n_mcl_params + n_lcl_params

        num_candidates = max(n_walkers * oversample_factor, n_walkers)
        candidates = np.zeros((num_candidates, total_params), dtype=np.float64)

        # Sample uniformly within bounds; tiny jitter for fixed params to avoid singular columns
        tiny_jitter = 1e-12
        for i in range(n_mcl_params):
            lower, upper = mcl_bounds[i]
            if upper == lower:
                candidates[:, i] = lower + tiny_jitter * np.random.randn(num_candidates)
            else:
                candidates[:, i] = np.random.uniform(lower, upper, size=num_candidates)

        for i in range(n_lcl_params):
            lower, upper = lcl_bounds[i]
            col = i + n_mcl_params
            if upper == lower:
                candidates[:, col] = lower + tiny_jitter * np.random.randn(num_candidates)
            else:
                candidates[:, col] = np.random.uniform(lower, upper, size=num_candidates)

        # Evaluate log probability for candidates
        log_probs = np.full(num_candidates, -np.inf, dtype=np.float64)
        for idx in range(num_candidates):
            lp = self.log_probability(candidates[idx], thetas, applied_forces, sigma_noise)
            log_probs[idx] = lp

        finite_mask = np.isfinite(log_probs)
        if not np.any(finite_mask):
            print("Uniform init: No finite candidates; falling back to LS Gaussian init.")
            return self.initial_walkers(n_walkers, std=0.05, ls_result=ls_result)

        # Select top-K fraction
        finite_indices = np.where(finite_mask)[0]
        k = max(int(top_frac * len(finite_indices)), 1)
        top_indices_sorted = finite_indices[np.argsort(log_probs[finite_indices])[::-1]]
        top_indices = top_indices_sorted[:k]
        top_candidates = candidates[top_indices]

        # Build n_walkers by sampling from top candidates with perturbations
        initial_positions = np.zeros((n_walkers, total_params), dtype=np.float64)
        # Scale perturbation by 1% of parameter range (or small epsilon if fixed)
        per_param_scale = np.zeros(total_params, dtype=np.float64)
        for i in range(n_mcl_params):
            l, u = mcl_bounds[i]
            per_param_scale[i] = max(0.01 * (u - l), 1e-9)
        for i in range(n_lcl_params):
            l, u = lcl_bounds[i]
            per_param_scale[i + n_mcl_params] = max(0.01 * (u - l), 1e-9)

        for i in range(n_walkers):
            base = top_candidates[np.random.randint(0, len(top_candidates))]
            perturbed = base + np.random.randn(total_params) * per_param_scale
            # Clip to bounds
            for j in range(n_mcl_params):
                l, u = mcl_bounds[j]
                perturbed[j] = np.clip(perturbed[j], l, u)
            for j in range(n_lcl_params):
                l, u = lcl_bounds[j]
                perturbed[j + n_mcl_params] = np.clip(perturbed[j + n_mcl_params], l, u)
            initial_positions[i] = perturbed

        return initial_positions

    def sample(self, thetas, applied_forces, sigma_noise=1e-3, random_state=None, ls_result=None, **kwargs):
        """

        """

        # Override default parameters if provided
        n_walkers = kwargs.get('n_walkers', self.n_walkers)
        n_steps = kwargs.get('n_steps', self.n_steps)
        n_burnin = kwargs.get('n_burnin', 300)  # Default burnin for CompleteMCMCSampler
        n_params = 8

        moves = [
            emcee.moves.StretchMove(),  # Good for correlated parameters
            emcee.moves.DEMove(),       # Escapes local maxima
            emcee.moves.WalkMove(),     # Local exploration
        ]
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params, self.log_probability,
            args=(thetas, applied_forces, sigma_noise),
            moves=moves
        )

        # Initialize walkers via uniform sampling and selecting top 10% most likely
        initial_positions = self._initial_walkers_uniform_topk(
            n_walkers, thetas, applied_forces, sigma_noise, oversample_factor=10, top_frac=0.1, ls_result=ls_result
        )
        
        # Debug: Check walker diversity and validity
        print(f"Initial walker parameter ranges:")
        for i in range(initial_positions.shape[1]):
            min_val = np.min(initial_positions[:, i])
            max_val = np.max(initial_positions[:, i])
            print(f"  Param {i}: [{min_val:.6f}, {max_val:.6f}] (range: {max_val-min_val:.6f})")
        
        # Check if initial walkers are valid (finite log probability)
        print("Checking initial walker validity...")
        valid_walkers = 0
        for i in range(min(5, n_walkers)):  # Check first 5 walkers
            log_prob = self.log_probability(initial_positions[i], thetas, applied_forces, sigma_noise)
            print(f"  Walker {i}: log_prob = {log_prob:.2f}")
            if np.isfinite(log_prob):
                valid_walkers += 1
        
        print(f"Valid walkers: {valid_walkers}/{min(5, n_walkers)}")
        
        if valid_walkers == 0:
            print("WARNING: No valid initial walkers! This will cause 0% acceptance rate.")
            # Try to fix by using a more conservative initialization
            print("Attempting conservative initialization...")
            initial_positions = self.initial_walkers(n_walkers, std=0.01, ls_result=ls_result)
        
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        samples = sampler.get_chain(discard=n_burnin, flat=True)
        acceptance_rate = np.mean(sampler.acceptance_fraction)
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))

        if len(samples) > num_samples:
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
        
        # Split parameters into MCL (first 4) and LCL (last 4) parameters
        mcl_params = params[:4]  # [k, alpha, l_0, f_ref]
        lcl_params = params[4:]  # [k, alpha, l_0, f_ref]
        
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



class MCMCSampler(BaseSampler):
    """
    MCMC sampler using emcee for Bayesian inference.
    """
    
    def __init__(self, constraint_manager=None, n_walkers=64, n_steps=350, n_burnin=300):
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

    def initial_walkers(self, map_params, n_walkers, n_params, std=0.1):
        """Initialize walkers in unconstrained space around the MAP estimate."""
        constraints_list = self.constraint_manager.get_constraints_list()
        assert_parameter_format(map_params)
        
        if isinstance(map_params, dict):
            map_params = np.array(list(map_params.values()))
            
        initial_positions = np.zeros((n_walkers, n_params))
        
        # k parameter: Gaussian around MAP with noise, clipped to bounds
        initial_positions[:, 0] = np.clip(
            map_params[0] + np.random.normal(0, std * map_params[0], n_walkers),
            constraints_list[0][0], constraints_list[0][1]
        )
        
        # Other parameters: Uniform within bounds
        for i in range(1, n_params):
            initial_positions[:, i] = np.random.uniform(
                constraints_list[i][0], constraints_list[i][1], n_walkers
            )

        assert_parameter_format(initial_positions[0])
            
        return initial_positions

    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, num_samples=50, 
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
        if isinstance(map_params, dict):
            param_names = list(map_params.keys())
            map_params = np.array(list(map_params.copy().values()))
        else:
            map_params = map_params.copy()

        self.map_params = map_params
        assert_parameter_format(self.map_params)

        # Override default parameters if provided
        n_walkers = kwargs.get('n_walkers', self.n_walkers)
        n_steps = kwargs.get('n_steps', self.n_steps)
        n_burnin = kwargs.get('n_burnin', self.n_burnin)
        
        n_params = len(map_params)


        # Set up MCMC sampler with multiple move types
        moves = [
            emcee.moves.StretchMove(),  # Good for correlated parameters
            # emcee.moves.DEMove(),       # Escapes local maxima
            # emcee.moves.WalkMove(),     # Local exploration
        ]
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params, self.log_probability,
            args=(x_data, y_data, func, sigma_noise),
            moves=moves
        )
        
        initial_positions = self.initial_walkers(map_params, n_walkers, n_params)
        sampler.run_mcmc(initial_positions, n_steps, progress=True)       
        # Get after burn-in and transform to constrained space
        samples = sampler.get_chain(discard=n_burnin, flat=True)       

        mcmc_means = np.mean(samples, axis=0)
        print(f"Parameter means: {dict(zip(param_names, mcmc_means))}")

        # Compare likelihood of LS and MCMC mean solutions
        ls_params = kwargs.get('ls_result', {}).get('params')
        if ls_params is not None:
            ls_ll = self.log_probability(ls_params, x_data, y_data, func, sigma_noise)
            mcmc_ll = self.log_probability(mcmc_means, x_data, y_data, func, sigma_noise)
            print("\nLog likelihood comparison:")
            print(f"LS solution log likelihood: {ls_ll:.2f}")
            print(f"MCMC mean log likelihood: {mcmc_ll:.2f}")
        
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = np.mean(sampler.acceptance_fraction)

        
        
        return cov_matrix, std_params, samples, self.acceptance_rate
    