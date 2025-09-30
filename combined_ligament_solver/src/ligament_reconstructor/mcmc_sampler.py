from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
from src.ligament_models.constraints import ConstraintManager
from src.ligament_models.transformations import batch_inv_constraint_transform, batch_constraint_transform, slide_domain_inv_constraint_transform, slide_to_standard_domain, slide_domain_constraint_transform
import emcee
from copy import deepcopy



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
        try:
            # Transform to constrained space
            constrained_params = self._transform_to_constrained(params)
            
            # Get model predictions
            y_pred = func.vectorized_function(x_data, constrained_params)
            
            # Check for invalid predictions
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
    
    def __init__(self, constraint_manager=None, n_walkers=128, n_steps=310, n_burnin=300):
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
        if not np.all(np.isfinite(params)):
            return -np.inf
        
        # Transform to constrained space
        try:
            constrained_params = self._transform_to_constrained(params)
        except:
            return -np.inf
        
        # Check all parameters are within bounds
        constraints_list = self.constraint_manager.get_constraints_list()
        for i, (lower, upper) in enumerate(constraints_list):
            if not (lower <= constrained_params[i] <= upper):
                return -np.inf
        
        # Uniform prior: log(1/volume) = -log(volume)
        # For uniform priors, this is just a constant
        return 0.0
    
    def _transform_to_constrained(self, params: np.ndarray) -> np.ndarray:
        """Helper method to transform parameters to constrained space."""
        constrained_params = slide_domain_constraint_transform(
            params, self.constraint_manager.get_constraints_list(), self.map_params
        )
        constrained_params = slide_to_standard_domain(
            constrained_params, self.constraint_manager, self.map_params
        )
        return np.array(list(constrained_params.values()))
    
    def initial_walkers(self, map_params, n_walkers, n_params, std=0.1):
        """
        Initialize walkers in unconstrained space around the MAP estimate.
        """
        constraints_list = self.constraint_manager.get_constraints_list()
        
        if isinstance(map_params, dict):
            map_params = np.array(list(map_params.values()))
        
        # Create initial positions in constrained space
        initial_positions = np.zeros((n_walkers, n_params))
        
        # Parameter 0 (k): Gaussian around MAP with noise
        initial_positions[:, 0] = np.clip(
            map_params[0] + np.random.normal(0, std * map_params[0], n_walkers),
            constraints_list[0][0], constraints_list[0][1]
        )
        
        # Parameter 1 (alpha): Uniform within bounds
        initial_positions[:, 1] = np.random.uniform(
            constraints_list[1][0], constraints_list[1][1], n_walkers
        )
        
        # Parameter 2 (l_0): Gaussian around MAP with noise
        initial_positions[:, 2] = np.clip(
            map_params[2] + np.random.normal(0, std * map_params[2], n_walkers),
            constraints_list[2][0], constraints_list[2][1]
        )
        
        # Parameter 3 (l_slide): Uniform within slide range
        slide_range = constraints_list[2][1] - constraints_list[2][0]
        initial_positions[:, 3] = np.random.uniform(-slide_range/2, slide_range/2, n_walkers)
        
        # Transform to unconstrained space
        return np.array([
            list(slide_domain_inv_constraint_transform(
                pos, constraints_list, self.map_params
            ).values()) for pos in initial_positions
        ])
    
    def initial_walkers_screened(self, map_params, n_walkers, n_params, std=0.1, 
                                screen_percentage=0.1, x_data=None, y_data=None, func=None, sigma_noise=1e-3):
        """
        Initialize walkers by screening candidates and selecting the highest probability ones.
        
        Args:
            map_params: MAP estimate parameters
            n_walkers: Number of walkers needed
            n_params: Number of parameters
            std: Standard deviation for parameter sampling
            screen_percentage: Fraction of candidates to keep (e.g., 0.05 for top 5%)
            x_data: Input data for likelihood evaluation
            y_data: Target data for likelihood evaluation
            func: Function for likelihood evaluation
            sigma_noise: Noise standard deviation
        """
        # Generate many more candidates than needed
        n_candidates = int(n_walkers / screen_percentage)
        candidate_positions = self.initial_walkers(map_params, n_candidates, n_params, std)
        
        # Evaluate log-probability for all candidates
        log_probs = np.array([
            self.log_probability(candidate, x_data, y_data, func, sigma_noise)
            for candidate in candidate_positions
        ])
        
        # Select top candidates
        n_select = max(n_walkers, int(screen_percentage * n_candidates))
        top_indices = np.argsort(log_probs)[-n_select:]
        
        # Randomly select final walkers from top candidates
        if len(top_indices) >= n_walkers:
            selected_indices = np.random.choice(top_indices, n_walkers, replace=False)
        else:
            # Fill remaining with random selection if needed
            remaining_needed = n_walkers - len(top_indices)
            remaining_indices = np.random.choice(
                np.setdiff1d(np.arange(n_candidates), top_indices), 
                remaining_needed, replace=False
            )
            selected_indices = np.concatenate([top_indices, remaining_indices])
        
        print(f"Selected {len(selected_indices)} walkers from {n_candidates} candidates")
        print(f"Log-probability range: {log_probs[selected_indices].min():.2f} to {log_probs[selected_indices].max():.2f}")
        
        return candidate_positions[selected_indices]
    
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

        self.map_params = map_params.copy()

        # Override default parameters if provided
        n_walkers = kwargs.get('n_walkers', self.n_walkers)
        n_steps = kwargs.get('n_steps', self.n_steps)
        n_burnin = kwargs.get('n_burnin', self.n_burnin)
        
        n_params = len(map_params)


        # Set up MCMC sampler with multiple move types
        moves = [
            emcee.moves.StretchMove(),  # Good for correlated parameters
            emcee.moves.DEMove(),       # Escapes local maxima
            emcee.moves.WalkMove(),     # Local exploration
        ]
        
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params, self.log_probability,
            args=(x_data, y_data, func, sigma_noise),
            moves=moves
        )
        
        # Initialize walkers in unconstrained space
        # Use screened initialization by default, but allow override
        use_screening = kwargs.get('use_screening', True)
        screen_percentage = kwargs.get('screen_percentage', 0.1)
        
        if use_screening:
            initial_positions = self.initial_walkers_screened(
                map_params, n_walkers, n_params, 
                screen_percentage=screen_percentage,
                x_data=x_data, y_data=y_data, func=func, sigma_noise=sigma_noise
            )
        else:
            initial_positions = self.initial_walkers(map_params, n_walkers, n_params)
        
        # Optional: return initial positions for visualization (disable MCMC)
        if kwargs.get('visualize_only', False):
            untransformed_walkers = []
            for pos in initial_positions:
                pos = slide_domain_constraint_transform(pos, self.constraint_manager.get_constraints_list(), self.map_params)
                pos = slide_to_standard_domain(pos, self.constraint_manager, self.map_params)
                print(pos)
                untransformed_walkers.append(np.array(list(pos.values())))
            return None, None, untransformed_walkers, 1
    
        if random_state is not None:
            np.random.seed(random_state)
        
        # Run MCMC
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        # Get samples after burn-in and transform to constrained space
        samples_unconstrained = sampler.get_chain(discard=n_burnin, flat=True)
        samples = self._transform_samples_to_constrained(samples_unconstrained)
        
        print(f"Valid samples shape: {samples.shape}")
        
        # Compute statistics
        param_names = list(self.map_params.keys())
        mcmc_means = np.mean(samples, axis=0)
        print(f"Parameter means: {dict(zip(param_names, mcmc_means))}")
        
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = np.mean(sampler.acceptance_fraction)
        
        return cov_matrix, std_params, samples, self.acceptance_rate
    
    def _transform_samples_to_constrained(self, samples_unconstrained):
        """Transform MCMC samples from unconstrained to constrained space."""
        samples = []
        constraints = self.constraint_manager.get_constraints_list()
        
        for sample in samples_unconstrained:
            try:
                sample_dict = slide_domain_constraint_transform(sample, constraints, self.map_params)
                sample_dict = slide_to_standard_domain(sample_dict, self.constraint_manager, self.map_params)
                
                # Check if sample meets all constraints
                if all(lower <= value <= upper for (lower, upper), value in zip(constraints, sample_dict.values())):
                    samples.append(np.array(list(sample_dict.values())))
            except:
                continue  # Skip invalid samples
                
        return np.array(samples)
    