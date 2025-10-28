from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple, Optional, Any
from src.ligament_models.constraints import ConstraintManager
import emcee
from copy import deepcopy
from src.ligament_models.blankevoort import BlankevoortFunction

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
    
    def __init__(self, knee_config, constraint_manager, n_walkers=64, n_steps=350, n_burnin=300):
        super().__init__(constraint_manager)
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.knee_config = knee_config  # Store pre-built knee model
        
        # Create and cache a KneeModel instance to reuse across likelihood evaluations
        # This avoids the expensive rebuild cost on every call
        from src.ligament_models.blankevoort import BlankevoortFunction
        from src.statics_solver.models.statics_model import KneeModel
        
        # Build model once with dummy parameters
        self._cached_knee_model = KneeModel(self.knee_config, log=False)
        self._cached_knee_model.build_geometry()
        # Dummy build


    def _log_prior_single(self, params: np.ndarray, constraint_manager) -> float:
        """Check constraints for a single ligament's parameters."""
        if not np.all(np.isfinite(params)):
            return -np.inf
        
        constraints_list = constraint_manager.get_constraints_list()
        for i, (lower, upper) in enumerate(constraints_list):
            if not (lower <= params[i] <= upper):
                return -np.inf
        
        return 0.0
    
    def log_probability(self, params: np.ndarray, thetas: np.ndarray, 
                       applied_forces: np.ndarray, sigma_noise: float) -> float:
        """Compute log-probability (prior + likelihood)."""
        assert_parameter_format(params)
        
        
        # Check both constraint managers (MCL and LCL)
        mcl_params = params[:4]
        lcl_params = params[4:]
        
        # Check MCL constraints
        mcl_lp = self._log_prior_single(mcl_params, self.constraint_manager[0])
        if not np.isfinite(mcl_lp):
            return -np.inf
        
        # Check LCL constraints
        lcl_lp = self._log_prior_single(lcl_params, self.constraint_manager[1])
        if not np.isfinite(lcl_lp):
            return -np.inf
        
        ll = self.log_likelihood(params, thetas, applied_forces, sigma_noise)
        
        total_log_prob = mcl_lp + lcl_lp + ll
        
        return total_log_prob

    def initial_walkers(self, n_walkers, std=0.1, ls_result=None):
        """Initialize walkers in unconstrained space around reasonable starting values or least squares results."""
        mcl_constraints_manager = self.constraint_manager[0]
        lcl_constraints_manager = self.constraint_manager[1]

        mcl_constraints_list = mcl_constraints_manager.get_constraints_list()
        lcl_constraints_list = lcl_constraints_manager.get_constraints_list()
        
        # Total parameters: 4 for MCL + 4 for LCL = 8
        n_mcl_params = len(mcl_constraints_list)
        n_lcl_params = len(lcl_constraints_list)
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
        
        # Initialize walkers for MCL parameters
        for i in range(n_mcl_params):
            lower, upper = mcl_constraints_list[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i] = lower
            else:
                # Variable parameter - start near starting values with small noise
                start_val = mcl_start_values[i]
                # Ensure start value is within bounds
                start_val = np.clip(start_val, lower, upper)
                # Add small random noise around the starting value
                noise_scale = min(std * (upper - lower), 0.01 * (upper - lower))
                initial_positions[:, i] = start_val + np.random.normal(0, noise_scale, n_walkers)
                # Clip to bounds
                initial_positions[:, i] = np.clip(initial_positions[:, i], lower, upper)
        
        # Initialize walkers for LCL parameters
        for i in range(n_lcl_params):
            lower, upper = lcl_constraints_list[i]
            if lower == upper:
                # Fixed parameter - set all walkers to the same value
                initial_positions[:, i + n_mcl_params] = lower
            else:
                # Variable parameter - start near starting values with small noise
                start_val = lcl_start_values[i]
                # Ensure start value is within bounds
                start_val = np.clip(start_val, lower, upper)
                # Add small random noise around the starting value
                noise_scale = min(std * (upper - lower), 0.01 * (upper - lower))
                initial_positions[:, i + n_mcl_params] = start_val + np.random.normal(0, noise_scale, n_walkers)
                # Clip to bounds
                initial_positions[:, i + n_mcl_params] = np.clip(initial_positions[:, i + n_mcl_params], lower, upper)
            
        return initial_positions

    def initial_walkers_screened(self, n_walkers, std=0.1, 
                                screen_percentage=0.1, thetas=None, applied_forces=None, sigma_noise=1e-3, ls_result=None):
        """
        Initialize walkers by screening candidates and selecting the highest probability ones.
        
        Args:
            n_walkers: Number of walkers needed
            std: Standard deviation for parameter sampling
            screen_percentage: Fraction of candidates to keep (e.g., 0.05 for top 5%)
            thetas: Input data for likelihood evaluation
            applied_forces: Target data for likelihood evaluation
            sigma_noise: Noise standard deviation
            ls_result: Least squares optimization results for better initialization
        """
        # Generate many more candidates than needed
        n_candidates = int(n_walkers / screen_percentage)
        candidate_positions = self.initial_walkers(n_candidates, std, ls_result)
        
        # Evaluate log-probability for all candidates
        log_probs = np.array([
            self.log_probability(candidate, thetas, applied_forces, sigma_noise)
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

    def sample(self, thetas, applied_forces, lcl_lengths, mcl_lengths, sigma_noise=1e-3, random_state=None, ls_result=None, **kwargs):
        """
        Generate samples using MCMC.
        
        Args:
            thetas: Input data points (knee angles)
            applied_forces: Target data points (applied forces)
            lcl_lengths: LCL lengths for each theta
            mcl_lengths: MCL lengths for each theta
            sigma_noise: Noise standard deviation
            random_state: Random state for reproducibility
            ls_result: Least squares optimization results for better initialization
            **kwargs: Additional parameters (can override n_walkers, n_steps, n_burnin)
            
        Returns:
            cov_matrix: MCMC covariance matrix
            std_params: Standard deviations of parameters
            mcmc_samples: All MCMC samples
            acceptance_fraction: Acceptance fraction of the sampler
        """

        self.lcl_lengths = lcl_lengths
        self.mcl_lengths = mcl_lengths

        # Override default parameters if provided
        n_walkers = kwargs.get('n_walkers', self.n_walkers)
        n_steps = kwargs.get('n_steps', self.n_steps)
        n_burnin = kwargs.get('n_burnin', 300)  # Default burnin for CompleteMCMCSampler
        
        # Total parameters: 4 for MCL + 4 for LCL = 8
        n_params = 8
        
        # Set up MCMC sampler with multiple move types
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
        
        # Initialize walkers in unconstrained space
        # Use screened initialization by default, but allow override
        use_screening = kwargs.get('use_screening', True)
        screen_percentage = kwargs.get('screen_percentage', 0.1)
        
        if use_screening:
            initial_positions = self.initial_walkers_screened(
                n_walkers, 
                screen_percentage=screen_percentage,
                thetas=thetas, applied_forces=applied_forces, sigma_noise=sigma_noise,
                ls_result=ls_result
            )
        else:
            initial_positions = self.initial_walkers(n_walkers, ls_result=ls_result)
        
        # Optional: return initial positions for visualization (disable MCMC)
        if kwargs.get('visualize_only', False):
            return None, None, initial_positions, 1
    
        if random_state is not None:
            np.random.seed(random_state)
        
        # Check for any columns with zero variance (all identical values)
        zero_var_cols = np.where(np.std(initial_positions, axis=0) == 0)[0]
        if len(zero_var_cols) > 0:
            # Add small random noise to fixed parameters to ensure linear independence
            for col in zero_var_cols:
                fixed_value = initial_positions[0, col]
                # Add very small noise (1e-10) to make walkers linearly independent
                noise = np.random.normal(0, 1e-10, initial_positions.shape[0])
                initial_positions[:, col] = fixed_value + noise
        
        # Run MCMC
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        # Get samples after burn-in
        samples = sampler.get_chain(discard=n_burnin, flat=True)
        print(f"Valid samples shape: {samples.shape}")
        
        # Compute statistics
        mcmc_means = np.mean(samples, axis=0)
        print(f"Parameter means: {mcmc_means}")
        
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = np.mean(sampler.acceptance_fraction)
        
        return cov_matrix, std_params, samples, self.acceptance_rate

    def log_likelihood(self, params: np.ndarray, thetas: np.ndarray, 
                      applied_forces: np.ndarray, sigma_noise: float = 1e-3) -> float:
        """
        Compute log-likelihood function assuming Gaussian noise.
        
        For a given theta, calculate the elongation of the ligaments using the knee model
        Then calculate the force using the Blankevoort function and supplied parameters
        The supplied parameters are the MCL and LCL parameters in that order
        Then use these values to calculate the necessary applied force again using the knee model.
        Finally report both these values and the (likelihood) difference between them.
        """
        assert_parameter_format(params)
        
        # Split parameters into MCL (first 4) and LCL (last 4) parameters
        mcl_params = params[:4]  # [k, alpha, l_0, f_ref]
        lcl_params = params[4:]  # [k, alpha, l_0, f_ref]
        
        # Validate parameter arrays
        if not np.all(np.isfinite(mcl_params)) or not np.all(np.isfinite(lcl_params)):
            return -np.inf
        
        mcl_func = BlankevoortFunction(mcl_params)
        lcl_func = BlankevoortFunction(lcl_params)
        
        # Use cached model and update parameters (fast!)
        knee_model = self._cached_knee_model
        knee_model.build_ligament_forces(mcl_func, lcl_func)
        
        # Calculate predicted forces for all thetas
        results = knee_model.calculate_thetas(thetas)
        predicted_forces = results['applied_forces']
        
        # Compute residuals
        residuals = np.array(applied_forces) - np.array(predicted_forces)   
        
        # Gaussian log-likelihood: -½∑(residuals²/σ²) - N*log(σ√(2π))
        log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(thetas) * np.log(sigma_noise * np.sqrt(2 * np.pi))
        # Ensure log_like is a numeric value before checking if it's finite
        
        try:
            log_like = float(log_like)
            return log_like if np.isfinite(log_like) else -np.inf
        except (TypeError, ValueError):
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
                untransformed_walkers.append(pos)
            return None, None, untransformed_walkers, 1
    
        if random_state is not None:
            np.random.seed(random_state)
        
        # Run MCMC
        sampler.run_mcmc(initial_positions, n_steps, progress=True)
        
        # Get samples after burn-in and transform to constrained space
        samples = sampler.get_chain(discard=n_burnin, flat=True)
        print(f"Valid samples shape: {samples.shape}")
        
        # Compute statistics
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
    