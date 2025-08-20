import numpy as np
from scipy.optimize import minimize
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import constraint_transform, inverse_constraint_transform
import emcee
from .base_sampler import BaseSampler


class MCMCSampler(BaseSampler):
    """
    MCMC sampler using emcee for Bayesian inference.
    """
    
    def __init__(self, constraint_manager=None, n_walkers=32, n_steps=300, n_burnin=100):
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
        """
        Initialize walkers in unconstrained space around the transformed MAP estimate.
        """
        constraints_list = constraint_manager.get_constraints_list()
        
        # Create initial positions array with shape (n_walkers, n_params)
        initial_positions = np.zeros((n_walkers, n_params))
        
        # First 3 parameters: MAP estimate + small noise
        for i in range(3):
            param_name = self.constraint_manager.get_param_names()[i]
            # First parameter: MAP estimate + small noise
            initial_positions[:, 0] = map_params[param_name] * (1 + np.random.normal(0, 0.01, n_walkers))
            # Clip to constraints
            initial_positions[:, 0] = np.clip(
                initial_positions[:, 0],
                constraints_list[0][0],
                constraints_list[0][1]
            )

            # Second parameter: uniform distribution
            initial_positions[:, 1] = np.random.uniform(
                constraints_list[1][0],
                constraints_list[1][1],
                n_walkers
            )

            # Third parameter: uniform distribution
            initial_positions[:, 2] = np.random.uniform(
                constraints_list[2][0],
                constraints_list[2][1],
                n_walkers
            )

        # Fourth parameter: uniform distribution
        initial_positions[:, 3] = np.random.uniform(
            constraints_list[3][0],
            constraints_list[3][1], 
            n_walkers
        )
        # Transform to unconstrained space
        initial_positions_transformed = np.array([
            constraint_transform(pos, constraint_manager) for pos in initial_positions
        ])
        
        return initial_positions_transformed
    
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
        
        # Set up the MCMC sampler
        sampler = emcee.EnsembleSampler(
            n_walkers, n_params, self.log_probability,
            args=(x_data, y_data, func, sigma_noise),
            moves=emcee.moves.StretchMove(a=2.0)
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
        
        # Store convergence metrics
        try:
            autocorr_time = sampler.get_autocorr_time()
        except Exception as e:
            print(f"Warning: Could not compute autocorrelation time: {e}")
            autocorr_time = None
            
        self.convergence_metrics = {
            'autocorr_time': autocorr_time,
            'mean_acceptance_fraction': self.acceptance_rate,
            'n_walkers': n_walkers,
            'n_steps': n_steps,
            'n_burnin': n_burnin
        }
        
        return cov_matrix, std_params, samples, self.acceptance_rate
    
    def check_chain_length(self):
        """
        Check if the chain is long enough for reliable autocorrelation time estimation.
        
        Returns:
            is_sufficient: Boolean indicating if chain length is sufficient
            message: Description of the check result
        """
        if self.samples is None:
            return False, "No samples available"
        
        n_samples = len(self.samples)
        
        if self.convergence_metrics.get('autocorr_time') is not None:
            max_autocorr = np.max(self.convergence_metrics['autocorr_time'])
            required_length = 50 * max_autocorr
            
            if n_samples >= required_length:
                return True, f"Chain length ({n_samples}) is sufficient for autocorrelation estimation"
            else:
                return False, f"Chain length ({n_samples}) is too short. Need at least {required_length:.0f} samples"
        else:
            return False, "Autocorrelation time not available"


