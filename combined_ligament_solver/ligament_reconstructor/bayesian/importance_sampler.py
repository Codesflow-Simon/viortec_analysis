import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Tuple, Any
from .base_sampler import BaseSampler


class ImportanceSampler(BaseSampler):
    """
    Importance Sampling sampler for Bayesian inference.
    
    This method samples from a proposal distribution and reweights the samples
    to approximate the posterior distribution.
    """
    
    def __init__(self, constraint_manager=None, proposal_scale=1.0):
        """
        Initialize Importance sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
            proposal_scale: Scale factor for the proposal distribution
        """
        super().__init__(constraint_manager)
        self.proposal_scale = proposal_scale
    
    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, 
               n_samples=1000, random_state=None, **kwargs):
        """
        Generate samples using importance sampling.
        
        Args:
            map_params: MAP estimate (optimal parameters in constrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            n_samples: Number of samples to generate
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            cov_matrix: Covariance matrix (in constrained space)
            std_params: Standard deviations of parameters (in constrained space)
            samples: Parameter samples (in constrained space)
            acceptance_rate: Always 1.0 for importance sampling
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        # Convert MAP parameters to array
        map_params_array = np.array(list(map_params.values()), dtype=float)
        
        # Transform to unconstrained space
        if self.constraint_manager is not None:
            from ligament_models.transformations import constraint_transform
            map_params_unconstrained = constraint_transform(map_params_array, self.constraint_manager)
        else:
            map_params_unconstrained = map_params_array
        
        # Define proposal distribution (Gaussian centered at MAP)
        n_params = len(map_params_unconstrained)
        proposal_mean = map_params_unconstrained
        proposal_cov = self.proposal_scale * np.eye(n_params)
        
        # Sample from proposal distribution
        samples_unconstrained = np.random.multivariate_normal(
            proposal_mean, 
            proposal_cov, 
            size=n_samples
        )
        
        # Compute importance weights
        weights = self._compute_importance_weights(
            samples_unconstrained, proposal_mean, proposal_cov, 
            x_data, y_data, func, sigma_noise
        )
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Resample with replacement based on weights
        resampled_indices = np.random.choice(
            n_samples, size=n_samples, p=weights, replace=True
        )
        resampled_samples_unconstrained = samples_unconstrained[resampled_indices]
        
        # Transform samples back to constrained space
        if self.constraint_manager is not None:
            from ligament_models.transformations import inverse_constraint_transform
            samples = np.array([inverse_constraint_transform(s, self.constraint_manager) 
                               for s in resampled_samples_unconstrained])
        else:
            samples = resampled_samples_unconstrained
        
        # Compute statistics in constrained space
        cov_matrix = np.cov(samples, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = samples
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = 1.0  # Importance sampling always accepts
        
        # Store convergence metrics
        self.convergence_metrics = {
            'effective_sample_size': self._compute_effective_sample_size(weights),
            'max_weight': np.max(weights),
            'min_weight': np.min(weights),
            'weight_std': np.std(weights),
            'n_samples': n_samples,
            'method': 'importance_sampling'
        }
        
        return cov_matrix, std_params, samples, 1.0
    
    def _compute_importance_weights(self, samples, proposal_mean, proposal_cov, 
                                  x_data, y_data, func, sigma_noise):
        """
        Compute importance weights for samples.
        
        Args:
            samples: Samples from proposal distribution
            proposal_mean: Mean of proposal distribution
            proposal_cov: Covariance of proposal distribution
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            weights: Importance weights
        """
        n_samples = len(samples)
        weights = np.zeros(n_samples)
        
        # Define proposal distribution
        proposal_dist = multivariate_normal(proposal_mean, proposal_cov)
        
        for i, sample in enumerate(samples):
            # Compute log-posterior (target distribution)
            log_target = self.log_probability(sample, x_data, y_data, func, sigma_noise)
            
            # Compute log-proposal
            log_proposal = proposal_dist.logpdf(sample)
            
            # Compute importance weight (up to normalization)
            weights[i] = np.exp(log_target - log_proposal)
        
        return weights
    
    def _compute_effective_sample_size(self, weights):
        """
        Compute effective sample size based on importance weights.
        
        Args:
            weights: Normalized importance weights
            
        Returns:
            ess: Effective sample size
        """
        ess = 1.0 / np.sum(weights**2)
        return ess
    
    def get_importance_weights(self):
        """
        Get the importance weights from the last sampling run.
        
        Returns:
            weights: Importance weights (if available)
        """
        if hasattr(self, '_last_weights'):
            return self._last_weights
        else:
            return None
    
    def set_proposal_scale(self, scale):
        """
        Set the scale factor for the proposal distribution.
        
        Args:
            scale: New scale factor
        """
        self.proposal_scale = scale
