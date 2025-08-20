import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from typing import Dict, Tuple, Any
from .base_sampler import BaseSampler


class VariationalSampler(BaseSampler):
    """
    Variational Inference sampler for Bayesian inference.
    
    This method approximates the posterior distribution with a Gaussian
    by minimizing the KL divergence between the approximate and true posterior.
    """
    
    def __init__(self, constraint_manager=None, learning_rate=0.01, max_iter=1000):
        """
        Initialize Variational sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
            learning_rate: Learning rate for optimization
            max_iter: Maximum number of optimization iterations
        """
        super().__init__(constraint_manager)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
    
    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, 
               n_samples=1000, random_state=None, **kwargs):
        """
        Generate samples using variational inference.
        
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
            acceptance_rate: Always 1.0 for variational inference
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
        
        # Initialize variational parameters
        n_params = len(map_params_unconstrained)
        mu = map_params_unconstrained.copy()  # Mean
        # Use smaller initial standard deviations to prevent overflow
        log_sigma = np.log(0.01 * np.ones(n_params))  # Log standard deviations
        
        # Optimize variational parameters
        mu_opt, log_sigma_opt = self._optimize_variational_parameters(
            mu, log_sigma, x_data, y_data, func, sigma_noise
        )
        
        # Construct covariance matrix
        sigma_opt = np.exp(log_sigma_opt)
        cov_matrix_unconstrained = np.diag(sigma_opt**2)
        
        # Generate samples in unconstrained space
        samples_unconstrained = np.random.multivariate_normal(
            mu_opt, 
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
        self.acceptance_rate = 1.0  # Variational inference always accepts
        
        # Store convergence metrics
        self.convergence_metrics = {
            'variational_mean': mu_opt,
            'variational_std': sigma_opt,
            'n_samples': n_samples,
            'method': 'variational_inference'
        }
        
        return cov_matrix, std_params, samples, 1.0
    
    def _optimize_variational_parameters(self, mu_init, log_sigma_init, x_data, y_data, func, sigma_noise):
        """
        Optimize variational parameters using stochastic gradient descent.
        
        Args:
            mu_init: Initial mean parameters
            log_sigma_init: Initial log standard deviation parameters
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            mu_opt: Optimized mean parameters
            log_sigma_opt: Optimized log standard deviation parameters
        """
        mu = mu_init.copy()
        log_sigma = log_sigma_init.copy()
        
        n_params = len(mu)
        
        for iteration in range(self.max_iter):
            # Generate samples for Monte Carlo estimation
            n_mc_samples = 10
            epsilon = np.random.normal(0, 1, (n_mc_samples, n_params))
            sigma = np.exp(log_sigma)
            
            # Compute gradients using Monte Carlo estimation
            mu_grad = np.zeros(n_params)
            log_sigma_grad = np.zeros(n_params)
            
            for i in range(n_mc_samples):
                # Sample from variational distribution
                theta = mu + sigma * epsilon[i]
                
                # Compute log-posterior
                log_post = self.log_probability(theta, x_data, y_data, func, sigma_noise)
                
                # Compute gradients
                mu_grad += log_post * epsilon[i] / sigma
                log_sigma_grad += log_post * (epsilon[i]**2 - 1)
            
            # Average gradients
            mu_grad /= n_mc_samples
            log_sigma_grad /= n_mc_samples
            
            # Update parameters
            mu += self.learning_rate * mu_grad
            log_sigma += self.learning_rate * log_sigma_grad
            
            # Ensure log_sigma doesn't become too small
            log_sigma = np.maximum(log_sigma, -10)
            
            # Print progress every 100 iterations
            if iteration % 100 == 0:
                elbo = self._compute_elbo(mu, log_sigma, x_data, y_data, func, sigma_noise)
                print(f"Iteration {iteration}, ELBO: {elbo:.4f}")
        
        return mu, log_sigma
    
    def _compute_elbo(self, mu, log_sigma, x_data, y_data, func, sigma_noise, n_samples=100):
        """
        Compute Evidence Lower BOund (ELBO).
        
        Args:
            mu: Mean parameters
            log_sigma: Log standard deviation parameters
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            n_samples: Number of Monte Carlo samples
            
        Returns:
            elbo: Evidence Lower BOund value
        """
        sigma = np.exp(log_sigma)
        n_params = len(mu)
        
        # Monte Carlo estimation of expected log-likelihood
        expected_log_likelihood = 0
        for _ in range(n_samples):
            epsilon = np.random.normal(0, 1, n_params)
            theta = mu + sigma * epsilon
            expected_log_likelihood += self.log_likelihood(theta, x_data, y_data, func, sigma_noise)
        expected_log_likelihood /= n_samples
        
        # KL divergence between variational distribution and prior
        # Assuming standard normal prior
        kl_divergence = 0.5 * np.sum(mu**2 + sigma**2 - 2*log_sigma - 1)
        
        elbo = expected_log_likelihood - kl_divergence
        return elbo
