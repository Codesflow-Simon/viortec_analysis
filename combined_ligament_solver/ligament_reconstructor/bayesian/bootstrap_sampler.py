import numpy as np
from scipy.optimize import minimize
from typing import Dict, Tuple, Any, List
from .base_sampler import BaseSampler


class BootstrapSampler(BaseSampler):
    """
    Bootstrap sampler for Bayesian inference.
    
    This method uses resampling of the data to estimate parameter uncertainty
    by fitting the model to multiple bootstrap samples of the original data.
    """
    
    def __init__(self, constraint_manager=None):
        """
        Initialize Bootstrap sampler.
        
        Args:
            constraint_manager: Optional ConstraintManager for parameter transformations
        """
        super().__init__(constraint_manager)
    
    def sample(self, map_params, x_data, y_data, func, sigma_noise=1e-3, 
               n_bootstrap=1000, random_state=None, **kwargs):
        """
        Generate samples using bootstrap resampling.
        
        Args:
            map_params: MAP estimate (optimal parameters in constrained space)
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            n_bootstrap: Number of bootstrap samples
            random_state: Random state for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            cov_matrix: Covariance matrix (in constrained space)
            std_params: Standard deviations of parameters (in constrained space)
            samples: Parameter samples (in constrained space)
            acceptance_rate: Always 1.0 for bootstrap
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_data = len(x_data)
        n_params = len(map_params)
        
        # Store bootstrap parameter estimates
        bootstrap_params = []
        
        # Generate bootstrap samples
        for i in range(n_bootstrap):
            # Resample data with replacement
            indices = np.random.choice(n_data, size=n_data, replace=True)
            x_bootstrap = x_data[indices]
            y_bootstrap = y_data[indices]
            
            # Fit model to bootstrap sample
            try:
                params_bootstrap = self._fit_to_bootstrap_sample(
                    x_bootstrap, y_bootstrap, func, sigma_noise
                )
                bootstrap_params.append(params_bootstrap)
            except Exception as e:
                print(f"Bootstrap sample {i} failed: {e}")
                continue
        
        if len(bootstrap_params) == 0:
            raise ValueError("All bootstrap samples failed to converge")
        
        # Convert to numpy array
        bootstrap_params = np.array(bootstrap_params)
        
        # Compute statistics
        cov_matrix = np.cov(bootstrap_params, rowvar=False)
        std_params = np.sqrt(np.diag(cov_matrix))
        
        # Store results
        self.samples = bootstrap_params
        self.covariance_matrix = cov_matrix
        self.parameter_std = std_params
        self.acceptance_rate = 1.0  # Bootstrap always accepts
        
        # Store convergence metrics
        self.convergence_metrics = {
            'n_bootstrap': n_bootstrap,
            'successful_bootstrap_samples': len(bootstrap_params),
            'method': 'bootstrap'
        }
        
        return cov_matrix, std_params, bootstrap_params, 1.0
    
    def _fit_to_bootstrap_sample(self, x_data, y_data, func, sigma_noise):
        """
        Fit model to a bootstrap sample of the data.
        
        Args:
            x_data: Bootstrap sample of input data
            y_data: Bootstrap sample of target data
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            
        Returns:
            params: Fitted parameters in constrained space
        """
        from copy import deepcopy
        func_copy = deepcopy(func)
        
        # Get current parameters as initial guess
        current_params = np.array(list(func_copy.get_params().values()), dtype=float)
        
        # Transform to unconstrained space for optimization
        if self.constraint_manager is not None:
            from ligament_models.transformations import constraint_transform
            initial_params = constraint_transform(current_params, self.constraint_manager)
        else:
            initial_params = current_params
        
        # Define objective function (negative log-likelihood)
        def objective(params):
            return -self.log_likelihood(params, x_data, y_data, func_copy, sigma_noise)
        
        # Optimize
        result = minimize(
            objective, 
            initial_params, 
            method='L-BFGS-B',
            options={'maxiter': 1000}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        # Transform back to constrained space
        if self.constraint_manager is not None:
            from ligament_models.transformations import inverse_constraint_transform
            params = inverse_constraint_transform(result.x, self.constraint_manager)
        else:
            params = result.x
        
        return params
    
    def get_bootstrap_confidence_intervals(self, confidence_level=0.95):
        """
        Get bootstrap confidence intervals for parameters.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            confidence_intervals: Dictionary with lower and upper bounds for each parameter
        """
        if self.samples is None:
            raise ValueError("No bootstrap samples available. Run sample() first.")
        
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for i in range(self.samples.shape[1]):
            param_samples = self.samples[:, i]
            lower = np.percentile(param_samples, lower_percentile)
            upper = np.percentile(param_samples, upper_percentile)
            confidence_intervals[f'param_{i}'] = {'lower': lower, 'upper': upper}
        
        return confidence_intervals
