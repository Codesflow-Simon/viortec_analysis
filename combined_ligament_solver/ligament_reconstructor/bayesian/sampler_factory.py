from typing import Dict, Any, Optional
from ligament_models.constraints import ConstraintManager

from .base_sampler import BaseSampler
from .mcmc import MCMCSampler
from .laplace_sampler import LaplaceSampler
from .bootstrap_sampler import BootstrapSampler
from .variational_sampler import VariationalSampler
from .importance_sampler import ImportanceSampler


class SamplerFactory:
    """
    Factory class for creating different Bayesian sampling methods.
    
    This class provides a convenient interface for creating and configuring
    different sampling algorithms with sensible defaults.
    """
    
    AVAILABLE_METHODS = {
        'mcmc': MCMCSampler,
        'laplace': LaplaceSampler,
        'bootstrap': BootstrapSampler,
        'variational': VariationalSampler,
        'importance': ImportanceSampler
    }
    
    @classmethod
    def create_sampler(cls, method: str, constraint_manager: Optional[ConstraintManager] = None, 
                      **kwargs) -> BaseSampler:
        """
        Create a sampler instance for the specified method.
        
        Args:
            method: Sampling method ('mcmc', 'laplace', 'bootstrap', 'variational', 'importance')
            constraint_manager: Optional ConstraintManager for parameter transformations
            **kwargs: Method-specific parameters
            
        Returns:
            sampler: Configured sampler instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method not in cls.AVAILABLE_METHODS:
            available = ', '.join(cls.AVAILABLE_METHODS.keys())
            raise ValueError(f"Unknown method '{method}'. Available methods: {available}")
        
        sampler_class = cls.AVAILABLE_METHODS[method]
        return sampler_class(constraint_manager=constraint_manager, **kwargs)
    
    @classmethod
    def get_default_parameters(cls, method: str) -> Dict[str, Any]:
        """
        Get default parameters for a sampling method.
        
        Args:
            method: Sampling method
            
        Returns:
            params: Dictionary of default parameters
        """
        defaults = {
            'mcmc': {
                'n_walkers': 64,
                'n_steps': 200,
                'n_burnin': 40
            },
            'laplace': {
                'n_samples': 1000
            },
            'bootstrap': {
                'n_bootstrap': 1000
            },
            'variational': {
                'learning_rate': 0.01,
                'max_iter': 1000,
                'n_samples': 1000
            },
            'importance': {
                'proposal_scale': 1.0,
                'n_samples': 1000
            }
        }
        
        return defaults.get(method, {})
    
    @classmethod
    def list_available_methods(cls) -> list:
        """
        Get list of available sampling methods.
        
        Returns:
            methods: List of available method names
        """
        return list(cls.AVAILABLE_METHODS.keys())
    
    @classmethod
    def get_method_description(cls, method: str) -> str:
        """
        Get description of a sampling method.
        
        Args:
            method: Sampling method
            
        Returns:
            description: Method description
        """
        descriptions = {
            'mcmc': 'Markov Chain Monte Carlo using emcee ensemble sampler',
            'laplace': 'Laplace approximation with Gaussian posterior',
            'bootstrap': 'Bootstrap resampling for uncertainty estimation',
            'variational': 'Variational inference with Gaussian approximation',
            'importance': 'Importance sampling with proposal distribution'
        }
        
        return descriptions.get(method, 'No description available')
    
    @classmethod
    def compare_methods(cls, map_params, x_data, y_data, func, sigma_noise=1e-3,
                       constraint_manager=None, methods=None, **kwargs):
        """
        Compare multiple sampling methods on the same problem.
        
        Args:
            map_params: MAP estimate
            x_data: Input data points
            y_data: Target data points
            func: Function to evaluate
            sigma_noise: Noise standard deviation
            constraint_manager: Optional ConstraintManager
            methods: List of methods to compare (default: all available)
            **kwargs: Additional parameters for each method
            
        Returns:
            results: Dictionary with results for each method
        """
        if methods is None:
            methods = cls.list_available_methods()
        
        results = {}
        
        for method in methods:
            try:
                print(f"Running {method} sampler...")
                
                # Create sampler with default parameters
                default_params = cls.get_default_parameters(method)
                sampler_params = {**default_params, **kwargs}
                
                sampler = cls.create_sampler(method, constraint_manager, **sampler_params)
                
                # Run sampling
                cov_matrix, std_params, samples, acceptance_rate = sampler.sample(
                    map_params, x_data, y_data, func, sigma_noise
                )
                
                # Store results
                results[method] = {
                    'covariance_matrix': cov_matrix,
                    'parameter_std': std_params,
                    'samples': samples,
                    'acceptance_rate': acceptance_rate,
                    'convergence_metrics': sampler.get_convergence_metrics(),
                    'success': True
                }
                
                print(f"{method} completed successfully")
                
            except Exception as e:
                print(f"{method} failed: {e}")
                results[method] = {
                    'success': False,
                    'error': str(e)
                }
        
        return results
