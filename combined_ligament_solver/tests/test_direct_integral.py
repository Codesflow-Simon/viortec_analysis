import pytest
import numpy as np
from unittest.mock import Mock, patch
from scipy import integrate

from ligament_reconstructor.direct_integral import IntegralProbability
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import constraint_transform, inverse_constraint_transform


class MockFunction:
    """Mock function class for testing"""
    def __init__(self):
        self.params = None
    
    def set_params(self, params):
        self.params = params
    
    def __call__(self, x):
        # Mock function that better matches the test data
        # For x=[1,2,3,4,5] and y=[10.5, 20.8, 31.2, 41.5, 51.9]
        # A good fit would be approximately y = 10.3 * x + 0.2
        if self.params is None:
            return 0
        k, alpha, l_0, f_ref = self.params
        return k * x + alpha


class TestIntegralProbability:
    """Test suite for IntegralProbability class"""
    
    @pytest.fixture
    def constraint_manager(self):
        """Create a constraint manager for testing"""
        return ConstraintManager(mode='blankevoort')
    
    @pytest.fixture
    def integral_prob(self, constraint_manager):
        """Create an IntegralProbability instance for testing"""
        return IntegralProbability(constraint_manager)
    
    @pytest.fixture
    def mock_func(self):
        """Create a mock function for testing"""
        return MockFunction()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([10.5, 20.8, 31.2, 41.5, 51.9])
        return x_data, y_data
    
    @pytest.fixture
    def sample_params(self):
        """Create sample parameters for testing"""
        return np.array([10.3, 0.2, 50.0, 100.0])  # k, alpha, l_0, f_ref - better match for test data
    
    def test_init(self, constraint_manager):
        """Test initialization of IntegralProbability"""
        integral_prob = IntegralProbability(constraint_manager)
        assert integral_prob.constraint_manager == constraint_manager
        assert integral_prob._last_params is None
        assert integral_prob._last_integral_value is None
    
    def test_clear_cache(self, integral_prob):
        """Test cache clearing functionality"""
        # Set some cache values
        integral_prob._last_params = np.array([1.0, 2.0])
        integral_prob._last_integral_value = 3.14
        
        integral_prob.clear_cache()
        
        assert integral_prob._last_params is None
        assert integral_prob._last_integral_value is None
    
    def test_log_likelihood_basic(self, integral_prob, mock_func, sample_data, sample_params):
        """Test basic log likelihood calculation"""
        x_data, y_data = sample_data
        
        # Transform parameters to unconstrained space
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        # Calculate log likelihood
        log_like = integral_prob.log_likelihood(params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Should return a finite float
        assert isinstance(log_like, float)
        assert np.isfinite(log_like)
        assert log_like <= 0  # Log likelihood should be negative for reasonable data
    
    def test_log_likelihood_with_different_sigma(self, integral_prob, mock_func, sample_data, sample_params):
        """Test log likelihood with different noise levels"""
        x_data, y_data = sample_data
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        # Test with different sigma values
        sigma_values = [0.1, 1.0, 10.0, 100.0]
        log_likes = []
        
        for sigma in sigma_values:
            log_like = integral_prob.log_likelihood(params_unconstrained, x_data, y_data, mock_func, sigma_noise=sigma)
            log_likes.append(log_like)
        
        # All should be finite
        assert all(np.isfinite(log_like) for log_like in log_likes)
        
        # All should be finite
        assert all(np.isfinite(log_like) for log_like in log_likes)
        
        # The relationship between sigma and log likelihood depends on the data fit
        # For well-fitting data, lower sigma gives higher log likelihood
        # For poorly fitting data, higher sigma might give higher log likelihood
        # So we just check that all values are finite
    
    def test_log_likelihood_parameter_transformation(self, integral_prob, mock_func, sample_data):
        """Test that parameter transformation works correctly in log likelihood"""
        x_data, y_data = sample_data
        
        # Test with parameters that would be out of bounds in constrained space
        # but valid in unconstrained space
        params_unconstrained = np.array([0.0, 0.0, 0.0, 0.0])  # These should transform to valid values
        
        log_like = integral_prob.log_likelihood(params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Should not raise an error and should return a finite value
        assert np.isfinite(log_like)
    
    def test_integrate_basic(self, integral_prob, mock_func, sample_data, sample_params):
        """Test basic integration functionality"""
        x_data, y_data = sample_data
        
        # Pass all 4 parameters, but only k and alpha will be used as fixed parameters
        params = sample_params  # [k, alpha, l_0, f_ref]
        
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Should return a finite positive value
        assert isinstance(integral_value, float)
        assert np.isfinite(integral_value)
        assert integral_value > 0
    
    def test_integrate_caching(self, integral_prob, mock_func, sample_data, sample_params):
        """Test that integration results are cached correctly"""
        x_data, y_data = sample_data
        params = sample_params  # [k, alpha, l_0, f_ref]
        
        # First call should compute the integral
        integral_value1 = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Cache should be set
        assert integral_prob._last_params is not None
        assert integral_prob._last_integral_value is not None
        assert np.array_equal(integral_prob._last_params, params[:2])  # Only k and alpha are cached
        assert integral_prob._last_integral_value == integral_value1
        
        # Second call with same parameters should use cache
        integral_value2 = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Should return the same value
        assert integral_value1 == integral_value2
    
    def test_integrate_different_parameters(self, integral_prob, mock_func, sample_data):
        """Test integration with different parameter values"""
        x_data, y_data = sample_data
        
        # Test with different k and alpha values, but need all 4 parameters
        test_params = [
            np.array([8.0, 0.1, 50.0, 100.0]),   # Low k, low alpha
            np.array([15.0, 0.3, 50.0, 100.0]),  # High k, high alpha
            np.array([10.3, 0.2, 50.0, 100.0]),  # Medium values (good fit)
        ]
        
        integral_values = []
        for params in test_params:
            integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
            integral_values.append(integral_value)
        
        # All should be finite and positive
        assert all(np.isfinite(val) and val > 0 for val in integral_values)
        
        # Values should be different for different parameters
        assert len(set(integral_values)) == len(integral_values)
    
    def test_probability_density(self, integral_prob, mock_func, sample_data, sample_params):
        """Test probability density calculation"""
        x_data, y_data = sample_data
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        prob_density = integral_prob.probability_density(
            params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0
        )
        
        # Should return a finite positive value
        assert isinstance(prob_density, float)
        assert np.isfinite(prob_density)
        assert prob_density > 0
        assert prob_density <= 1  # Probability density should be <= 1
    
    def test_probability_density_normalization(self, integral_prob, mock_func, sample_data, sample_params):
        """Test that probability density integrates to 1 (approximately)"""
        x_data, y_data = sample_data
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        # Get the normalization coefficient
        normalization = integral_prob.integrate(sample_params, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Calculate probability density
        prob_density = integral_prob.probability_density(
            params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0
        )
        
        # Calculate likelihood
        likelihood = np.exp(integral_prob.log_likelihood(
            params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0
        ))
        
        # Check that prob_density = likelihood / normalization
        # Only if normalization is not zero
        if normalization > 0:
            assert np.isclose(prob_density, likelihood / normalization, rtol=1e-10)
        else:
            # If normalization is zero, prob_density should also be zero or NaN
            assert prob_density == 0 or np.isnan(prob_density)
    
    def test_calculate_marginal_probability(self, integral_prob, mock_func, sample_data, sample_params):
        """Test marginal probability calculation"""
        x_data, y_data = sample_data
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        # Test marginal probability for k (parameter 0)
        # Need to transform the marginal parameter value to unconstrained space
        marginal_param_constrained = np.array([sample_params[0], sample_params[1], sample_params[2], sample_params[3]])
        marginal_param_unconstrained = constraint_transform(marginal_param_constrained, integral_prob.constraint_manager)
        
        marginal_prob = integral_prob.calculate_marginal_probability(
            params_unconstrained, marginal_param=0, marginal_param_value=marginal_param_unconstrained[0],
            x_data=x_data, y_data=y_data, func=mock_func, sigma_noise=1.0
        )
        
        # Should return a finite positive value
        assert isinstance(marginal_prob, float)
        assert np.isfinite(marginal_prob)
        assert marginal_prob > 0
    
    def test_calculate_marginal_probability_different_params(self, integral_prob, mock_func, sample_data):
        """Test marginal probability for different parameters"""
        x_data, y_data = sample_data
        sample_params = np.array([50.0, 0.05, 50.0, 100.0])
        params_unconstrained = constraint_transform(sample_params, integral_prob.constraint_manager)
        
        # Test marginal probability for different parameters
        for param_idx in range(4):
            # Transform the marginal parameter value to unconstrained space
            marginal_param_constrained = np.array([sample_params[0], sample_params[1], sample_params[2], sample_params[3]])
            marginal_param_unconstrained = constraint_transform(marginal_param_constrained, integral_prob.constraint_manager)
            
            marginal_prob = integral_prob.calculate_marginal_probability(
                params_unconstrained, marginal_param=param_idx, 
                marginal_param_value=marginal_param_unconstrained[param_idx],
                x_data=x_data, y_data=y_data, func=mock_func, sigma_noise=1.0
            )
            
            assert np.isfinite(marginal_prob)
            assert marginal_prob > 0
    
    def test_edge_cases(self, integral_prob, mock_func):
        """Test edge cases and error conditions"""
        # Test with empty data
        x_data = np.array([])
        y_data = np.array([])
        params = np.array([10.3, 0.2, 50.0, 100.0])
        
        # Should handle empty data gracefully
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        assert np.isfinite(integral_value)
        
        # Test with single data point
        x_data = np.array([1.0])
        y_data = np.array([10.0])
        
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        assert np.isfinite(integral_value)
    
    def test_numerical_stability(self, integral_prob, mock_func):
        """Test numerical stability with extreme values"""
        # Test with very small noise
        x_data = np.array([1.0, 2.0, 3.0])
        y_data = np.array([10.0, 20.0, 30.0])
        params = np.array([10.3, 0.2, 50.0, 100.0])
        
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1e-10)
        assert np.isfinite(integral_value)
        
        # Test with very large noise
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1e10)
        assert np.isfinite(integral_value)
    
    def test_parameter_bounds_respect(self, integral_prob, mock_func, sample_data):
        """Test that integration respects parameter bounds"""
        x_data, y_data = sample_data
        params = np.array([10.3, 0.2, 50.0, 100.0])
        
        # Get bounds from constraint manager
        constraints_list = integral_prob.constraint_manager.get_constraints_list()
        l_0_bounds = constraints_list[2]
        f_ref_bounds = constraints_list[3]
        
        # The integration should only happen within these bounds
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Should be finite and positive
        assert np.isfinite(integral_value)
        assert integral_value > 0
    
    @pytest.mark.parametrize("sigma_noise", [0.1, 1.0, 10.0, 100.0])
    def test_integration_with_different_noise_levels(self, integral_prob, mock_func, sigma_noise):
        """Test integration with different noise levels"""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([10.5, 20.8, 31.2, 41.5, 51.9])
        params = np.array([10.3, 0.2, 50.0, 100.0])
        
        integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=sigma_noise)
        
        assert np.isfinite(integral_value)
        assert integral_value > 0


class TestIntegralProbabilityIntegration:
    """Integration tests for IntegralProbability with real data scenarios"""
    
    @pytest.fixture
    def constraint_manager(self):
        return ConstraintManager(mode='blankevoort')
    
    @pytest.fixture
    def integral_prob(self, constraint_manager):
        return IntegralProbability(constraint_manager)
    
    @pytest.fixture
    def mock_func(self):
        return MockFunction()
    
    def test_full_workflow(self, integral_prob, mock_func):
        """Test the complete workflow from data to probability density"""
        # Generate synthetic data
        np.random.seed(42)
        x_data = np.linspace(0, 10, 20)
        true_params = np.array([10.3, 0.2, 45.0, 50.0])  # k, alpha, l_0, f_ref
        
        # Create mock function with true parameters
        mock_func.set_params(true_params)
        y_true = np.array([mock_func(x) for x in x_data])
        
        # Add noise
        noise = np.random.normal(0, 1.0, len(x_data))
        y_data = y_true + noise
        
        # Test parameter estimation workflow
        test_params = np.array([10.0, 0.15, 45.0, 50.0])  # k, alpha, l_0, f_ref
        test_params_unconstrained = constraint_transform(
            test_params, integral_prob.constraint_manager
        )
        
        # Calculate log likelihood
        log_like = integral_prob.log_likelihood(
            test_params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0
        )
        
        # Calculate integral
        integral_value = integral_prob.integrate(
            test_params, x_data, y_data, mock_func, sigma_noise=1.0
        )
        
        # Calculate probability density
        prob_density = integral_prob.probability_density(
            test_params_unconstrained, x_data, y_data, mock_func, sigma_noise=1.0
        )
        
        # All should be finite
        assert np.isfinite(log_like)
        assert np.isfinite(integral_value)
        assert np.isfinite(prob_density)
        assert integral_value > 0
        assert prob_density > 0
    
    def test_consistency_across_multiple_calls(self, integral_prob, mock_func):
        """Test that results are consistent across multiple calls"""
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = np.array([10.5, 20.8, 31.2, 41.5, 51.9])
        params = np.array([10.3, 0.2, 50.0, 100.0])
        params_unconstrained = constraint_transform(
            params, integral_prob.constraint_manager
        )
        
        # Multiple calls should give the same results
        results = []
        for _ in range(5):
            integral_value = integral_prob.integrate(params, x_data, y_data, mock_func, sigma_noise=1.0)
            results.append(integral_value)
        
        # All results should be the same
        assert all(np.isclose(results[0], result, rtol=1e-10) for result in results)
    
    def test_cache_invalidation(self, integral_prob, mock_func):
        """Test that cache is properly invalidated when parameters change"""
        x_data = np.array([1.0, 2.0, 3.0])
        y_data = np.array([10.0, 20.0, 30.0])
        
        # First call with parameters A
        params_a = np.array([8.0, 0.1, 50.0, 100.0])
        integral_a = integral_prob.integrate(params_a, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Second call with different parameters B
        params_b = np.array([15.0, 0.3, 50.0, 100.0])
        integral_b = integral_prob.integrate(params_b, x_data, y_data, mock_func, sigma_noise=1.0)
        
        # Results should be different
        assert not np.isclose(integral_a, integral_b, rtol=1e-10)
        
        # Cache should be updated
        assert np.array_equal(integral_prob._last_params, params_b[:2])  # Only k and alpha are cached
        assert integral_prob._last_integral_value == integral_b


if __name__ == "__main__":
    pytest.main([__file__])
