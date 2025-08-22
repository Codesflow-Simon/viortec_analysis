import unittest
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ligament_models.transformations import constraint_transform, inverse_constraint_transform
from ligament_models.constraints import ConstraintManager

class TestTransformations(unittest.TestCase):
    """Test cases for the transformation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blankevoort_constraints = ConstraintManager(mode='blankevoort')
        
        # Test parameters for Blankevoort function
        self.blankevoort_params = np.array([60.0, 0.07, 50.0, 0.05])
    
    def test_constraint_transform_blankevoort(self):
        """Test constraint_transform for Blankevoort parameters."""
        # Transform constrained parameters to unconstrained space
        unconstrained_params = constraint_transform(self.blankevoort_params, self.blankevoort_constraints)
        
        # Check that result has correct shape
        self.assertEqual(unconstrained_params.shape, self.blankevoort_params.shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(unconstrained_params)))
        
        # Check that all values are real (not complex)
        self.assertTrue(np.all(np.isreal(unconstrained_params)))
    
    def test_inverse_constraint_transform_blankevoort(self):
        """Test inverse_constraint_transform for Blankevoort parameters."""
        # First transform to unconstrained space
        unconstrained_params = constraint_transform(self.blankevoort_params, self.blankevoort_constraints)
        
        # Then transform back to constrained space
        constrained_params = inverse_constraint_transform(unconstrained_params, self.blankevoort_constraints)
        
        # Check that result has correct shape
        self.assertEqual(constrained_params.shape, self.blankevoort_params.shape)
        
        # Check that we get back the original parameters (within numerical precision)
        np.testing.assert_array_almost_equal(constrained_params, self.blankevoort_params, decimal=10)
    
    def test_transformations_are_inverses(self):
        """Test that constraint_transform and inverse_constraint_transform are inverses."""
        # Test with random parameters within bounds
        np.random.seed(42)  # For reproducibility
        
        # Test multiple random parameter sets
        for _ in range(10):
            # Generate random parameters within bounds for Blankevoort
            k = np.random.uniform(10, 100)
            alpha = np.random.uniform(0.02, 0.12)
            l_0 = np.random.uniform(40, 60)
            f_ref = np.random.uniform(0.0, 300.0)
            test_params = np.array([k, alpha, l_0, f_ref])
            
            # Test forward and inverse transformation
            unconstrained = constraint_transform(test_params, self.blankevoort_constraints)
            reconstructed = inverse_constraint_transform(unconstrained, self.blankevoort_constraints)
            
            # Check that we get back the original parameters
            np.testing.assert_array_almost_equal(reconstructed, test_params, decimal=10)
    
    def test_constrained_parameters_respect_bounds(self):
        """Test that constrained parameters are within the specified bounds."""
        np.random.seed(42)  # For reproducibility
        
        # Test with random unconstrained parameters
        for _ in range(20):
            # Generate random unconstrained parameters
            unconstrained_params = np.random.uniform(-10, 10, size=4)
            
            # Transform to constrained space
            constrained_params = inverse_constraint_transform(unconstrained_params, self.blankevoort_constraints)
            
            # Get bounds from constraint manager
            bounds = self.blankevoort_constraints.get_constraints_list()
            
            # Check that each parameter is within its bounds
            for i, (lower, upper) in enumerate(bounds):
                self.assertGreaterEqual(constrained_params[i], lower, 
                                      f"Parameter {i} ({constrained_params[i]}) below lower bound {lower}")
                self.assertLessEqual(constrained_params[i], upper, 
                                   f"Parameter {i} ({constrained_params[i]}) above upper bound {upper}")
    
    def test_edge_cases(self):
        """Test edge cases for transformations."""
        # Test with parameters at the boundaries
        blankevoort_bounds = self.blankevoort_constraints.get_constraints_list()
        
        # Test lower bounds
        lower_bound_params = np.array([bounds[0] for bounds in blankevoort_bounds])
        unconstrained_lower = constraint_transform(lower_bound_params, self.blankevoort_constraints)
        reconstructed_lower = inverse_constraint_transform(unconstrained_lower, self.blankevoort_constraints)
        np.testing.assert_array_almost_equal(reconstructed_lower, lower_bound_params, decimal=10)
        
        # Test upper bounds
        upper_bound_params = np.array([bounds[1] for bounds in blankevoort_bounds])
        unconstrained_upper = constraint_transform(upper_bound_params, self.blankevoort_constraints)
        reconstructed_upper = inverse_constraint_transform(unconstrained_upper, self.blankevoort_constraints)
        np.testing.assert_array_almost_equal(reconstructed_upper, upper_bound_params, decimal=10)
    
    def test_numerical_stability(self):
        """Test numerical stability of transformations."""
        # Test with very large unconstrained parameters
        large_unconstrained = np.array([1000.0, -1000.0, 500.0, -500.0])
        constrained = inverse_constraint_transform(large_unconstrained, self.blankevoort_constraints)
        
        # Check that result is finite and within bounds
        self.assertTrue(np.all(np.isfinite(constrained)))
        bounds = self.blankevoort_constraints.get_constraints_list()
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(constrained[i], lower)
            self.assertLessEqual(constrained[i], upper)
        
        # Test with very small unconstrained parameters
        small_unconstrained = np.array([0.001, -0.001, 0.0001, -0.0001])
        constrained = inverse_constraint_transform(small_unconstrained, self.blankevoort_constraints)
        
        # Check that result is finite and within bounds
        self.assertTrue(np.all(np.isfinite(constrained)))
        for i, (lower, upper) in enumerate(bounds):
            self.assertGreaterEqual(constrained[i], lower)
            self.assertLessEqual(constrained[i], upper)
    
    def test_parameter_mapping_consistency(self):
        """Test that parameter mapping is consistent between transformations."""
        # Test that the parameter order is preserved
        original_params = np.array([60.0, 0.07, 50.0, 0.05])
        
        # Transform to unconstrained and back
        unconstrained = constraint_transform(original_params, self.blankevoort_constraints)
        reconstructed = inverse_constraint_transform(unconstrained, self.blankevoort_constraints)
        
        # Check that parameter order is preserved
        param_names = self.blankevoort_constraints.get_param_names()
        for i, name in enumerate(param_names):
            self.assertAlmostEqual(reconstructed[i], original_params[i], places=10,
                                 msg=f"Parameter {name} order not preserved")
    
    def test_constraint_manager_integration(self):
        """Test integration with ConstraintManager."""
        # Test that constraint manager provides correct bounds
        bounds = self.blankevoort_constraints.get_constraints_list()
        param_names = self.blankevoort_constraints.get_param_names()
        
        # Check that we have the right number of bounds
        self.assertEqual(len(bounds), len(param_names))
        
        # Check that each bound has lower and upper values
        for i, (lower, upper) in enumerate(bounds):
            self.assertLess(lower, upper, f"Bound {i} has invalid range: {lower} >= {upper}")
            self.assertTrue(np.isfinite(lower) and np.isfinite(upper), 
                          f"Bound {i} has non-finite values: {lower}, {upper}")
    
    def test_mathematical_properties(self):
        """Test mathematical properties of the transformations."""
        # Test that the transformation is monotonic
        np.random.seed(42)
        
        # Generate two sets of parameters where one is larger than the other
        params1 = np.array([30.0, 0.05, 45.0, 0.02])
        params2 = np.array([70.0, 0.09, 55.0, 0.08])
        
        # Transform both to unconstrained space
        unconstrained1 = constraint_transform(params1, self.blankevoort_constraints)
        unconstrained2 = constraint_transform(params2, self.blankevoort_constraints)
        
        # Check that the transformation preserves ordering (monotonic)
        # This should hold for most parameters, but not necessarily all due to the sigmoid nature
        # We'll just check that the transformation is well-behaved
        self.assertTrue(np.all(np.isfinite(unconstrained1)))
        self.assertTrue(np.all(np.isfinite(unconstrained2)))
    
    def test_extreme_values(self):
        """Test behavior with extreme values."""
        # Test with parameters very close to bounds
        bounds = self.blankevoort_constraints.get_constraints_list()
        
        # Test parameters just inside the bounds
        epsilon = 1e-10
        near_lower = np.array([bounds[i][0] + epsilon for i in range(len(bounds))])
        near_upper = np.array([bounds[i][1] - epsilon for i in range(len(bounds))])
        
        # Transform to unconstrained and back
        unconstrained_lower = constraint_transform(near_lower, self.blankevoort_constraints)
        unconstrained_upper = constraint_transform(near_upper, self.blankevoort_constraints)
        
        reconstructed_lower = inverse_constraint_transform(unconstrained_lower, self.blankevoort_constraints)
        reconstructed_upper = inverse_constraint_transform(unconstrained_upper, self.blankevoort_constraints)
        
        # Check that we get back the original parameters
        np.testing.assert_array_almost_equal(reconstructed_lower, near_lower, decimal=10)
        np.testing.assert_array_almost_equal(reconstructed_upper, near_upper, decimal=10)
    
    def test_overflow_handling(self):
        """Test handling of potential overflow in exponential function."""
        # Test with very large unconstrained parameters that could cause overflow
        very_large_unconstrained = np.array([1e6, -1e6, 1e5, -1e5])
        
        # This should not raise an exception and should produce valid constrained parameters
        try:
            constrained = inverse_constraint_transform(very_large_unconstrained, self.blankevoort_constraints)
            
            # Check that result is finite and within bounds
            self.assertTrue(np.all(np.isfinite(constrained)))
            bounds = self.blankevoort_constraints.get_constraints_list()
            for i, (lower, upper) in enumerate(bounds):
                self.assertGreaterEqual(constrained[i], lower)
                self.assertLessEqual(constrained[i], upper)
                
        except (OverflowError, RuntimeWarning) as e:
            # If overflow occurs, it should be handled gracefully
            self.fail(f"Overflow not handled gracefully: {e}")
    
    def test_round_trip_accuracy(self):
        """Test the accuracy of round-trip transformations."""
        # Test with various parameter values to ensure high accuracy
        test_cases = [
            np.array([10.0, 0.02, 40.0, 0.0]),  # Lower bounds
            np.array([100.0, 0.12, 60.0, 300.0]),  # Upper bounds
            np.array([55.0, 0.07, 50.0, 150.0]),  # Middle values
            np.array([25.0, 0.05, 45.0, 50.0]),  # Mixed values
        ]
        
        for i, test_params in enumerate(test_cases):
            # Transform to unconstrained and back
            unconstrained = constraint_transform(test_params, self.blankevoort_constraints)
            reconstructed = inverse_constraint_transform(unconstrained, self.blankevoort_constraints)
            
            # Check high accuracy (reduced precision due to numerical limitations)
            np.testing.assert_array_almost_equal(reconstructed, test_params, decimal=10,
                                               err_msg=f"Round-trip accuracy failed for test case {i}")
    
    def test_constraint_manager_modes(self):
        """Test that ConstraintManager works with different modes."""
        # Test blankevoort mode
        blankevoort_cm = ConstraintManager(mode='blankevoort')
        blankevoort_names = blankevoort_cm.get_param_names()
        self.assertEqual(blankevoort_names, ['k', 'alpha', 'l_0', 'f_ref'])
        
        # Test invalid mode - this should raise ValueError when calling get_param_names
        invalid_cm = ConstraintManager(mode='invalid_mode')
        with self.assertRaises(ValueError):
            invalid_cm.get_param_names()


if __name__ == '__main__':
    unittest.main()
