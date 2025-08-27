import unittest
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ligament_models.transformations import constraint_transform, inverse_constraint_transform, sliding_operation
from ligament_models.constraints import ConstraintManager
from ligament_models import BlankevoortFunction

class TestTransformations(unittest.TestCase):
    """Test cases for the transformation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.blankevoort_constraints = ConstraintManager(mode='blankevoort')
        
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
            for i in range(len(reconstructed)):
                assert abs(reconstructed[i] - test_params[i]) < 1e-10
    
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
    
    def test_sliding_operation_identity(self):
        """Test that sliding by a then -a is the identity operation."""
        # Test parameters
        original_params = {
            'k': 60.0,
            'alpha': 0.07,
            'l_0': 50.0,
            'f_ref': 150.0
        }
        
        # Test with different slide factors
        slide_factors = [1.0, -2.5, 5.0, -10.0, 0.5]
        
        for slide_factor in slide_factors:
            # Slide by slide_factor
            slid_params = sliding_operation(original_params.copy(), slide_factor)
            
            # Slide back by -slide_factor
            restored_params = sliding_operation(slid_params.copy(), -slide_factor)
            
            # Check that we get back the original parameters
            self.assertAlmostEqual(restored_params['k'], original_params['k'], places=10)
            self.assertAlmostEqual(restored_params['alpha'], original_params['alpha'], places=10)
            self.assertAlmostEqual(restored_params['l_0'], original_params['l_0'], places=10)
            self.assertAlmostEqual(restored_params['f_ref'], original_params['f_ref'], places=10)
    
    def test_sliding_operation_mechanics(self):
        """Test the mechanics of sliding operation."""
        original_params = {
            'k': 60.0,
            'alpha': 0.07,
            'l_0': 50.0,
            'f_ref': 150.0
        }
        
        slide_factor = 2.0
        
        # Apply sliding operation
        slid_params = sliding_operation(original_params.copy(), slide_factor)
        
        # Check that l_0 is increased by slide_factor
        self.assertAlmostEqual(slid_params['l_0'], original_params['l_0'] + slide_factor, places=10)
        
        # Check that f_ref is decreased by k * slide_factor * (1 + alpha/2)
        expected_f_ref = original_params['f_ref'] - original_params['k'] * slide_factor * (1 + original_params['alpha']/2)
        self.assertAlmostEqual(slid_params['f_ref'], expected_f_ref, places=10)
        
        # Check that k and alpha remain unchanged
        self.assertAlmostEqual(slid_params['k'], original_params['k'], places=10)
        self.assertAlmostEqual(slid_params['alpha'], original_params['alpha'], places=10)
    
    def test_sliding_operation_loss_invariance(self):
        """Test that sliding preserves force differences when all data > l_0."""
        # This test verifies that sliding by -a preserves the relative force differences
        # when all data points are greater than l_0
        
        # Test parameters
        params = {
            'k':60.0,
            'alpha': 0.15,
            'l_0': 50.0,
            'f_ref': 150.0
        }

        # Create test data where all values are in the linear region
        # The linear region starts at l_0 + l_0*alpha = l_0*(1+alpha)
        transition_point = params['l_0'] * (1 + params['alpha'])  # This is where linear region starts
        test_data = np.array([transition_point + 5.0, transition_point + 10.0, transition_point + 15.0, transition_point + 20.0, transition_point + 25.0])
        
        # Calculate force for original parameters using BlankevoortFunction
        original_params_array = np.array([params['k'], params['alpha'], params['l_0'], params['f_ref']])
        blankevoort_func = BlankevoortFunction(original_params_array)
        original_forces = blankevoort_func(test_data)
        
        # Test with different slide factors
        slide_factors = [-5.0, -10.0, -15.0, -20.0, -25.0]
        for slide_factor in slide_factors:
            slid_params = sliding_operation(params.copy(), slide_factor)
            
            # Calculate force for slid parameters using BlankevoortFunction
            slid_params_array = np.array([slid_params['k'], slid_params['alpha'], slid_params['l_0'], slid_params['f_ref']])
            slid_blankevoort_func = BlankevoortFunction(slid_params_array)
            slid_forces = slid_blankevoort_func(test_data)
            
            # Plot original forces, slid forces and test data points
            max_diff = np.max(np.abs(slid_forces - original_forces))
            
            # The forces should be identical in the linear region
            self.assertLess(max_diff, 1e-10, f"Sliding operation should preserve forces in linear region. Max diff: {max_diff}")
            
            # Verify that the sliding operation worked correctly
            self.assertAlmostEqual(slid_params['l_0'], params['l_0'] + slide_factor, places=10)

            # The correct f_ref adjustment accounts for the fact that transition_length = l_0 * alpha changes
            expected_f_ref = params['f_ref'] - params['k'] * slide_factor * (1 + params['alpha']/2)
            self.assertAlmostEqual(slid_params['f_ref'], expected_f_ref, places=10)
    
    def test_sliding_operation_edge_cases(self):
        """Test edge cases for sliding operation."""
        # Test with zero slide factor
        params = {'k': 60.0, 'alpha': 0.07, 'l_0': 50.0, 'f_ref': 150.0}
        result = sliding_operation(params.copy(), 0.0)
        
        # Should be identical to original
        self.assertAlmostEqual(result['l_0'], params['l_0'], places=10)
        self.assertAlmostEqual(result['f_ref'], params['f_ref'], places=10)
        self.assertAlmostEqual(result['k'], params['k'], places=10)
        self.assertAlmostEqual(result['alpha'], params['alpha'], places=10)
        
        # Test with very large slide factor
        large_slide = 1000.0
        result = sliding_operation(params.copy(), large_slide)
        
        self.assertAlmostEqual(result['l_0'], params['l_0'] + large_slide, places=10)
        self.assertAlmostEqual(result['f_ref'], params['f_ref'] - params['k'] * large_slide * (1 + params['alpha']/2), places=10)
        
        # Test with negative slide factor
        negative_slide = -5.0
        result = sliding_operation(params.copy(), negative_slide)
        
        self.assertAlmostEqual(result['l_0'], params['l_0'] + negative_slide, places=10)
        self.assertAlmostEqual(result['f_ref'], params['f_ref'] - params['k'] * negative_slide * (1 + params['alpha']/2), places=10)

    def test_sliding_copy(self):
        """Test that sliding_operation creates a copy and doesn't modify original."""
        params = {'k': 60.0, 'alpha': 0.07, 'l_0': 50.0, 'f_ref': 150.0}
        original = params.copy()
        
        # Perform sliding operation
        slide_factor = 10.0
        result = sliding_operation(params, slide_factor)
        
        # Check original is unmodified
        self.assertEqual(params['k'], original['k'])
        self.assertEqual(params['alpha'], original['alpha']) 
        self.assertEqual(params['l_0'], original['l_0'])
        self.assertEqual(params['f_ref'], original['f_ref'])
        
        # Check result is modified correctly
        self.assertNotEqual(result['l_0'], original['l_0'])
        self.assertNotEqual(result['f_ref'], original['f_ref'])

if __name__ == '__main__':
    unittest.main()
