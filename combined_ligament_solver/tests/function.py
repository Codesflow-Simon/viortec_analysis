import unittest
import numpy as np
import pytest
from sympy import symbols
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ligament_models import LigamentFunction, BlankevoortFunction

class TestBlankevoortFunction(unittest.TestCase):
    """Test cases for the BlankevoortFunction class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Parameters: [k, alpha, l_0, f_ref]
        self.params = np.array([60, 1.07, 44, 0.05])
        self.function = BlankevoortFunction(self.params)
        self.x_test = np.array([20, 40, 60, 80, 100])
    
    def test_init_valid_params(self):
        """Test initialization with valid parameters."""
        self.assertTrue(np.array_equal(self.function.params, self.params))
    
    def test_init_invalid_params(self):
        """Test initialization with invalid number of parameters."""
        invalid_params = np.array([1.0, 2.0, 3.0])  # Only 3 params instead of 4
        with self.assertRaises(ValueError):
            BlankevoortFunction(invalid_params)
    
    def test_get_param_symbols(self):
        """Test get_param_symbols method."""
        symbols = self.function.get_param_symbols()
        expected_symbols = ['k', 'alpha', 'l_0', 'f_ref']
        for i, symbol in enumerate(symbols):
            self.assertEqual(str(symbol), expected_symbols[i])
    
    def test_sympy_implementation(self):
        """Test sympy_implementation method returns valid expression."""
        expr = self.function.sympy_implementation()
        self.assertIsNotNone(expr)
        # Check that it's a Piecewise expression
        self.assertTrue(hasattr(expr, 'args'))
    
    def test_function_evaluation(self):
        """Test function evaluation at various points."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function.function(x_values, self.params)
        
        # Check that result has correct shape
        self.assertEqual(result.shape, x_values.shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
        
        # For x < l_0, the function should return -f_ref (since force is 0 and we subtract f_ref)
        l_0 = self.params[2]
        f_ref = self.params[3]
        for i, x in enumerate(x_values):
            if x < l_0:
                self.assertAlmostEqual(result[i], -f_ref, places=10)
            else:
                # For x >= l_0, the function should be >= -f_ref
                self.assertGreaterEqual(result[i], -f_ref)
    
    def test_call_method(self):
        """Test the __call__ method."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function(x_values)
        expected = self.function.function(x_values, self.params)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_derivative_dx(self):
        """Test first derivative with respect to x."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function.dx(x_values)
        
        # Check that result has correct shape
        self.assertEqual(result.shape, x_values.shape)
        
        # Check that derivative is continuous (no infinite values)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_derivative_d2x2(self):
        """Test second derivative with respect to x."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function.d2x2(x_values)
        
        # Check that result has correct shape
        self.assertEqual(result.shape, x_values.shape)
        
        # Check that derivative is finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_jacobian(self):
        """Test Jacobian with respect to parameters."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function.jac(x_values)
        
        # Check that result has correct shape (n_params x n_points)
        expected_shape = (len(self.params), len(x_values))
        self.assertEqual(result.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_jacobian_scalar_input(self):
        """Test Jacobian with scalar input."""
        x_scalar = 20
        result = self.function.jac(x_scalar)
        
        # Check that result has correct shape (n_params,)
        expected_shape = (len(self.params),)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_hessian(self):
        """Test Hessian with respect to parameters."""
        x_values = np.array([20, 40, 60, 80, 100])
        result = self.function.hess(x_values)
        
        # Check that result has correct shape (n_points x n_params x n_params)
        expected_shape = (len(x_values), len(self.params), len(self.params))
        self.assertEqual(result.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_hessian_scalar_input(self):
        """Test Hessian with scalar input."""
        x_scalar = 20
        result = self.function.hess(x_scalar)
        
        # Check that result has correct shape (n_params x n_params)
        expected_shape = (len(self.params), len(self.params))
        self.assertEqual(result.shape, expected_shape)
        
        # Check that all values are finite
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_piecewise_regions(self):
        """Test function behavior in different piecewise regions."""
        l_0 = self.params[2]  # l_0
        alpha = self.params[1]  # alpha (corrected index)
        f_ref = self.params[3]  # f_ref
        alpha_l_0 = alpha * l_0
        
        # Test region 1: x < l_0 (should be -f_ref since force is 0 and we subtract f_ref)
        x_below_l0 = np.array([l_0 - 1.0])
        result = self.function(x_below_l0)
        np.testing.assert_array_almost_equal(result, [-f_ref])
        
        # Test region 2: l_0 <= x <= alpha * l_0 (quadratic)
        x_region2 = np.array([l_0 + 0.5])
        result = self.function(x_region2)
        self.assertGreater(result[0], -f_ref)  # Should be greater than -f_ref
        
        # Test region 3: x > alpha * l_0 (linear)
        x_region3 = np.array([alpha_l_0 + 1.0])
        result = self.function(x_region3)
        self.assertGreater(result[0], -f_ref)  # Should be greater than -f_ref

    def test_hessian_symmetry(self):
        """Test that the Hessian is symmetric."""
        x_values = np.array([20, 40, 60, 80, 100])
        hessian = self.function.hess(x_values)

        tol = 1e-6
        for hessian_mat in hessian:
            # Test symmetry
            if not np.allclose(hessian_mat, hessian_mat.T, rtol=tol):
                self.fail(f"Hessian is not symmetric, got \n{hessian_mat}")

    def test_vectorized_function(self):
        """Test vectorized function evaluation."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Original parameters
            [70, 1.07, 44, 0.05],  # Different k
            [60, 1.17, 44, 0.05],  # Different alpha
            [60, 1.07, 54, 0.05]   # Different l_0
        ])
        
        result = self.function.vectorized_function(self.x_test, params_array)
        
        # Check shape
        expected_shape = (4, len(self.x_test))
        self.assertEqual(result.shape, expected_shape)
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(result)))
        
        # Check consistency with single evaluation
        single_result = self.function.vectorized_function(self.x_test, self.params.reshape(1, -1))
        expected_single = self.function(self.x_test)
        np.testing.assert_array_almost_equal(single_result[0], expected_single)

    def test_vectorized_jacobian(self):
        """Test vectorized Jacobian computation."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Original parameters
            [70, 1.07, 44, 0.05],  # Different k
            [60, 1.17, 44, 0.05]   # Different alpha
        ])
        
        result = self.function.vectorized_jacobian(self.x_test, params_array)
        
        # Check shape
        expected_shape = (3, 4, len(self.x_test))  # (n_param_sets, n_params, n_points)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(result)))
        
        # Check consistency with single evaluation
        single_result = self.function.vectorized_jacobian(self.x_test, self.params.reshape(1, -1))
        expected_single = self.function.jac(self.x_test)
        np.testing.assert_array_almost_equal(single_result[0], expected_single)

    def test_vectorized_hessian(self):
        """Test vectorized Hessian computation."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Original parameters
            [70, 1.07, 44, 0.05],  # Different k
            [60, 1.17, 44, 0.05]   # Different alpha
        ])
        
        result = self.function.vectorized_hessian(self.x_test, params_array)
        
        # Check shape
        expected_shape = (3, len(self.x_test), 4, 4)  # (n_param_sets, n_points, n_params, n_params)
        self.assertEqual(result.shape, expected_shape)
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(result)))
        
        # Check consistency with single evaluation
        single_result = self.function.vectorized_hessian(self.x_test, self.params.reshape(1, -1))
        expected_single = self.function.hess(self.x_test)
        np.testing.assert_array_almost_equal(single_result[0], expected_single)
        
        # Check symmetry for all Hessians
        for param_set_idx in range(3):
            for point_idx in range(len(self.x_test)):
                hess = result[param_set_idx, point_idx]
                np.testing.assert_array_almost_equal(hess, hess.T, decimal=10)

# Import loss functions from the correct location
from ligament_reconstructor.ligament_optimiser import loss, loss_jac, loss_hess

class LossOptimisation(unittest.TestCase):
    """Test cases for the loss optimisation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = np.array([60, 1.07, 44, 0.05])  # Corrected parameter order
        self.function = BlankevoortFunction(self.params)
        self.x_data = np.linspace(20, 50, 5)
        self.y_data = self.function(self.x_data)
        self.loss_func = lambda params: loss(params, self.x_data, self.y_data, self.function)
        self.loss_jac_func = lambda params: loss_jac(params, self.x_data, self.y_data, self.function)
        self.loss_hess_func = lambda params: loss_hess(params, self.x_data, self.y_data, self.function)

    def test_loss_function(self):
        """Test that the loss function is correct."""
        params = np.array([60, 1.07, 44, 0.05])  # Corrected parameter order
        loss_val = self.loss_func(params)
        self.assertEqual(loss_val, 0)

        params = np.array([60, 1.06, 44, 0.05])  # Slightly different alpha
        loss_val = self.loss_func(params)
        self.assertGreater(loss_val, 0)

    def test_loss_jac(self):
        """Test that the loss Jacobian is correct."""
        params = np.array([60, 1.07, 44, 0.05])  # Corrected parameter order
        loss_jac_val = self.loss_jac_func(params)
        zero_jac = np.zeros_like(loss_jac_val)
        np.testing.assert_array_almost_equal(loss_jac_val, zero_jac, decimal=10)

    def test_loss_hess_psd(self):
        """Test that the Hessian is positive semi-definite at the optimal point."""
        params = np.array([60, 1.07, 44, 0.05])  # Corrected parameter order
        hess = self.loss_hess_func(params)
        
        # Check eigenvalues are non-negative
        eigenvals = np.linalg.eigvals(hess)
        self.assertTrue(np.all(eigenvals >= -1e-10), 
                       f"Hessian is not PSD, eigenvalues: {eigenvals}")

    def test_vectorized_loss_function(self):
        """Test that the vectorized loss function works correctly."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Optimal parameters
            [60, 1.06, 44, 0.05],  # Slightly different alpha
            [65, 1.07, 44, 0.05],  # Different k
            [60, 1.07, 45, 0.05]   # Different l_0
        ])
        
        loss_vals = self.loss_func(params_array)
        
        # Check shape
        self.assertEqual(loss_vals.shape, (4,))
        
        # First set should have zero loss (optimal parameters)
        self.assertAlmostEqual(loss_vals[0], 0, places=10)
        
        # Other sets should have positive loss
        self.assertTrue(np.all(loss_vals[1:] > 0))
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(loss_vals)))

    def test_vectorized_loss_jac(self):
        """Test that the vectorized loss Jacobian works correctly."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Optimal parameters
            [60, 1.06, 44, 0.05],  # Slightly different alpha
            [65, 1.07, 44, 0.05]   # Different k
        ])
        
        loss_jac_vals = self.loss_jac_func(params_array)
        
        # Check shape
        expected_shape = (3, 4)  # (n_param_sets, n_params)
        self.assertEqual(loss_jac_vals.shape, expected_shape)
        
        # First set should have zero gradient (optimal parameters)
        np.testing.assert_array_almost_equal(loss_jac_vals[0], np.zeros(4), decimal=10)
        
        # Other sets should have non-zero gradients
        self.assertTrue(np.any(loss_jac_vals[1:] != 0))
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(loss_jac_vals)))

    def test_vectorized_loss_hess(self):
        """Test that the vectorized loss Hessian works correctly."""
        # Test with multiple parameter sets
        params_array = np.array([
            [60, 1.07, 44, 0.05],  # Optimal parameters
            [60, 1.06, 44, 0.05],  # Slightly different alpha
            [65, 1.07, 44, 0.05]   # Different k
        ])
        
        loss_hess_vals = self.loss_hess_func(params_array)
        
        # Check shape
        expected_shape = (3, 4, 4)  # (n_param_sets, n_params, n_params)
        self.assertEqual(loss_hess_vals.shape, expected_shape)
        
        # Check that all Hessians are symmetric
        for i in range(3):
            hess = loss_hess_vals[i]
            np.testing.assert_array_almost_equal(hess, hess.T, decimal=10)
        
        # Check that the optimal Hessian (first one) is positive semi-definite
        optimal_hess = loss_hess_vals[0]
        eigenvals = np.linalg.eigvals(optimal_hess)
        self.assertTrue(np.all(eigenvals >= -1e-10), 
                       f"Optimal Hessian is not PSD, eigenvalues: {eigenvals}")
        
        # Check that all Hessians are symmetric (this should always be true)
        for i in range(3):
            hess = loss_hess_vals[i]
            eigenvals = np.linalg.eigvals(hess)
            # Non-optimal Hessians can have negative eigenvalues, but should still be symmetric
            self.assertTrue(np.allclose(hess, hess.T, atol=1e-10), 
                           f"Hessian {i} is not symmetric")
        
        # Check that results are finite
        self.assertTrue(np.all(np.isfinite(loss_hess_vals)))

    def test_consistency_between_single_and_vectorized(self):
        """Test consistency between single parameter and vectorized versions."""
        # Single parameter set
        params_single = np.array([60, 1.07, 44, 0.05])
        
        # Same parameters in vectorized format
        params_vectorized = params_single.reshape(1, -1)
        
        # Test loss function
        loss_single = self.loss_func(params_single)
        loss_vectorized = self.loss_func(params_vectorized)
        self.assertAlmostEqual(loss_single, loss_vectorized[0], places=10)
        
        # Test loss Jacobian
        jac_single = self.loss_jac_func(params_single)
        jac_vectorized = self.loss_jac_func(params_vectorized)
        np.testing.assert_array_almost_equal(jac_single, jac_vectorized[0], decimal=10)
        
        # Test loss Hessian
        hess_single = self.loss_hess_func(params_single)
        hess_vectorized = self.loss_hess_func(params_vectorized)
        np.testing.assert_array_almost_equal(hess_single, hess_vectorized[0], decimal=10)


class TestFunctionIntegration(unittest.TestCase):
    """Integration tests for function classes."""
    
    def test_parameter_consistency(self):
        """Test that parameter handling is consistent across methods."""
        # Test BlankevoortFunction
        blankevoort_params = np.array([60.0, 1.5, 10.0, 0.1])  # Corrected parameter order
        blankevoort_func = BlankevoortFunction(blankevoort_params.copy())
        
        new_params = np.array([70.0, 1.6, 11.0, 0.15])  # Corrected parameter order
        blankevoort_func.set_params(new_params)
        
        x_test = np.array([15.0])
        result1 = blankevoort_func(x_test)
        result2 = blankevoort_func.function(x_test, new_params)
        np.testing.assert_array_almost_equal(result1, result2)
    
    def test_numerical_consistency(self):
        """Test numerical consistency between different evaluation methods."""
        # Test BlankevoortFunction
        blankevoort_params = np.array([60.0, 1.5, 10.0, 0.1])  # Corrected parameter order
        blankevoort_func = BlankevoortFunction(blankevoort_params)
        
        x_test = np.array([12.0, 15.0, 18.0])
        
        result_call = blankevoort_func(x_test)
        result_function = blankevoort_func.function(x_test, blankevoort_params)
        np.testing.assert_array_almost_equal(result_call, result_function)


class TestBroadcastingBehavior(unittest.TestCase):
    """Test broadcasting behavior with different shapes of x and params."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.base_params = np.array([60, 1.07, 44, 0.05])
        self.function = BlankevoortFunction(self.base_params)
    
    def test_scalar_x_vectorized_params(self):
        """Test broadcasting with scalar x and multiple parameter sets."""
        x_scalar = np.array([30.0])  # Convert to array for vectorized methods
        params_array = np.array([
            [60, 1.07, 44, 0.05],
            [70, 1.07, 44, 0.05],
            [60, 1.17, 44, 0.05]
        ])
        
        # Test vectorized function
        result = self.function.vectorized_function(x_scalar, params_array)
        expected_shape = (3, 1)
        self.assertEqual(result.shape, expected_shape)
        
        # Test vectorized Jacobian
        jac_result = self.function.vectorized_jacobian(x_scalar, params_array)
        expected_jac_shape = (3, 4, 1)
        self.assertEqual(jac_result.shape, expected_jac_shape)
        
        # Test vectorized Hessian
        hess_result = self.function.vectorized_hessian(x_scalar, params_array)
        expected_hess_shape = (3, 1, 4, 4)
        self.assertEqual(hess_result.shape, expected_hess_shape)
        
        # Check consistency with single evaluations
        for i in range(3):
            single_result = self.function.function(x_scalar[0], params_array[i])
            self.assertAlmostEqual(result[i, 0], single_result, places=10)
    
    def test_vector_x_scalar_params(self):
        """Test broadcasting with vector x and single parameter set."""
        x_vector = np.array([20, 30, 40, 50])
        params_scalar = np.array([60, 1.07, 44, 0.05])
        
        # Test vectorized function
        result = self.function.vectorized_function(x_vector, params_scalar.reshape(1, -1))
        expected_shape = (1, 4)
        self.assertEqual(result.shape, expected_shape)
        
        # Test vectorized Jacobian
        jac_result = self.function.vectorized_jacobian(x_vector, params_scalar.reshape(1, -1))
        expected_jac_shape = (1, 4, 4)
        self.assertEqual(jac_result.shape, expected_jac_shape)
        
        # Test vectorized Hessian
        hess_result = self.function.vectorized_hessian(x_vector, params_scalar.reshape(1, -1))
        expected_hess_shape = (1, 4, 4, 4)
        self.assertEqual(hess_result.shape, expected_hess_shape)
        
        # Check consistency with single evaluation
        single_result = self.function.function(x_vector, params_scalar)
        np.testing.assert_array_almost_equal(result[0], single_result, decimal=10)
    
    def test_vector_x_vectorized_params(self):
        """Test broadcasting with vector x and multiple parameter sets."""
        x_vector = np.array([20, 30, 40, 50])
        params_array = np.array([
            [60, 1.07, 44, 0.05],
            [70, 1.07, 44, 0.05],
            [60, 1.17, 44, 0.05],
            [60, 1.07, 54, 0.05]
        ])
        
        # Test vectorized function
        result = self.function.vectorized_function(x_vector, params_array)
        expected_shape = (4, 4)
        self.assertEqual(result.shape, expected_shape)
        
        # Test vectorized Jacobian
        jac_result = self.function.vectorized_jacobian(x_vector, params_array)
        expected_jac_shape = (4, 4, 4)
        self.assertEqual(jac_result.shape, expected_jac_shape)
        
        # Test vectorized Hessian
        hess_result = self.function.vectorized_hessian(x_vector, params_array)
        expected_hess_shape = (4, 4, 4, 4)
        self.assertEqual(hess_result.shape, expected_hess_shape)
        
        # Check consistency with single evaluations
        for i in range(4):
            single_result = self.function.function(x_vector, params_array[i])
            np.testing.assert_array_almost_equal(result[i], single_result, decimal=10)
    
    def test_single_point_multiple_params(self):
        """Test broadcasting with single x point and many parameter sets."""
        x_single = np.array([35.0])
        params_array = np.array([
            [60, 1.07, 44, 0.05],
            [70, 1.07, 44, 0.05],
            [60, 1.17, 44, 0.05],
            [60, 1.07, 54, 0.05],
            [65, 1.12, 49, 0.075]
        ])
        
        # Test vectorized function
        result = self.function.vectorized_function(x_single, params_array)
        expected_shape = (5, 1)
        self.assertEqual(result.shape, expected_shape)
        
        # Test vectorized Jacobian
        jac_result = self.function.vectorized_jacobian(x_single, params_array)
        expected_jac_shape = (5, 4, 1)
        self.assertEqual(jac_result.shape, expected_jac_shape)
        
        # Test vectorized Hessian
        hess_result = self.function.vectorized_hessian(x_single, params_array)
        expected_hess_shape = (5, 1, 4, 4)
        self.assertEqual(hess_result.shape, expected_hess_shape)
        
        # Check consistency with single evaluations
        for i in range(5):
            single_result = self.function.function(x_single, params_array[i])
            np.testing.assert_array_almost_equal(result[i], single_result, decimal=10)
    
    def test_edge_cases_broadcasting(self):
        """Test edge cases in broadcasting behavior."""
        # Test with empty x array - this should work but return empty results
        x_empty = np.array([])
        params_array = np.array([[60, 1.07, 44, 0.05]])
        
        # Empty x should return empty results
        result = self.function.vectorized_function(x_empty, params_array)
        expected_shape = (1, 0)
        self.assertEqual(result.shape, expected_shape)
        
        # Test with single parameter set as 1D array
        x_vector = np.array([20, 30, 40])
        params_1d = np.array([60, 1.07, 44, 0.05])
        
        # This should work by automatically reshaping
        result = self.function.vectorized_function(x_vector, params_1d)
        expected_shape = (1, 3)
        self.assertEqual(result.shape, expected_shape)
        
        # Test with very large parameter arrays
        x_small = np.array([30.0])
        params_large = np.random.rand(100, 4) * 100  # 100 random parameter sets
        
        result = self.function.vectorized_function(x_small, params_large)
        expected_shape = (100, 1)
        self.assertEqual(result.shape, expected_shape)
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_broadcasting_consistency_loss_functions(self):
        """Test that loss functions handle broadcasting correctly."""
        x_data = np.array([20, 30, 40, 50])
        y_data = self.function(x_data)
        
        # Test with single parameter set
        params_single = np.array([60, 1.07, 44, 0.05])
        loss_single = loss(params_single, x_data, y_data, self.function)
        
        # Test with same parameters in vectorized format
        params_vectorized = params_single.reshape(1, -1)
        loss_vectorized = loss(params_vectorized, x_data, y_data, self.function)
        
        self.assertAlmostEqual(loss_single, loss_vectorized[0], places=10)
        
        # Test with multiple parameter sets
        params_multiple = np.array([
            [60, 1.07, 44, 0.05],
            [70, 1.07, 44, 0.05],
            [60, 1.17, 44, 0.05]
        ])
        loss_multiple = loss(params_multiple, x_data, y_data, self.function)
        
        expected_shape = (3,)
        self.assertEqual(loss_multiple.shape, expected_shape)
        
        # First set should have zero loss (optimal parameters)
        self.assertAlmostEqual(loss_multiple[0], 0, places=10)
        
        # Other sets should have positive loss
        self.assertTrue(np.all(loss_multiple[1:] > 0))
    
    def test_broadcasting_with_different_x_lengths(self):
        """Test broadcasting with different lengths of x arrays."""
        params_array = np.array([
            [60, 1.07, 44, 0.05],
            [70, 1.07, 44, 0.05]
        ])
        
        # Test with different x lengths
        x_lengths = [1, 3, 5, 10]
        
        for length in x_lengths:
            x_test = np.linspace(20, 50, length)
            
            # Test vectorized function
            result = self.function.vectorized_function(x_test, params_array)
            expected_shape = (2, length)
            self.assertEqual(result.shape, expected_shape)
            
            # Test vectorized Jacobian
            jac_result = self.function.vectorized_jacobian(x_test, params_array)
            expected_jac_shape = (2, 4, length)
            self.assertEqual(jac_result.shape, expected_jac_shape)
            
            # Test vectorized Hessian
            hess_result = self.function.vectorized_hessian(x_test, params_array)
            expected_hess_shape = (2, length, 4, 4)
            self.assertEqual(hess_result.shape, expected_hess_shape)
            
            # Check that all results are finite
            self.assertTrue(np.all(np.isfinite(result)))
            self.assertTrue(np.all(np.isfinite(jac_result)))
            self.assertTrue(np.all(np.isfinite(hess_result)))
    
    def test_scalar_vs_array_x_behavior(self):
        """Test the difference between scalar and array x inputs."""
        x_scalar = 30.0
        x_array = np.array([30.0])
        params = np.array([60, 1.07, 44, 0.05])
        
        # Test regular function methods with scalar x
        func_scalar = self.function.function(x_scalar, params)
        func_array = self.function.function(x_array, params)
        
        # Both should work and give the same result
        self.assertAlmostEqual(func_scalar, func_array[0], places=10)
        
        # Test Jacobian with scalar x
        jac_scalar = self.function.jac(x_scalar)
        jac_array = self.function.jac(x_array)
        
        # Both should work and give the same result (but different shapes)
        # jac_scalar has shape (4,) and jac_array has shape (4, 1)
        np.testing.assert_array_almost_equal(jac_scalar, jac_array.flatten(), decimal=10)
        
        # Test Hessian with scalar x
        hess_scalar = self.function.hess(x_scalar)
        hess_array = self.function.hess(x_array)
        
        # Both should work and give the same result (but different shapes)
        # hess_scalar has shape (4, 4) and hess_array has shape (1, 4, 4)
        np.testing.assert_array_almost_equal(hess_scalar, hess_array[0], decimal=10)
        
        # Test vectorized methods - these handle both scalar and array x
        params_array = params.reshape(1, -1)
        
        # This should work with array x
        vec_func = self.function.vectorized_function(x_array, params_array)
        self.assertEqual(vec_func.shape, (1, 1))
        
        # This should also work with scalar x (automatic conversion)
        vec_func_auto = self.function.vectorized_function(x_scalar, params_array)
        self.assertEqual(vec_func_auto.shape, (1,))
        np.testing.assert_array_almost_equal(vec_func[0], vec_func_auto, decimal=10)


if __name__ == '__main__':
    unittest.main() 