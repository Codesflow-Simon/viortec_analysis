import unittest
import numpy as np
import pytest
from sympy import symbols
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ligament_reconstructor.modelling.function import LigamentFunction, TrilinearFunction, BlankevoortFunction

class TestBlankevoortFunction(unittest.TestCase):
    """Test cases for the BlankevoortFunction class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Parameters: [alpha, k, l_0, f_ref]
        self.params = np.array([1.07, 60, 44, 0.05])
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
        expected_symbols = ['alpha', 'k', 'l_0', 'f_ref']
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
        
        # Check that all values are non-negative (force should be non-negative)
        self.assertTrue(np.all(result >= 0))
    
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
        alpha = self.params[0]  # alpha
        alpha_l_0 = alpha * l_0
        
        # Test region 1: x < l_0 (should be 0)
        x_below_l0 = np.array([l_0 - 1.0])
        result = self.function(x_below_l0)
        np.testing.assert_array_almost_equal(result, [0.0])
        
        # Test region 2: l_0 <= x <= alpha * l_0 (quadratic)
        x_region2 = np.array([l_0 + 0.5])
        result = self.function(x_region2)
        self.assertGreater(result[0], 0)
        
        # Test region 3: x > alpha * l_0 (linear)
        x_region3 = np.array([alpha_l_0 + 1.0])
        result = self.function(x_region3)
        self.assertGreater(result[0], 0)

    def test_hessian_symmetry(self):
        """Test that the Hessian is symmetric."""
        x_values = np.array([20, 40, 60, 80, 100])
        hessian = self.function.hess(x_values)

        tol = 1e-6
        for hessian_mat in hessian:
            # Test symmetry
            if not np.allclose(hessian_mat, hessian_mat.T, rtol=tol):
                self.fail(f"Hessian is not symmetric, got \n{hessian_mat}")

from ligament_reconstructor.modelling.loss import loss, loss_jac, loss_hess

class LossOptimisation(unittest.TestCase):
    """Test cases for the loss optimisation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = np.array([1.07, 60, 44, 0.05])
        self.function = BlankevoortFunction(self.params)
        self.x_data = np.linspace(20, 50, 5)
        self.y_data = self.function(self.x_data)
        self.loss_func = lambda params: loss(params, self.x_data, self.y_data, self.function)
        self.loss_jac_func = lambda params: loss_jac(params, self.x_data, self.y_data, self.function, self.function.jac)
        self.loss_hess_func = lambda params: loss_hess(params, self.x_data, self.y_data, self.function, self.function.jac, self.function.hess)

    def test_loss_function(self):
        """Test that the loss function is correct."""
        params = np.array([1.07, 60, 44, 0.05])
        loss = self.loss_func(params)
        self.assertEqual(loss, 0)

        params = np.array([1.06, 60, 44, 0.05])
        loss = self.loss_func(params)
        self.assertGreater(loss, 0)

    def test_loss_jac(self):
        """Test that the loss function is correct."""
        params = np.array([1.07, 60, 44, 0.05])
        loss_jac = self.loss_jac_func(params)
        zero_jac = np.zeros_like(loss_jac)
        for i in range(len(loss_jac)):
            self.assertEqual(loss_jac[i], zero_jac[i])

    def test_loss_hess_psd(self):
        """Test that the Hessian is positive semi-definite at the optimal point."""
        params = np.array([1.07, 60, 44, 0.05])
        hess = self.loss_hess_func(params)
        
        # Check eigenvalues are non-negative
        eigenvals = np.linalg.eigvals(hess)
        self.assertTrue(np.all(eigenvals >= -1e-10), 
                       f"Hessian is not PSD, eigenvalues: {eigenvals}")


class TestFunctionIntegration(unittest.TestCase):
    """Integration tests for function classes."""
    
    def test_parameter_consistency(self):
        """Test that parameter handling is consistent across methods."""
        # Test TrilinearFunction
        trilinear_params = np.array([100.0, 200.0, 300.0, 10.0, 1.2, 1.5])
        trilinear_func = TrilinearFunction(trilinear_params.copy())
        
        # Change parameters and verify all methods use updated parameters
        new_params = np.array([150.0, 250.0, 350.0, 12.0, 1.3, 1.6])
        trilinear_func.set_params(new_params)
        
        x_test = np.array([15.0])
        result1 = trilinear_func(x_test)
        result2 = trilinear_func.function(x_test, new_params)
        np.testing.assert_array_almost_equal(result1, result2)
        
        # Test BlankevoortFunction
        blankevoort_params = np.array([1.5, 100.0, 10.0, 0.1])
        blankevoort_func = BlankevoortFunction(blankevoort_params.copy())
        
        new_params = np.array([1.6, 120.0, 11.0, 0.15])
        blankevoort_func.set_params(new_params)
        
        result1 = blankevoort_func(x_test)
        result2 = blankevoort_func.function(x_test, new_params)
        np.testing.assert_array_almost_equal(result1, result2)
    
    def test_numerical_consistency(self):
        """Test numerical consistency between different evaluation methods."""
        # Test that function evaluation is consistent
        trilinear_params = np.array([100.0, 200.0, 300.0, 10.0, 1.2, 1.5])
        trilinear_func = TrilinearFunction(trilinear_params)
        
        x_test = np.array([12.0, 15.0, 18.0])
        
        # Test that __call__ and function give same results
        result_call = trilinear_func(x_test)
        result_function = trilinear_func.function(x_test, trilinear_params)
        np.testing.assert_array_almost_equal(result_call, result_function)
        
        # Test BlankevoortFunction
        blankevoort_params = np.array([1.5, 100.0, 10.0, 0.1])
        blankevoort_func = BlankevoortFunction(blankevoort_params)
        
        result_call = blankevoort_func(x_test)
        result_function = blankevoort_func.function(x_test, blankevoort_params)
        np.testing.assert_array_almost_equal(result_call, result_function)


if __name__ == '__main__':
    unittest.main() 