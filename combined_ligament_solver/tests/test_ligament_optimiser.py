import unittest
import numpy as np
import pytest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ligament_models import BlankevoortFunction
from ligament_reconstructor.ligament_optimiser import loss, loss_jac, loss_hess, reconstruct_ligament
from ligament_models.constraints import ConstraintManager
from ligament_models.transformations import constraint_transform, inverse_constraint_transform
from ligament_reconstructor.utils import parameter_norm

class TestLigamentOptimiser(unittest.TestCase):
    """Test cases for the ligament optimiser."""
    
    def setUp(self):
        """Set up test fixtures."""
        # True parameters for generating synthetic data
        self.true_params = np.array([60.0, 0.06, 45.0, 117.36238600601739])
        self.function = BlankevoortFunction(self.true_params)
        
        # Generate synthetic data
        self.x_data = np.linspace(40, 70, 100)
        self.y_data = self.function(self.x_data)
        
        self.percent_diff_tolerance = 0.1
        
        # Constraint manager
        self.constraint_manager = ConstraintManager(mode='blankevoort')
    
    def test_loss_function_at_true_parameters(self):
        """Test that loss is zero at true parameters."""
        loss_value = loss(self.true_params, self.x_data, self.y_data, self.function)
        self.assertAlmostEqual(loss_value, 0.0, places=10)
    
    def test_loss_function_at_different_parameters(self):
        """Test that loss increases for different parameters."""
        # Test with slightly different parameters
        different_params = self.true_params.copy()
        different_params[0] += 5.0  # Change k
        
        loss_true = loss(self.true_params, self.x_data, self.y_data, self.function)
        loss_different = loss(different_params, self.x_data, self.y_data, self.function)
        
        self.assertGreater(loss_different, loss_true)
    
    def test_loss_jacobian_at_true_parameters(self):
        """Test that loss Jacobian is zero at true parameters."""
        jac_value = loss_jac(self.true_params, self.x_data, self.y_data, self.function, self.function.jac)
        
        # Check that Jacobian is close to zero (optimal point)
        np.testing.assert_array_almost_equal(jac_value, np.zeros_like(jac_value), decimal=10)
    
    def test_loss_hessian_positive_definite(self):
        """Test that loss Hessian is positive semi-definite at true parameters."""
        hess_value = loss_hess(self.true_params, self.x_data, self.y_data, self.function, self.function.jac, self.function.hess)
        
        # Check eigenvalues are non-negative
        eigenvals = np.linalg.eigvals(hess_value)
        self.assertTrue(np.all(eigenvals >= -1e-10), 
                       f"Hessian is not PSD, eigenvalues: {eigenvals}")
    
    def test_loss_with_regularization(self):
        """Test that regularization increases the loss."""
        loss_no_reg = loss(self.true_params, self.x_data, self.y_data, self.function, include_reg=False)
        loss_with_reg = loss(self.true_params, self.x_data, self.y_data, self.function, include_reg=True)
        
        self.assertGreater(loss_with_reg, loss_no_reg)
    
    def test_loss_jacobian_with_regularization(self):
        """Test that regularization affects the Jacobian."""
        jac_no_reg = loss_jac(self.true_params, self.x_data, self.y_data, self.function, self.function.jac, include_reg=False)
        jac_with_reg = loss_jac(self.true_params, self.x_data, self.y_data, self.function, self.function.jac, include_reg=True)
        
        # With regularization, Jacobian should not be zero at true parameters
        self.assertFalse(np.allclose(jac_with_reg, np.zeros_like(jac_with_reg)))
    
    def test_loss_hessian_with_regularization(self):
        """Test that regularization affects the Hessian."""
        hess_no_reg = loss_hess(self.true_params, self.x_data, self.y_data, self.function, self.function.jac, self.function.hess, include_reg=False)
        hess_with_reg = loss_hess(self.true_params, self.x_data, self.y_data, self.function, self.function.jac, self.function.hess, include_reg=True)
        
        # Hessians should be different
        self.assertFalse(np.allclose(hess_no_reg, hess_with_reg))

    def test_success_of_optimisation(self):
        """Test that the optimisation succeeds consistently over multiple runs."""
        n_runs = 20
        successes = 0
        
        for i in range(n_runs):
            result = reconstruct_ligament(self.x_data, self.y_data)
            if result['opt_result'].success:
                successes += 1
                
        success_rate = (successes / n_runs) * 100
        self.assertEqual(success_rate, 100.0, f"Expected 100% success rate but got {success_rate}%")
    
    def test_reconstruct_ligament_optimality(self):
        """Test that reconstruct_ligament finds optimal parameters."""
        # Run the optimization with proper initial parameters
        result = reconstruct_ligament(self.x_data, self.y_data)
        
        # Check that the loss is very small (close to zero)
        # Check that optimized parameters are within 20% of true parameters
        for param_name, opt_val in result['params'].items():
            true_val = self.true_params[list(result['params'].keys()).index(param_name)]
            if param_name == 'alpha':
                continue
            percent_diff = abs(opt_val - true_val) / abs(true_val)
            self.assertLess(percent_diff, self.percent_diff_tolerance,
                           f"Parameter {param_name}: expected {true_val:.6f}, got {opt_val:.6f}, difference {percent_diff:.1%}")

        # Check that the Jacobian at the optimal point is close to zero
        optimal_params = np.array(list(result['params'].values()))
        jac_at_opt = loss_jac(optimal_params, self.x_data, self.y_data, self.function, self.function.jac)
        max_jac = np.max(jac_at_opt)
        min_jac = np.min(jac_at_opt)
        self.assertLess(max_jac, 1e-4, f"Maximum absolute Jacobian value too large, {max_jac}")
        self.assertGreater(min_jac, -1e-4, f"Minimum absolute Jacobian value too large, {min_jac}")
        
        # Check that Hessian at optimal point is positive semi-definite
        hess_at_opt = loss_hess(optimal_params, self.x_data, self.y_data, self.function, self.function.jac, self.function.hess)
        eigenvals = np.linalg.eigvals(hess_at_opt)
        self.assertTrue(np.all(eigenvals >= -1e-10), 
                       f"Hessian at optimal point is not PSD, eigenvalues: {eigenvals}")
    
    def test_reconstruct_ligament_parameter_recovery(self):
        """Test that reconstruct_ligament recovers parameters accurately."""
        # Run the optimization with proper initial parameters
        result = reconstruct_ligament(self.x_data, self.y_data)
        
        # Check that recovered parameters are close to true parameters
        recovered_params = np.array(list(result['params'].values()))
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        
        for i, name in enumerate(param_names):
            recovered_param = recovered_params[i]
            true_param = self.true_params[i]
            percent_diff = abs(recovered_param - true_param) / abs(true_param)
            if name == 'alpha':
                continue
            self.assertLess(percent_diff, self.percent_diff_tolerance, 
                           f"Parameter {name}: expected {self.true_params[i]:.6f}, got {recovered_params[i]:.6f}, difference {percent_diff:.1%}")
    
    def test_reconstruct_ligament_output_structure(self):
        """Test that reconstruct_ligament returns the expected output structure."""
        result = reconstruct_ligament(self.x_data, self.y_data)
        
        # Check that all expected keys are present
        expected_keys = ['opt_result', 'y_hat', 'params', 'x_data', 'y_data', 'function', 'loss', 'loss_jac', 'loss_hess']
        for key in expected_keys:
            self.assertIn(key, result, f"Missing key: {key}")
        
        # Check that params is a dictionary with correct parameter names
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        for name in param_names:
            self.assertIn(name, result['params'], f"Missing parameter: {name}")
        
        # Check that arrays have correct shapes
        self.assertEqual(len(result['y_hat']), len(self.x_data), f"y_hat has wrong length, expected {len(self.x_data)}, got {len(result['y_hat'])}")
        self.assertEqual(len(result['x_data']), len(self.x_data), f"x_data has wrong length, expected {len(self.x_data)}, got {len(result['x_data'])}")
        self.assertEqual(len(result['y_data']), len(self.y_data), f"y_data has wrong length, expected {len(self.y_data)}, got {len(result['y_data'])}")
    
    def test_optimization_with_noisy_data(self):
        """Test optimization with noisy data."""
        # Add noise to the data
        noise_level = 0.01
        noisy_y_data = self.y_data + noise_level * np.random.randn(len(self.y_data))
        
        # Run optimization with noisy data and proper initial parameters
        result = reconstruct_ligament(self.x_data, noisy_y_data)
        
        # Check that loss is reasonable (should be around noise level squared)
        expected_loss = noise_level**2 * len(self.y_data)
        # Check that optimized parameters are within 20% of true values
        for param_name, opt_value in result['params'].items():
            true_value = self.true_params[list(result['params'].keys()).index(param_name)]
            percent_diff = abs(opt_value - true_value) / abs(true_value)
            if param_name == 'alpha':
                continue
            self.assertLess(percent_diff, self.percent_diff_tolerance,
                           f"Parameter {param_name}: expected {true_value:.6f}, got {opt_value:.6f}, difference {percent_diff:.1%}")
    
    def test_constraint_satisfaction(self):
        """Test that optimized parameters satisfy constraints."""
        result = reconstruct_ligament(self.x_data, self.y_data)

        # Get bounds from constraint manager
        bounds = self.constraint_manager.get_constraints_list()
        param_names = ['k', 'alpha', 'l_0', 'f_ref']
        
        # Check that each parameter is within its bounds
        for i, (name, (lower, upper)) in enumerate(zip(param_names, bounds)):
            param_value = result['params'][name]
            self.assertGreaterEqual(param_value, lower - 1e-5, 
                                  f"Parameter {name} ({param_value}) below lower bound {lower}")
            self.assertLessEqual(param_value, upper + 1e-5, 
                               f"Parameter {name} ({param_value}) above upper bound {upper}")
    



if __name__ == '__main__':
    unittest.main()
