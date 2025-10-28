import unittest
import numpy as np
import pytest
import sys
import os
import yaml
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from ligament_models.blankevoort import BlankevoortFunction
from statics_solver.models.statics_model import KneeModel
from plotting_tools import visualize_ligament_curves, visualize_theta_force_curves

class TestPlottingTools(unittest.TestCase):
    """Test cases for the plotting tools."""
    
    def setUp(self):
        # Load config files
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
        constraints_path = os.path.join(os.path.dirname(__file__), '..', 'constraints.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        with open(constraints_path, 'r') as f:
            self.constraints_config = yaml.safe_load(f)
        
        # Create mock data for testing
        self.mock_data = {
            'thetas': np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            'applied_force': np.array([10, 20, 30, 40, 50]),
            'length_known_a': np.array([25, 26, 27, 28, 29]),  # LCL lengths
            'force_known_a': np.array([5, 10, 15, 20, 25]),   # LCL forces
            'length_known_b': np.array([30, 31, 32, 33, 34]), # MCL lengths
            'force_known_b': np.array([8, 12, 16, 20, 24])    # MCL forces
        }
        
        # Create mock MCMC samples (8 parameters: 4 for MCL, 4 for LCL)
        np.random.seed(42)  # For reproducible tests
        self.mock_samples = np.random.normal(0, 1, (100, 8))
    
    def test_visualize_ligament_curves_basic(self):
        """Test that visualize_ligament_curves runs without errors."""
        try:
            # Change to a temporary directory to avoid cluttering the workspace
            with tempfile.TemporaryDirectory() as temp_dir:
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                
                # Call the function
                visualize_ligament_curves(self.config, self.mock_samples, self.mock_data)
                
                # Check that the plot file was created
                self.assertTrue(os.path.exists('ligament_curves.png'))
                
                # Restore original directory
                os.chdir(original_cwd)
                
        except Exception as e:
            self.fail(f"visualize_ligament_curves raised an exception: {e}")
    
    def test_plotting_functions_import(self):
        """Test that plotting functions can be imported and have the expected signatures."""
        # Test that the functions exist and are callable
        self.assertTrue(callable(visualize_ligament_curves))
        self.assertTrue(callable(visualize_theta_force_curves))
        
        # Test that BlankevoortFunction can be instantiated
        test_params = np.array([1.0, 2.0, 3.0, 4.0])
        try:
            func = BlankevoortFunction(test_params)
            self.assertIsNotNone(func)
        except Exception as e:
            self.fail(f"BlankevoortFunction instantiation failed: {e}")

if __name__ == '__main__':
    unittest.main()
