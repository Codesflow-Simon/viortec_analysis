import unittest
import numpy as np
import pytest
import sys
import os
import sympy
import warnings
from sympy import symbols, Matrix

# Add the src directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.ligament_models.blankevoort import BlankevoortFunction
from src.statics_solver.src.springs import BlankevoortSpring
from src.statics_solver.src.reference_frame import Point, ReferenceFrame
from src.statics_solver.src.rigid_body import Force

class TestBlankevoortSpring(unittest.TestCase):
    """Test cases for the BlankevoortSpring class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create reference frame
        self.frame = ReferenceFrame("test_frame")
        self.frame.set_as_ground_frame()
        
        # Create test points
        self.point1 = Point([0, 0, 0], self.frame)
        self.point2 = Point([1, 0, 0], self.frame)
        
        # Standard parameters for Blankevoort function
        self.k = 60.0
        self.alpha = 0.06
        self.l_0 = 50.0
        self.f_ref = 0.0
    
    def test_init_with_numeric_params(self):
        """Test initialization with numeric parameters."""
        spring = BlankevoortSpring(self.point1, self.point2, "test_spring", 
                                 self.k, self.alpha, self.l_0)
        
        self.assertEqual(spring.name, "test_spring")
        self.assertEqual(spring.k, self.k)
        self.assertEqual(spring.alpha, self.alpha)
        self.assertEqual(spring.l_0, self.l_0)
        self.assertEqual(spring.function.__class__.__name__, 'BlankevoortFunction')
        
        # Check that points are stored correctly
        self.assertEqual(spring.point_1.coordinates, self.point1.coordinates)
        self.assertEqual(spring.point_2.coordinates, self.point2.coordinates)
    
    def test_init_with_symbolic_params(self):
        """Test initialization with symbolic parameters."""
        k, alpha, l_0 = symbols('k alpha l_0')
        spring = BlankevoortSpring(self.point1, self.point2, "symbolic_spring", 
                                 k, alpha, l_0)
        
        self.assertEqual(spring.k, k)
        self.assertEqual(spring.alpha, alpha)
        self.assertEqual(spring.l_0, l_0)
        self.assertEqual(spring.function.__class__.__name__, 'BlankevoortFunction')
    
    def test_from_ligament_function(self):
        """Test creation from ligament function."""
        params = np.array([self.k, self.alpha, self.l_0, self.f_ref])
        ligament_func = BlankevoortFunction(params)
        
        spring = BlankevoortSpring.from_ligament_function(
            self.point1, self.point2, "from_func", ligament_func)
        
        self.assertEqual(spring.name, "from_func")
        self.assertEqual(spring.k, self.k)
        self.assertEqual(spring.alpha, self.alpha)
        self.assertEqual(spring.l_0, self.l_0)
        self.assertEqual(spring.function.__class__.__name__, 'BlankevoortFunction')
    
    def test_get_spring_length_normal_strain(self):
        """Test spring length calculation for normal strain."""
        spring = BlankevoortSpring(self.point1, self.point2, "normal_spring", 
                                 self.k, self.alpha, self.l_0)
        
        length = spring.get_spring_length()
        expected_length = (self.point2 - self.point1).norm()
        
        self.assertEqual(length, expected_length)
    
    def test_get_spring_length_no_warning_normal_strain(self):
        """Test that no warning is issued for normal strain."""
        # Create points with normal strain
        normal_strain_length = self.l_0 * 1.1  # 10% strain
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([normal_strain_length, 0, 0], self.frame)
        
        spring = BlankevoortSpring(point1, point2, "normal_strain_spring", 
                                 self.k, self.alpha, self.l_0)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            length = spring.get_spring_length()
            
            # Check that no warning was issued
            self.assertEqual(len(w), 0)
    
    def test_get_force_magnitude_compression(self):
        """Test force magnitude calculation in compression."""
        # Create points closer together than rest length
        compression_length = self.l_0 * 0.9  # 10% compression
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([compression_length, 0, 0], self.frame)
        
        spring = BlankevoortSpring(point1, point2, "compression_spring", 
                                 self.k, self.alpha, self.l_0)
        
        force_mag = spring.get_force_magnitude()
        
        # Should return 0 for compression (x < 0 in Blankevoort function)
        self.assertEqual(float(force_mag), 0.0)
    
    def test_get_force_magnitude_tension_small(self):
        """Test force magnitude calculation in small tension."""
        # Create points with small tension (within transition region)
        small_tension_length = self.l_0 * 1.02  # 2% tension
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([small_tension_length, 0, 0], self.frame)
        
        spring = BlankevoortSpring(point1, point2, "small_tension_spring", 
                                 self.k, self.alpha, self.l_0)
        
        force_mag = spring.get_force_magnitude()
        
        # Should return a positive value for tension
        self.assertGreater(float(force_mag), 0)
    
    def test_get_force_magnitude_tension_large(self):
        """Test force magnitude calculation in large tension."""
        # Create points with large tension (beyond transition region)
        large_tension_length = self.l_0 * 1.1  # 10% tension
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([large_tension_length, 0, 0], self.frame)
        
        spring = BlankevoortSpring(point1, point2, "large_tension_spring", 
                                 self.k, self.alpha, self.l_0)
        
        force_mag = spring.get_force_magnitude()
        
        # Should return a positive value for tension
        self.assertGreater(float(force_mag), 0)
    
    def test_get_force_magnitude_at_rest(self):
        """Test force magnitude calculation at rest length."""
        # Create points at rest length
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([self.l_0, 0, 0], self.frame)
        
        spring = BlankevoortSpring(point1, point2, "rest_spring", 
                                 self.k, self.alpha, self.l_0)
        
        force_mag = spring.get_force_magnitude()
        
        # Should return 0 at rest length
        self.assertEqual(float(force_mag), 0.0)
    
    def test_get_force_magnitude_symbolic(self):
        """Test force magnitude calculation with symbolic parameters."""
        k, alpha, l_0 = symbols('k alpha l_0')
        spring = BlankevoortSpring(self.point1, self.point2, "symbolic_spring", 
                                 k, alpha, l_0)
        
        # For symbolic parameters, we need to use the sympy implementation
        # The BlankevoortFunction's __call__ method uses numpy which can't handle symbolic comparisons
        current_length = spring.get_spring_length()
        sympy_expr = spring.function.sympy_implementation()
        
        # Substitute the current length into the symbolic expression
        x = symbols('x')
        force_mag = sympy_expr.subs(x, current_length)
        
        # Should return a symbolic expression
        self.assertIsInstance(force_mag, sympy.Expr)
    
    def test_inheritance_from_abstract_spring(self):
        """Test that BlankevoortSpring inherits from AbstractSpring correctly."""
        spring = BlankevoortSpring(self.point1, self.point2, "inheritance_test", 
                                 self.k, self.alpha, self.l_0)
        
        # Test inherited methods
        p1, p2 = spring.get_points()
        self.assertEqual(p1.coordinates, self.point1.coordinates)
        self.assertEqual(p2.coordinates, self.point2.coordinates)
        
        # Test spring length
        length = spring.get_spring_length()
        expected_length = (self.point2 - self.point1).norm()
        self.assertEqual(length, expected_length)
        
        # Test force calculation
        force_p1 = spring.get_force_on_point1()
        force_p2 = spring.get_force_on_point2()
        
        self.assertIsInstance(force_p1, Force)
        self.assertIsInstance(force_p2, Force)
        self.assertEqual(force_p1.name, "inheritance_test_p1")
        self.assertEqual(force_p2.name, "inheritance_test_p2")
    
    def test_substitute_solutions(self):
        """Test substituting solutions into spring."""
        # Create symbolic variables
        x, y, z = symbols('x y z')
        symbolic_point1 = Point([x, y, z], self.frame)
        symbolic_point2 = Point([x+1, y, z], self.frame)
        
        spring = BlankevoortSpring(symbolic_point1, symbolic_point2, "symbolic", 
                                 self.k, self.alpha, self.l_0)
        
        # Substitute solutions
        solutions = {x: 1, y: 2, z: 3}
        spring.substitute_solutions(solutions)
        
        # Check that coordinates were substituted
        self.assertEqual(spring.point_1.coordinates, Matrix([1, 2, 3]))
        self.assertEqual(spring.point_2.coordinates, Matrix([2, 2, 3]))
    
    def test_different_parameter_combinations(self):
        """Test with different parameter combinations."""
        test_cases = [
            (10.0, 0.01, 30.0),  # Low k, small alpha, small l_0
            (100.0, 0.1, 100.0),  # High k, large alpha, large l_0
            (50.0, 0.05, 75.0),   # Medium values
        ]
        
        for k, alpha, l_0 in test_cases:
            with self.subTest(k=k, alpha=alpha, l_0=l_0):
                spring = BlankevoortSpring(self.point1, self.point2, f"test_{k}_{alpha}_{l_0}", 
                                         k, alpha, l_0)
                
                self.assertEqual(spring.k, k)
                self.assertEqual(spring.alpha, alpha)
                self.assertEqual(spring.l_0, l_0)
                
                # Test that force calculation works
                force_mag = spring.get_force_magnitude()
                self.assertTrue(isinstance(force_mag, (sympy.Expr, int, float)))


class TestBlankevoortSpringEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for BlankevoortSpring."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.frame = ReferenceFrame("test_frame")
        self.frame.set_as_ground_frame()
        self.point1 = Point([0, 0, 0], self.frame)
        self.point2 = Point([1, 0, 0], self.frame)
    
    def test_zero_length_spring(self):
        """Test spring with zero length."""
        point1 = Point([0, 0, 0], self.frame)
        point2 = Point([0, 0, 0], self.frame)  # Same point
        
        spring = BlankevoortSpring(point1, point2, "zero_spring", 60.0, 0.06, 50.0)
        
        # Force magnitude should be 0 (compression)
        force_mag = spring.get_force_magnitude()
        self.assertEqual(float(force_mag), 0.0)
    
    def test_very_small_parameters(self):
        """Test spring with very small parameters."""
        spring = BlankevoortSpring(self.point1, self.point2, "small_params", 
                                 1e-10, 1e-6, 1e-3)
        
        force_mag = spring.get_force_magnitude()
        self.assertTrue(isinstance(force_mag, (sympy.Expr, int, float)))
    
    def test_very_large_parameters(self):
        """Test spring with very large parameters."""
        spring = BlankevoortSpring(self.point1, self.point2, "large_params", 
                                 1e10, 1.0, 1e6)
        
        force_mag = spring.get_force_magnitude()
        self.assertTrue(isinstance(force_mag, (sympy.Expr, int, float)))
    
    def test_symbolic_parameters(self):
        """Test spring with symbolic parameters."""
        k, alpha, l_0 = symbols('k alpha l_0')
        spring = BlankevoortSpring(self.point1, self.point2, "symbolic", k, alpha, l_0)
        
        # For symbolic parameters, we need to use the sympy implementation
        current_length = spring.get_spring_length()
        sympy_expr = spring.function.sympy_implementation()
        x = symbols('x')
        force_mag = sympy_expr.subs(x, current_length)
        
        # Test that we can get energy (inherited from AbstractSpring)
        # Note: AbstractSpring doesn't implement get_energy, so this will raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            spring.get_energy()
        
        self.assertIsInstance(force_mag, sympy.Expr)
    
    def test_ligament_function_parameter_extraction(self):
        """Test that parameters are correctly extracted from ligament function."""
        # Test with different parameter values
        params = np.array([75.0, 0.08, 45.0, 5.0])
        ligament_func = BlankevoortFunction(params)
        
        spring = BlankevoortSpring.from_ligament_function(
            self.point1, self.point2, "param_test", ligament_func)
        
        # Check that parameters match (excluding f_ref which is not stored)
        self.assertEqual(spring.k, 75.0)
        self.assertEqual(spring.alpha, 0.08)
        self.assertEqual(spring.l_0, 45.0)


if __name__ == '__main__':
    unittest.main()
