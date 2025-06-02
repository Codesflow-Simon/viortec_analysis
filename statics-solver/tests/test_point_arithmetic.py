import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path to import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference_frame import ReferenceFrame, Point
from mappings import RotationalMapping, TranslationMapping, RigidBodyMapping

class TestPointArithmetic(unittest.TestCase):
    def setUp(self):
        # Create a ground frame for testing
        self.frame1 = ReferenceFrame("Frame1")
        self.frame1.set_as_ground_frame()
        
        # Create a different frame
        self.frame2 = ReferenceFrame("Frame2")
        rotation = RotationalMapping(matrix=np.eye(3))
        translation = TranslationMapping(np.array([1, 1, 1]))
        self.frame2.add_parent(self.frame1, RigidBodyMapping(rotation, translation))
        
        # Create test points
        self.p1 = Point(np.array([1, 2, 3]), self.frame1)
        self.p2 = Point(np.array([4, 5, 6]), self.frame1)
        self.p3 = Point(np.array([7, 8, 9]), self.frame2)  # Different frame

    def test_point_addition(self):
        # Test valid addition (same frame)
        result = self.p1 + self.p2
        expected = np.array([5, 7, 9])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test invalid addition (different frames)
        with self.assertRaises(ValueError):
            result = self.p1 + self.p3

    def test_point_subtraction(self):
        # Test valid subtraction (same frame)
        result = self.p2 - self.p1
        expected = np.array([3, 3, 3])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test invalid subtraction (different frames)
        with self.assertRaises(ValueError):
            result = self.p2 - self.p3

    def test_scalar_multiplication(self):
        # Test scalar multiplication
        result = self.p1 * 2
        expected = np.array([2, 4, 6])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test right multiplication
        result = 3 * self.p1
        expected = np.array([3, 6, 9])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test invalid multiplication
        with self.assertRaises(TypeError):
            result = self.p1 * "string"

    def test_scalar_division(self):
        # Test scalar division
        result = self.p1 / 2
        expected = np.array([0.5, 1, 1.5])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test division by zero
        with self.assertRaises(ZeroDivisionError):
            result = self.p1 / 0
            
        # Test invalid division
        with self.assertRaises(TypeError):
            result = self.p1 / "string"

    def test_dot_product(self):
        # Test valid dot product (same frame)
        result = self.p1.dot(self.p2)
        expected = 1*4 + 2*5 + 3*6  # 32
        self.assertEqual(result, expected)
        
        # Test invalid dot product (different frames)
        with self.assertRaises(ValueError):
            result = self.p1.dot(self.p3)

    def test_cross_product(self):
        # Test valid cross product (same frame)
        result = self.p1.cross(self.p2)
        # Cross product of [1,2,3] × [4,5,6] = [-3, 6, -3]
        expected = np.array([-3, 6, -3])
        self.assertTrue(np.array_equal(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test invalid cross product (different frames)
        with self.assertRaises(ValueError):
            result = self.p1.cross(self.p3)

    def test_norm(self):
        # Test norm calculation
        result = self.p1.norm()
        expected = np.sqrt(1**2 + 2**2 + 3**2)  # sqrt(14)
        self.assertAlmostEqual(result, expected)

    def test_normalize(self):
        # Test normalization
        result = self.p1.normalize()
        expected = self.p1.coordinates / self.p1.norm()
        self.assertTrue(np.allclose(result.coordinates, expected))
        self.assertEqual(result.reference_frame, self.frame1)
        
        # Test that result is a unit vector
        self.assertAlmostEqual(result.norm(), 1.0)
        
        # Test zero vector normalization
        zero_point = Point(np.array([0, 0, 0]), self.frame1)
        with self.assertRaises(ValueError):
            zero_point.normalize()

    def test_convert_and_operate(self):
        # Convert point to same frame as p3 then add
        p1_in_frame2 = self.p1.convert_to_frame(self.frame2)
        result = p1_in_frame2 + self.p3
        
        # Check result
        self.assertEqual(result.reference_frame, self.frame2)
        expected = p1_in_frame2.coordinates + self.p3.coordinates
        self.assertTrue(np.array_equal(result.coordinates, expected))

if __name__ == "__main__":
    unittest.main() 