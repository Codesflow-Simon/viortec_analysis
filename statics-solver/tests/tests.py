import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mappings import *
from reference_frame import ReferenceFrame, Point

class TestReferenceFrames(unittest.TestCase):
    def test_reference_frames_x_rotation(self):
        # Create a ground frame
        world_frame = ReferenceFrame("WorldFrame")
        world_frame.set_as_ground_frame()
        
        # Create a point in the world frame
        point = Point(np.array([1, 2, 3]), world_frame)
        
        # Create a body frame rotated 90 degrees around x-axis relative to world
        body_frame = ReferenceFrame("BodyFrame")
        # 90 deg rotation around x axis - this rotates y to z, and z to -y
        rotation = RotationalMapping(matrix=np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]))
        translation = TranslationMapping(np.array([0, 0, 0]))
        body_in_world = RigidBodyMapping(rotation, translation)
        body_frame.add_parent(world_frame, body_in_world, body_to_world=True)
        
        # Convert the point from world frame to body frame
        point_in_body = point.convert_to_frame(body_frame)
        
        # Expected: [1, -3, 2]
        # Explanation: x stays the same, y becomes z, and z becomes -y
        self.assertAlmostEqual(point_in_body.coordinates[0], 1)
        self.assertAlmostEqual(point_in_body.coordinates[1], -3)
        self.assertAlmostEqual(point_in_body.coordinates[2], 2)
    
    def test_reference_frames_translation(self):
        # Create a ground frame
        world_frame = ReferenceFrame("WorldFrame")
        world_frame.set_as_ground_frame()
        
        # Create a point in the world frame
        point = Point(np.array([1, 2, 3]), world_frame)
        
        # Create a body frame translated by [5, 10, 15] relative to world
        body_frame = ReferenceFrame("BodyFrame")
        rotation = RotationalMapping(matrix=np.eye(3))  # Identity rotation
        translation = TranslationMapping(np.array([5, 10, 15]))
        body_in_world = RigidBodyMapping(rotation, translation)
        body_frame.add_parent(world_frame, body_in_world, body_to_world=True)
        
        # Convert the point from world frame to body frame
        point_in_body = point.convert_to_frame(body_frame)
        
        # Expected: [1, 2, 3] - [5, 10, 15] = [-4, -8, -12]
        # When converting point from world to body, we subtract the body's position in world
        self.assertTrue(np.allclose(point_in_body.coordinates, np.array([-4, -8, -12])))
    
    def test_reference_frames_chained(self):
        # Create a chain of reference frames: World -> A -> B
        world_frame = ReferenceFrame("WorldFrame")
        world_frame.set_as_ground_frame()
        
        # Frame A: translated 5 units along x from world
        frame_a = ReferenceFrame("FrameA")
        a_rotation = RotationalMapping(matrix=np.eye(3))
        a_translation = TranslationMapping(np.array([5, 0, 0]))
        a_in_world = RigidBodyMapping(a_rotation, a_translation)
        frame_a.add_parent(world_frame, a_in_world, body_to_world=True)
        
        # Frame B: rotated 90 degrees around z from A
        frame_b = ReferenceFrame("FrameB")
        # 90 deg rotation around z - this rotates x to -y, and y to x
        b_rotation = RotationalMapping(matrix=np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ]))
        b_translation = TranslationMapping(np.array([0, 3, 0]))
        b_in_a = RigidBodyMapping(b_rotation, b_translation)
        frame_b.add_parent(frame_a, b_in_a, body_to_world=True)
        
        # Create a point in the world frame
        point = Point(np.array([10, 20, 30]), world_frame)
        
        # Convert point from world to B
        point_in_b = point.convert_to_frame(frame_b)
        
        # World to A: [10, 20, 30] -> [10-5, 20, 30] = [5, 20, 30]
        # A to B: 
        #   First subtract translation: [5, 20, 30] - [0, 3, 0] = [5, 17, 30]
        #   Then apply inverse rotation: 
        #   [0, 1, 0]   [5]   [ 17]
        #   [-1, 0, 0] * [17] = [-5]
        #   [0, 0, 1]   [30]   [30]
        self.assertTrue(np.allclose(point_in_b.coordinates, np.array([17, -5, 30])))
    
    def test_reference_frames_reverse_conversion(self):
        # Create a ground frame
        world_frame = ReferenceFrame("WorldFrame")
        world_frame.set_as_ground_frame()
        
        # Create a body frame with rotation and translation
        body_frame = ReferenceFrame("BodyFrame")
        rotation = RotationalMapping(matrix=np.array([
            [0, 0, 1],  # x -> z
            [1, 0, 0],  # y -> x
            [0, 1, 0]   # z -> y
        ]))
        translation = TranslationMapping(np.array([5, 10, 15]))
        body_in_world = RigidBodyMapping(rotation, translation)
        body_frame.add_parent(world_frame, body_in_world, body_to_world=True)
        
        # Create a point in the body frame
        point_in_body = Point(np.array([1, 2, 3]), body_frame)
        
        # Convert the point from body frame to world frame
        point_in_world = point_in_body.convert_to_frame(world_frame)
        
        # Body to World:
        # First apply rotation back:
        # [0, 1, 0]   [1]   [2]
        # [0, 0, 1] * [2] = [3]
        # [1, 0, 0]   [3]   [1]
        # Then add translation: [2, 3, 1] + [5, 10, 15] = [7, 13, 16]
        self.assertTrue(np.allclose(point_in_world.coordinates, np.array([7, 13, 16])))
    
    def test_point_constructor_shape(self):
        world_frame = ReferenceFrame("WorldFrame")
        world_frame.set_as_ground_frame()
        
        # Test with list input
        point1 = Point([1, 2, 3], world_frame) 
        self.assertTrue(np.allclose(point1.coordinates, np.array([1, 2, 3])))
        
        # Test with numpy array input
        point2 = Point(np.array([4, 5, 6]), world_frame)
        self.assertTrue(np.allclose(point2.coordinates, np.array([4, 5, 6])))
        
        # Test with invalid shape
        with self.assertRaises(ValueError):
            Point([1, 2], world_frame)  # Only 2 coordinates

if __name__ == "__main__":
    unittest.main()
