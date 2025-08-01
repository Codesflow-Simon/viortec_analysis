import numpy as np
import sympy
from mappings import RotationalMapping, TranslationMapping, RigidBodyMapping
from reference_frame import Point, ReferenceFrame
from rigid_body import Force

class JointModel(RigidBodyMapping):
    def __init__(self):
        pass

    def set_mapping(self, rotation: RotationalMapping, translation: TranslationMapping):
        self.rotation = rotation
        self.translation = translation

    def set_theta(self, theta):
        raise NotImplementedError("Subclass must implement set_theta")

    def get_contact_point(self):
        raise NotImplementedError("Subclass must implement get_contact_points")

    def get_constraint_force(self):
        raise NotImplementedError("Subclass must implement get_constraint_force")

class PivotJoint(JointModel):
    def __init__(self, body_frame: ReferenceFrame, world_frame: ReferenceFrame):
        super().__init__()
        self.body_frame = body_frame
        self.world_frame = world_frame

    def set_theta(self, theta):
        self.theta = theta
        rotation = RotationalMapping.from_euler_angles([0, 0, theta])
        translation = TranslationMapping(np.array([0, 0, 0]))
        self.set_mapping(rotation, translation)

    def get_contact_point(self):
        return Point([0, 0, 0], self.world_frame)

    def get_constraint_force(self):
        contact_point = self.get_contact_point()
        force_x = sympy.Symbol(f'Pivot_x')
        force_y = sympy.Symbol(f'Pivot_y')
        force_vector = sympy.Matrix([force_x, force_y, 0])
        constraint_forces = Force(f"PivotConstraintForce", Point(force_vector, self.world_frame), contact_point)
        unknown_list = [force_x, force_y]
        return constraint_forces, unknown_list

class TwoBallJoint(JointModel):
    """
    A joint made of two balls and a frame that rocks over them. This defines the position of the child frame.
    Args:
        body_frame: The child frame of the joint.
        world_frame: The world frame of the joint.
        distance: The horizontal distance from the center of the joint to ball center.
        distance_2: Optional distance from the center of the joint to the other ball center, otherwise symmetric.
        radius: The radius of the balls.
        slide: If the joint is sliding, rather than rolling.
    """
    def __init__(self, body_frame: ReferenceFrame, world_frame: ReferenceFrame, distance, distance_2=None, radius=None, slide=True):
        self.body_frame = body_frame
        self.world_frame = world_frame
        self.distance = distance
        self.distance_2 = distance_2 if distance_2 is not None else distance
        self.radius = radius
        self.slide = slide # Else the joint becomes a rolling joint.

        self.set_theta(0.0)

    def get_ball_centers(self, theta):
        radius = self.radius
        return [Point([-self.distance,  -radius, 0], self.world_frame), 
                Point([self.distance_2, -radius, 0], self.world_frame)]

    def _get_distance_for_theta(self, theta):
        """Helper method to get the appropriate distance based on theta sign
        distance here means the distance from the femur (x=0) to the top of the ball, since this may not\
        be symmetric, we define it piecewise.
        """
        if hasattr(theta, 'is_symbol'):
            # Symbolic theta - use SymPy Piecewise
            from sympy import Piecewise, GreaterThan
            return Piecewise(
                (-self.distance, GreaterThan(theta, 0)),
                (self.distance_2, True)  # Default case for theta <= 0
            )
        else:
            # Numeric theta - use conditional logic
            return -self.distance if theta > 0 else self.distance_2

    def set_theta(self, theta):
        self.theta = theta
        radius = self.radius
        distance = self._get_distance_for_theta(theta)

        # Next move the contact point to new location
        old_contact_to_new = Point([-radius*sympy.sin(theta), radius*(1-sympy.cos(theta)), 0], self.world_frame)

        origin_to_new_contact = self.get_contact_point(theta)

        rotation = RotationalMapping.from_euler_angles([0, 0, -theta])

        # translation_vector = (old_contact_to_new-origin_to_new_contact).coordinates.reshape(3,1) + rotation.inverse_apply(origin_to_new_contact.coordinates)
        translation_vector = rotation.apply((old_contact_to_new-origin_to_new_contact).coordinates) + origin_to_new_contact.coordinates.reshape(3,1)
        
        translation = TranslationMapping(translation_vector)

        self.set_mapping(rotation, translation)

    def get_contact_point(self, theta):
        radius = self.radius
        distance = self._get_distance_for_theta(theta)


        ball_angle = np.array([-radius*sympy.sin(theta), -radius*sympy.cos(theta), 0])
        ball_center = np.array([distance, -radius, 0])
        return Point(ball_angle + ball_center, self.world_frame)

    def get_constraint_force(self):
        contact_point = self.get_contact_point(self.theta)
        force_x = sympy.Symbol(f'TwoBallConstraintForce_x')
        force_y = sympy.Symbol(f'TwoBallConstraintForce_y')
        force_vector = sympy.Matrix([force_x, force_y, 0])
        constraint_forces = Force(f"TwoBallConstraintForce", Point(force_vector, self.world_frame), contact_point)
        unknown_list = [force_x, force_y]
        return constraint_forces, unknown_list

    def __str__(self):
        return f"Two-ball-mapping: {self.mapping}"
    
    def __repr__(self):
        return f"Two-ball-mapping: {self.mapping}"

    
