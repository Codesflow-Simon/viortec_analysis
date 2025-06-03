import sympy
from sympy import Expr # For type hinting
from reference_frame import Point
import warnings # For issuing warnings
from rigid_body import Force

class Spring:
    def __init__(self, point_1: Point, point_2: Point, name: str, k: [float, Expr], x0: [float, Expr]):

        self.point_1_orignal_frame = point_1.reference_frame # We use this for reporting later
        self.point_2_orignal_frame = point_2.reference_frame

        if point_1.reference_frame != point_2.reference_frame:
            point_2 = point_2.convert_to_frame(point_1.reference_frame)


        # Data validation
        if isinstance(k, (int, float)):
            if k <= 0:
                raise ValueError(f"Spring constant k for spring '{name}' must be numerically positive. Got {k}.")
        elif hasattr(k, 'is_positive'):
            if k.is_positive is sympy.false:
                warnings.warn(f"Spring constant k for spring '{name}' ({k}) is defined as non-positive. This is unusual.")
            elif k.is_positive is None and not (hasattr(k, 'is_zero') and k.is_zero is sympy.false):
                 warnings.warn(f"Spring constant k for spring '{name}' ({k}) has no positive assumption and could be zero.")
        if isinstance(x0, (int, float)):
            if x0 < 0:
                raise ValueError(f"Rest length x0 for spring '{name}' must be numerically non-negative. Got {x0}.")
        elif hasattr(x0, 'is_positive'):
            if x0.is_positive is sympy.false:
                 raise ValueError(f"Rest length x0 for spring '{name}' ({x0}) is defined as negative, which is not allowed.")
        elif hasattr(x0, 'is_zero') and x0.is_zero is sympy.true:
            pass
        elif hasattr(x0, 'is_negative') and x0.is_negative is sympy.true:
            raise ValueError(f"Rest length x0 for spring '{name}' ({x0}) is defined as negative, which is not allowed.")
            
        self.name = name
        self.k = k
        self.x0 = x0
        self.point_1 = point_1
        self.point_2 = point_2

    def get_force_magnitude(self) -> Expr:
        """Returns the symbolic magnitude of the spring force. Positive for tension, negative for compression."""
        current_length = (self.point_2 - self.point_1).norm()
        return self.k * (self.x0 - current_length)

    def get_force_direction_on_p2(self) -> Point:
        """
        Returns the unit vector Pointing from point_1 towards point_2.
        The force on point_2 will be along this direction if in tension, or opposite if in compression.
        """
        direction_vector = self.point_2 - self.point_1
        if direction_vector.norm().is_zero if hasattr(direction_vector.norm(), 'is_zero') else direction_vector.norm() == 0:
            warnings.warn(f"Spring '{self.name}' has zero length; force direction is undefined, returning zero vector.")
            return Point(sympy.zeros(3,1), self.point_1.reference_frame) 
        direction_vector = direction_vector.normalize()
        direction_vector = direction_vector.convert_to_frame(self.point_2_orignal_frame)
        return direction_vector

    def get_force_direction_on_p1(self) -> Point:
        """
        Returns the unit vector Pointing from point_2 towards point_1.
        The force on point_1 will be along this direction if in tension, or opposite if in compression.
        """
        direction_vector = -self.get_force_direction_on_p2()
        direction_vector = direction_vector.convert_to_frame(self.point_1_orignal_frame)
        return direction_vector

    def get_force_on_point2(self) -> Point:
        """Returns the symbolic force vector exerted BY the spring ON point_2."""
        magnitude = self.get_force_magnitude()
        direction_on_p2 = self.get_force_direction_on_p2() # from p1 to p2
        return Force(self.name + "_p2", magnitude * direction_on_p2, self.point_2)

    def get_force_on_point1(self) -> Point:
        """Returns the symbolic force vector exerted BY the spring ON point_1."""
        magnitude = self.get_force_magnitude()
        direction_on_p1 = self.get_force_direction_on_p1() # from p2 to p1
        return Force(self.name + "_p1", magnitude * direction_on_p1, self.point_1)