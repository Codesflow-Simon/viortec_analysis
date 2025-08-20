import sympy
from sympy import Expr # For type hinting
from .reference_frame import Point
import warnings # For issuing warnings
from .rigid_body import Force
from ligament_models import BlankevoortFunction

class AbstractSpring:
    def __init__(self, point_1: Point, point_2: Point, name: str):
        self.point_1_orignal_frame = point_1.reference_frame # We use this for reporting later
        self.point_2_orignal_frame = point_2.reference_frame

        if point_1.reference_frame != point_2.reference_frame:
            point_2 = point_2.convert_to_frame(point_1.reference_frame)

        self.name = name
        self.set_points(point_1, point_2)

    def set_points(self, point_1: Point=None, point_2: Point=None):
        if point_1 is not None:
            self.point_1 = point_1.convert_to_frame(self.point_1_orignal_frame)
        if point_2 is not None:
            self.point_2 = point_2.convert_to_frame(self.point_1_orignal_frame)

    def get_points(self):
        return self.point_1, self.point_2

    def get_force_direction_on_p2(self) -> Point:
        """
        Returns the unit vector Pointing from point_1 towards point_2.
        The force on point_2 will be along this direction if in tension, or opposite if in compression.
        """
        direction_vector_obj = self.point_1 - self.point_2
        norm_val = direction_vector_obj.norm()

        # Assuming direction_vector_obj.coordinates returns a 3x1 sympy.Matrix
        # Access components for piecewise definition
        dx = direction_vector_obj.coordinates[0,0]
        dy = direction_vector_obj.coordinates[1,0]
        dz = direction_vector_obj.coordinates[2,0]

        # Define each coordinate of the normalized vector using Piecewise
        # If norm_val is 0, the component is 0; otherwise, it's component / norm_val.
        # px = sympy.Piecewise((dx / norm_val, sympy.Ne(norm_val, 0)), (sympy.S.Zero, True))
        # py = sympy.Piecewise((dy / norm_val, sympy.Ne(norm_val, 0)), (sympy.S.Zero, True))
        # pz = sympy.Piecewise((dz / norm_val, sympy.Ne(norm_val, 0)), (sympy.S.Zero, True))
        px = dx / norm_val
        py = dy / norm_val
        pz = dz / norm_val
        
        # The warning for zero length is harder to emit conditionally in a purely symbolic way.
        # Consider if sympy.functions.elementary.miscellaneous.Min(1, norm_val) or similar could be used
        # to suppress warnings. For now, focusing on numerical stability.
        # if sympy.Eq(norm_val, 0) == sympy.true: # This comparison itself is tricky
        #    warnings.warn(f"Spring '{self.name}' has zero length; force direction is undefined, using zero vector.")

        piecewise_coord_matrix = sympy.Matrix([px, py, pz])
        
        # Create the Point with the symbolically safe coordinates
        # The frame should be the original frame of the direction vector before conversion
        result_point = Point(piecewise_coord_matrix, self.point_1.reference_frame)
        
        # Convert to the target frame as in the original logic
        result_point = result_point.convert_to_frame(self.point_2_orignal_frame)
        return result_point.normalize()

    def get_force_direction_on_p1(self) -> Point:
        """
        Returns the unit vector Pointing from point_2 towards point_1.
        The force on point_1 will be along this direction if in tension, or opposite if in compression.
        """
        direction_vector = -self.get_force_direction_on_p2()
        direction_vector = direction_vector.convert_to_frame(self.point_1_orignal_frame)
        return direction_vector.normalize()

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

    def get_spring_length(self) -> Expr:
        """Returns the symbolic length of the spring."""
        return (self.point_2 - self.point_1).norm()

    def get_force_magnitude(self) -> Expr:
        """Returns the symbolic magnitude of the spring force. Positive for tension, negative for compression."""
        raise NotImplementedError("Subclasses must implement this method")

    def get_energy(self) -> Expr:
        """Returns the symbolic energy stored in the spring."""
        raise NotImplementedError("Subclasses must implement this method")

    def substitute_solutions(self, solutions):
        """
        Substitutes the solutions into the spring.
        """
        self.point_1.substitute_solutions(solutions)
        self.point_2.substitute_solutions(solutions)


class LinearSpring(AbstractSpring):
    def __init__(self, point_1: Point, point_2: Point, name: str, k: [float, Expr], x0: [float, Expr]):
        super().__init__(point_1, point_2, name)

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
            
        self.k = k
        self.x0 = x0

    def get_force_magnitude(self) -> Expr:
        """Returns the symbolic magnitude of the spring force. Positive for tension."""
        current_length = self.get_spring_length()
        strain = (current_length - self.x0) / self.x0
        return self.k * strain

    def get_energy(self) -> Expr:
        """Returns the symbolic energy stored in the spring."""
        current_length = self.get_spring_length()
        return 0.5 * self.k * (current_length - self.x0)**2

class TriLinearSpring(AbstractSpring):
    def __init__(self, point_1: Point, point_2: Point, name: str, k_1: [float, Expr], k_2: [float, Expr], k_3: [float, Expr],  x_0: [float, Expr], a_1: [float, Expr], a_2: [float, Expr]):
        """
        A spring that has three different spring constants, and a different rest length for each spring.
        Consider the spring strain, x, which is zero at rest length x0.
        When x < a, the spring constant is k_1.
        When a < x < b, the spring constant is k_2.
        When x > b, the spring constant is k_3.
        Note we consider that a,b could be postive (tension) or negative (compression).
        """
        
        super().__init__(point_1, point_2, name)
        
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.x_0 = x_0
        self.a_1 = a_1
        self.a_2 = a_2

    def get_force_magnitude(self) -> Expr:
        """Returns the symbolic magnitude of the spring force. Positive for tension, negative for compression."""
        current_length = self.get_spring_length()
        elongation = current_length - self.x_0
        return sympy.Piecewise(
            (0, elongation < 0),
            (self.k_1 * elongation, elongation < self.a_1),
            (self.k_1 * self.a_1 + self.k_2 * (strain-self.a_1), strain >= self.a_1),
            (self.k_1 * self.a_1 + self.k_2 * (self.a_2 - self.a_1) + self.k_3 * (strain-self.a_2), strain >= self.a_2),
        )

class BlankevoortSpring(AbstractSpring):
    def __init__(self, point_1: Point, point_2: Point, name: str, k: [float, Expr], alpha: [float, Expr], l_0: [float, Expr]):
        """
        A spring that has a transition length, and a different spring constant for each transition length.
        Consider the spring strain, x, which is zero at rest length x0.
        When x < alpha, the spring constant is k.
        """
        from ligament_models import BlankevoortFunction
        self.function = BlankevoortFunction([k, alpha, l_0, 0])
        
        super().__init__(point_1, point_2, name)
        self.alpha = alpha
        self.k = k
        self.l_0 = l_0

    @staticmethod
    def from_ligament_function(point_1: Point, point_2: Point, name: str, ligament_function: BlankevoortFunction):
        params = ligament_function.get_params()
        spring = BlankevoortSpring(point_1, point_2, name, params[0], params[1], params[2])
        return spring

    def get_force_magnitude(self) -> Expr:
        """Returns the symbolic magnitude of the spring force. Positive for tension, negative for compression."""
        current_length = self.get_spring_length()
        result = self.function(current_length)
        return result
