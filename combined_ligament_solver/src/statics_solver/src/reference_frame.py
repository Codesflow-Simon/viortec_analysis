import numpy as np
from .mappings import *
import sympy

class ReferenceFrame:
    def __init__(self, name):
        """
        Initialize a reference frame.
        
        Args:
            name (str): Name of the reference frame
            parent (ReferenceFrame, optional): Parent reference frame. If None, this is the ground frame
            rigid_body_mapping (RigidBodyMapping, optional): Mapping to parent frame. Required if parent is not None
        """
        self.name = name
        self.is_ground_frame = None
        self.parent = None
        self.children = []
        self.rigid_body_mapping = None

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return hash((self.name, self.is_ground_frame, self.parent, tuple(self.children), self.rigid_body_mapping))

    def set_as_ground_frame(self):
        if self.parent is not None:
            raise ValueError("Cannot set a non-ground frame as the ground frame")
        self.is_ground_frame = True

    def add_parent(self, parent: 'ReferenceFrame', mapping_to_parent: RigidBodyMapping, body_to_world=False):
        # Mapping is world_in_body, so body_to_world=True means body_in_world
        if self.parent is not None:
            raise ValueError("Cannot add a parent to a frame that already has a parent")
        self.parent = parent
        self.rigid_body_mapping = mapping_to_parent.get_inverse() if body_to_world else mapping_to_parent

    def get_rigid_body_mapping_safe(self):
        if self.is_ground_frame:
            return IdentityMapping()
        return self.rigid_body_mapping

    def get_parent_safe(self):
        if self.is_ground_frame:
            return self
        return self.parent

    def find_common_ancestor(self, other: 'ReferenceFrame'):
        A = self
        B = other

        set_A = set([self])
        set_B = set([other])

        transform_A = []
        transform_B = []

        while len(set_A.intersection(set_B)) == 0:
            if A.parent is None and B.parent is None:
                raise ValueError("Reference frames have no common ancestor")

            if A.parent is not None:
                set_A = set_A.union(set([A.parent]))
                transform_A.append(A.get_rigid_body_mapping_safe())
                A = A.parent

            if B.parent is not None:
                set_B = set_B.union(set([B.parent]))
                transform_B.append(B.get_rigid_body_mapping_safe())
                B = B.parent

        intersection = set_A.intersection(set_B)
        if len(intersection) != 1:
            raise ValueError("Reference frames must have exactly one common ancestor")
        return list(intersection)[0], transform_A, transform_B            

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash((self.name, self.is_ground_frame, self.parent, tuple(self.children), self.rigid_body_mapping))

class Point:
    def __init__(self, coordinates, reference_frame):
        """
        Initialize a point in 3D space.
        
        Args:
            coordinates (list, tuple, np.ndarray, sympy.Matrix): 3D coordinates in the reference frame
            reference_frame (ReferenceFrame): The reference frame this point is defined in
        """
        if isinstance(coordinates, (list, tuple)):
            coordinates = sympy.Matrix(coordinates)
        elif isinstance(coordinates, np.ndarray): # Keep compatibility for a moment
            coordinates = sympy.Matrix(list(coordinates))
        elif not isinstance(coordinates, sympy.Matrix):
            raise TypeError("Coordinates must be a list, tuple, numpy array, or sympy.Matrix")

        if coordinates.shape != (3, 1) and coordinates.shape != (1, 3): # Allow row or column vector
             if coordinates.shape == (3,): # common numpy shape
                 coordinates = sympy.Matrix(coordinates).reshape(3,1)
             else:
                raise ValueError(f"Coordinates must be a 3D vector (3x1 or 1x3 Matrix), got shape {coordinates.shape}")
        
        if coordinates.shape == (1,3): # Ensure it's a column vector
            coordinates = coordinates.T

        self.coordinates = coordinates
        self.reference_frame = reference_frame

    def substitute_solutions(self, solutions):
        """Substitutes solutions into the coordinates of this point."""
        self.coordinates = self.coordinates.subs(solutions)
        
    def convert_to_parent(self):
        # Ensure matrix multiplication is handled correctly
        new_coords_matrix = self.reference_frame.rigid_body_mapping @ self.coordinates
        return Point(new_coords_matrix, self.reference_frame.parent)

    def convert_to_frame(self, target_frame: 'ReferenceFrame'):
        try:
            ancestor, transform_A, transform_B = self.reference_frame.find_common_ancestor(target_frame)
        except ValueError as e:
            raise ValueError(f"Cannot convert point {self} to frame {target_frame} because they have no common ancestor. {e}")
        
        current_point_coords = self.coordinates
        current_point_frame = self.reference_frame

        # Apply transforms from current frame up to common ancestor
        for transform in transform_A:
            current_point_coords = transform @ current_point_coords
            current_point_frame = current_point_frame.get_parent_safe()

        # Apply inverse transforms from target frame's path down to common ancestor in reverse
        
        # To bring from ancestor to target_frame, we need to apply the inverse of transforms B in order.
        # transform_B are transforms from target_frame up to ancestor.
        # So, to go from ancestor to target_frame, we apply their inverses in reverse order.

        # Let's trace the frames for the inverse transforms
        # Suppose target_frame -> P1 -> P2 -> ancestor. transform_B = [T_target_P1, T_P1_P2, T_P2_ancestor]
        # We have coords in 'ancestor'. We want them in 'target_frame'.
        # Coords_in_P2 = T_P2_ancestor_inv @ Coords_in_ancestor
        # Coords_in_P1 = T_P1_P2_inv @ Coords_in_P2
        # Coords_in_target = T_target_P1_inv @ Coords_in_P1

        temp_target_path = []
        curr_frame_for_path = target_frame
        while curr_frame_for_path != ancestor:
            temp_target_path.append(curr_frame_for_path.get_rigid_body_mapping_safe())
            curr_frame_for_path = curr_frame_for_path.get_parent_safe()
        
        for transform_inv_path in reversed(temp_target_path): # These are M_child_to_parent, apply inverse
             current_point_coords = transform_inv_path.inverse_apply(current_point_coords)
          

        return Point(current_point_coords, target_frame)

    def check_numeric(self):
        if not isinstance(self.coordinates, sympy.Matrix):
            return True
        if not self.coordinates.is_numeric():
            return False
        return True
    
    def _check_same_frame(self, other):
        """Check if both points are in the same reference frame"""
        if not isinstance(other, Point):
            raise TypeError(f"Unsupported operand type: {type(other)}. Operation only supported between Points.")
        
        if self.reference_frame != other.reference_frame:
            raise ValueError(f"Points must be in the same reference frame. Got {self.reference_frame.name} and {other.reference_frame.name}.")
        return True
    
    def __add__(self, other):
        """Add two points in the same reference frame"""
        self._check_same_frame(other)
        return Point(self.coordinates + other.coordinates, self.reference_frame)
    
    def __sub__(self, other):
        """Subtract two points in the same reference frame"""
        self._check_same_frame(other)
        return Point(self.coordinates - other.coordinates, self.reference_frame)
    
    def __mul__(self, scalar):
        """Multiply point coordinates by a scalar"""
        if not isinstance(scalar, (int, float, sympy.Expr)):
            raise TypeError(f"Can only multiply Point by scalar, not {type(scalar)}")
        return Point(self.coordinates * scalar, self.reference_frame)
    
    def __rmul__(self, scalar):
        """Right multiply point coordinates by a scalar"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Divide point coordinates by a scalar"""
        if not isinstance(scalar, (int, float, sympy.Expr)):
            raise TypeError(f"Can only divide Point by scalar, not {type(scalar)}")
        if scalar == 0:
            raise ZeroDivisionError("Cannot divide Point by zero")
        return Point(self.coordinates / scalar, self.reference_frame)
    
    def dot(self, other: 'Point'):
        """Compute dot product between two points in the same reference frame"""
        self._check_same_frame(other)
        # Sympy dot product for column matrices: A.T * B then extract scalar
        dot_product = (self.coordinates.T * other.coordinates)[0]
        return dot_product
    
    def cross(self, other: 'Point'):
        """Compute cross product between two points in the same reference frame"""
        self._check_same_frame(other)
        return Point(self.coordinates.cross(other.coordinates), self.reference_frame)
    
    def norm(self):
        """Compute the Euclidean norm (length) of the point vector"""
        return self.coordinates.norm()
    
    def normalize(self):
        """Return a normalized (unit) vector in the same direction"""
        norm = self.norm()
        if norm == 0: # SymPy might return literal 0 or symbolic 0
            # Need to be careful with symbolic comparison to zero
            if isinstance(norm, sympy.Number) and norm.is_zero:
                 raise ValueError("Cannot normalize zero vector")
            # If it's symbolic and could be zero, this is an issue.
            # For now, proceed, but this might need sympy.sympify(norm) == 0 or similar
            pass # Allow symbolic normalization if norm is not explicitly zero
        return Point(self.coordinates / norm, self.reference_frame)

    def __neg__(self):
        """Return the negation of this point vector"""
        return Point(-self.coordinates, self.reference_frame)
        
    def __str__(self):
        return f"Point in {self.reference_frame.name} frame: {self.coordinates.tolist()}" # Use tolist for nicer printing
    
    def __repr__(self):
        return f"Point in {self.reference_frame.name} frame: {self.coordinates.tolist()}"
    
    def __hash__(self):
        return hash((self.coordinates, self.reference_frame))

    
        