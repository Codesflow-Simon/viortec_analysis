import sympy
from sympy import Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt
import yaml

from ..src.mappings import *
from ..src.reference_frame import *
from ..src.rigid_body import *
from ..src.springs import *
from ..src.visualiser import *
from ..src.joint_models import TwoBallJoint, PivotJoint, AsymmetricTwoBallJoint
from .base import AbstractModel

class KneeModel(AbstractModel):
    def __init__(self, data, lig_function_left: BlankevoortFunction, lig_function_right: BlankevoortFunction, log=True):
        self.log = log
        self.data = data
        self.lig_function_left = lig_function_left
        self.lig_function_right = lig_function_right
        
        # Caching flags for optimization
        self._cached_geometry = None
        self._cached_ligament_params = None
        self._cached_theta = None
        self._linear_system_cache = None
        
        self.build_model()

    def update_data(self, data):
        self.data = data
        self.build_model()

    def build_model(self):
        """Build the complete model. This is the main entry point that delegates to sub-methods."""
        self._build_geometry()
        self._build_ligament_forces()
        self._assemble_equations()
        
        # Mark everything as cached
        self._cached_geometry = True
        self._cached_ligament_params = (self.lig_function_left.get_params().copy(), self.lig_function_right.get_params().copy())
        self._cached_theta = self.data['theta']
        self._linear_system_cache = None  # Will be computed on first solve

    def _build_geometry(self):
        """Build the geometric structure (frames, points, joint) based on theta and config."""
        # Some constants
        femur_perp = self.data['femur_perp']
        femur_length = self.data['femur_length']
        tibia_perp = self.data['tibia_perp']
        tibia_para = self.data['tibia_para']
        application_length = self.data['application_length']
        theta_val = self.data['theta']
        left_length = self.data['left_length']
        right_length = self.data['right_length']
        ball_radius_1 = self.data['ball_radius_1']
        ball_radius_2 = self.data['ball_radius_2']
        ball_distance = self.data['ball_distance']
        ball_distance_2 = self.data['ball_distance_2']

        # Frames
        self.world_frame = ReferenceFrame("WorldFrame")
        self.world_frame.set_as_ground_frame()
        self.tibia_frame = ReferenceFrame("TibiaFrame")

        self.ball_distance = ball_distance
        self.ball_distance_2 = ball_distance_2
        self.ball_radius_1 = ball_radius_1
        self.ball_radius_2 = ball_radius_2
        self.knee_joint = AsymmetricTwoBallJoint(self.tibia_frame, self.world_frame, distance=self.ball_distance, distance_2=self.ball_distance_2, radius_1=self.ball_radius_1, radius_2=self.ball_radius_2)
        self.knee_joint.set_theta(theta_val)

        self.tibia_frame.add_parent(self.world_frame, self.knee_joint)

        # Rigid bodies
        self.femur_body = RigidBody("Femur", self.world_frame)
        self.tibia_body = RigidBody("Tibia", self.tibia_frame)

        left_attatchment_height = left_length - tibia_para
        right_attatchment_height = right_length - tibia_para

        # Points of interest
        self.hip_point = Point([0, femur_length, 0], self.world_frame)
        self.knee_point = Point([0, 0, 0], self.world_frame)
        self.joint_ball_A = Point([self.ball_distance, -self.ball_radius_1, 0], self.world_frame)
        self.joint_ball_B = Point([-self.ball_distance, -self.ball_radius_2, 0], self.world_frame)
        self.lig_top_pointA = Point([femur_perp,0, 0], self.world_frame)
        self.lig_top_pointB = Point([-femur_perp,0, 0], self.world_frame)

        self.lig_on_tib_vis = Point([0, -tibia_para, 0], self.tibia_frame)
        self.lig_bottom_pointA = Point([tibia_perp, -tibia_para - left_attatchment_height, 0], self.tibia_frame)
        self.lig_bottom_pointB = Point([-tibia_perp, -tibia_para - right_attatchment_height, 0], self.tibia_frame)
        self.lig_bottom_pointA_flat = Point([tibia_perp, -tibia_para, 0], self.tibia_frame)
        self.lig_bottom_pointB_flat = Point([-tibia_perp, -tibia_para, 0], self.tibia_frame)
        self.application_point = Point([0, -application_length, 0], self.tibia_frame)

    def _build_ligament_forces(self):
        """Build ligament springs and applied forces."""
        app_Fx = self.data['app_Fx']

        # Springs
        self.lig_springA = BlankevoortSpring.from_ligament_function(self.lig_top_pointA, self.lig_bottom_pointA, "LigSpringA", self.lig_function_left)
        self.lig_springB = BlankevoortSpring.from_ligament_function(self.lig_top_pointB, self.lig_bottom_pointB, "LigSpringB", self.lig_function_right)

        # Applied force: Unknown force, no torques transferred
        self.force_vec_sym = [app_Fx,0 ,0 ]
        self.force_vector = Point(self.force_vec_sym, self.tibia_frame)
        self.applied_force = Force("AppliedForce", self.force_vector, self.application_point)

    def _assemble_equations(self):
        """Assemble the force and moment balance equations."""
        # Constraint forces
        self.constraint_force, self.constraint_unknowns = self.knee_joint.get_constraint_force()
        if self.log:
            print(f"Constraint force: {self.constraint_force}")

        self.tibia_body.add_force_pair(self.constraint_force, self.femur_body)

        # Register spring forces on the bodies
        self.femur_body.add_external_force(self.lig_springA.get_force_on_point1())
        self.tibia_body.add_external_force(self.lig_springA.get_force_on_point2())

        self.femur_body.add_external_force(self.lig_springB.get_force_on_point1())
        self.tibia_body.add_external_force(self.lig_springB.get_force_on_point2())

        self.tibia_body.add_external_force(self.applied_force)

    def solve(self):
        """Solve the linear system using direct numpy linear algebra instead of sympy.solve()."""
        # Check if we need to rebuild the linear system
        if self._linear_system_cache is None or self._needs_linear_system_rebuild():
            self._build_linear_system()
        
        # Get ligament elongations and forces (numeric)
        lig_a_length = float(self.lig_springA.get_spring_length())
        lig_b_length = float(self.lig_springB.get_spring_length())
        
        # Evaluate ligament forces using the ligament functions
        lig_a_force_magnitude = float(self.lig_function_left(lig_a_length))
        lig_b_force_magnitude = float(self.lig_function_right(lig_b_length))
        
        # Get ligament force directions (unit vectors)
        lig_a_direction = self.lig_springA.get_force_direction_on_p2()
        lig_b_direction = self.lig_springB.get_force_direction_on_p2()
        
        # Convert to numeric vectors
        lig_a_force_vec = np.array([float(lig_a_direction.coordinates[i]) for i in range(3)]) * lig_a_force_magnitude
        lig_b_force_vec = np.array([float(lig_b_direction.coordinates[i]) for i in range(3)]) * lig_b_force_magnitude
        
        # Get application point and contact point for moment calculations
        contact_point = self.knee_joint.get_contact_point(theta=self.data['theta'])
        contact_point_tibia = contact_point.convert_to_frame(self.tibia_frame)
        
        # Convert points to numeric arrays
        app_point_vec = np.array([float(self.application_point.coordinates[i]) for i in range(3)])
        contact_point_vec = np.array([float(contact_point_tibia.coordinates[i]) for i in range(3)])
        lig_a_point_vec = np.array([float(self.lig_bottom_pointA.coordinates[i]) for i in range(3)])
        lig_b_point_vec = np.array([float(self.lig_bottom_pointB.coordinates[i]) for i in range(3)])
        
        # Build the linear system Ax = b
        # Unknowns: [constraint_force_x, constraint_force_y, app_Fx]
        A = np.zeros((6, 3))
        b = np.zeros(6)
        
        # Force balance equations (3 equations)
        # Sum of forces = 0: constraint_force + applied_force + ligament_forces = 0
        A[0, 0] = 1.0  # constraint_force_x coefficient
        A[0, 2] = 1.0  # app_Fx coefficient  
        b[0] = -(lig_a_force_vec[0] + lig_b_force_vec[0])  # ligament forces in x
        
        A[1, 1] = 1.0  # constraint_force_y coefficient
        b[1] = -(lig_a_force_vec[1] + lig_b_force_vec[1])  # ligament forces in y
        
        b[2] = -(lig_a_force_vec[2] + lig_b_force_vec[2])  # ligament forces in z
        
        # Moment balance equations (3 equations)
        # Sum of moments = 0: r_constraint × constraint_force + r_app × applied_force + ligament_moments = 0
        # Moment = r × F, so for r = [rx, ry, rz] and F = [Fx, Fy, Fz]:
        # M = [ry*Fz - rz*Fy, rz*Fx - rx*Fz, rx*Fy - ry*Fx]
        
        # Constraint force moment (at contact point)
        A[3, 0] = contact_point_vec[1]  # ry * constraint_force_x (z-component of moment)
        A[3, 1] = -contact_point_vec[0]  # -rx * constraint_force_y (z-component of moment)
        
        # Applied force moment
        r_app = app_point_vec - contact_point_vec
        A[3, 2] = r_app[1]  # ry * app_Fx (z-component of moment)
        
        # Ligament moments
        r_lig_a = lig_a_point_vec - contact_point_vec
        r_lig_b = lig_b_point_vec - contact_point_vec
        lig_a_moment = np.cross(r_lig_a, lig_a_force_vec)
        lig_b_moment = np.cross(r_lig_b, lig_b_force_vec)
        
        b[3] = -(lig_a_moment[2] + lig_b_moment[2])  # z-component of ligament moments
        
        # x and y components of moment balance (if needed)
        A[4, 0] = -contact_point_vec[2]  # -rz * constraint_force_x
        A[4, 1] = contact_point_vec[2]   # rz * constraint_force_y
        A[4, 2] = -r_app[2]  # -rz * app_Fx
        b[4] = -(lig_a_moment[0] + lig_b_moment[0])
        
        A[5, 0] = contact_point_vec[1]   # ry * constraint_force_x
        A[5, 1] = -contact_point_vec[0]  # -rx * constraint_force_y
        A[5, 2] = r_app[0]  # rx * app_Fx
        b[5] = -(lig_a_moment[1] + lig_b_moment[1])
        
        # Solve the linear system
        try:
            solution = np.linalg.solve(A, b)
            constraint_force_x, constraint_force_y, app_Fx_solved = solution
        except np.linalg.LinAlgError:
            # Fallback to least squares if system is singular
            solution = np.linalg.lstsq(A, b, rcond=None)[0]
            constraint_force_x, constraint_force_y, app_Fx_solved = solution
        
        # Create solutions dictionary in the same format as before
        solutions = {
            self.constraint_unknowns[0]: constraint_force_x,
            self.constraint_unknowns[1]: constraint_force_y,
            self.force_vec_sym[0]: app_Fx_solved
        }

        if self.log:
            print(f"Spring A elongation: {lig_a_length}")
            print(f"Spring B elongation: {lig_b_length}")
            print(f"Solved forces: constraint_x={constraint_force_x:.3f}, constraint_y={constraint_force_y:.3f}, app_Fx={app_Fx_solved:.3f}")

        # Substitute solutions back into forces
        self.constraint_force.substitute_solutions(solutions)
        self.applied_force.substitute_solutions(solutions)

        # Add computed values to solutions dictionary
        solutions['lig_springA_length'] = lig_a_length
        solutions['lig_springB_length'] = lig_b_length

        # Create force objects for ligament forces
        lig_a_force_point = Point(lig_a_force_vec, self.tibia_frame)
        lig_b_force_point = Point(lig_b_force_vec, self.tibia_frame)
        solutions['lig_springA_force'] = Force("LigSpringA", lig_a_force_point, self.lig_bottom_pointA)
        solutions['lig_springB_force'] = Force("LigSpringB", lig_b_force_point, self.lig_bottom_pointB)

        
        solutions['applied_force'] = self.applied_force
        solutions['constraint_force'] = self.constraint_force
        
        return solutions

    def _needs_linear_system_rebuild(self):
        """Check if the linear system needs to be rebuilt based on what has changed."""
        if self._linear_system_cache is None:
            return True
        
        # Check if theta has changed (affects geometry)
        if self._cached_theta != self.data['theta']:
            return True
            
        # Check if ligament parameters have changed
        current_params = (self.lig_function_left.get_params().copy(), self.lig_function_right.get_params().copy())
        if self._cached_ligament_params != current_params:
            return True
            
        return False

    def _build_linear_system(self):
        """Build the linear system structure (this is called once and cached)."""
        # This method is called when we need to rebuild the linear system
        # For now, we'll just mark it as built - the actual system is built in solve()
        self._linear_system_cache = True

    def calculate_moment_arm(self, force_point, force_vector, pivot_point):
        """
        Calculate the moment arm of a force around a pivot point.
        """
        
        return (force_point - pivot_point).cross(force_vector).norm() / force_vector.norm()
    
    def set_theta(self, theta):
        """
        Fast path: Update only the joint angle without full model rebuild.
        
        Args:
            theta: New joint angle in radians
        """
        self.data['theta'] = theta
        self.knee_joint.set_theta(theta)
        
        # Update cached theta
        self._cached_theta = theta
        
        # Invalidate linear system cache since geometry changed
        self._linear_system_cache = None

    def set_ligament_functions(self, lig_left, lig_right):
        """
        Fast path: Update ligament functions without full model rebuild.
        
        Args:
            lig_left: New left ligament function (LCL)
            lig_right: New right ligament function (MCL)
        """
        self.lig_function_left = lig_left
        self.lig_function_right = lig_right
        
        # Update ligament springs by updating their functions directly
        # This avoids recreating the spring objects
        self.lig_springA.function = lig_left
        self.lig_springB.function = lig_right
        
        # Update cached parameters
        self._cached_ligament_params = (lig_left.get_params().copy(), lig_right.get_params().copy())
        
        # Invalidate linear system cache since ligament parameters changed
        self._linear_system_cache = None
    
    def update_ligament_parameters(self, mcl_params, lcl_params):
        """
        Update the ligament function parameters without rebuilding entire model.
        
        Args:
            mcl_params: numpy array of MCL parameters [k, alpha, l_0, f_ref]
            lcl_params: numpy array of LCL parameters [k, alpha, l_0, f_ref]
        """
        from src.ligament_models.blankevoort import BlankevoortFunction
        
        # Create new ligament function instances with updated parameters
        lig_left = BlankevoortFunction(lcl_params)
        lig_right = BlankevoortFunction(mcl_params)
        
        # Use the fast path method
        self.set_ligament_functions(lig_left, lig_right)
    
    def get_ligament_elongations(self):
        """
        Extract ligament elongations directly without full solve.
        
        Returns:
            tuple: (ligament_a_elongation, ligament_b_elongation)
        """
        return (self.lig_springA.get_spring_length(), self.lig_springB.get_spring_length())
            
    def plot_model(self, show_forces=False):
        vis = Visualiser2D(self.world_frame)
        vis.add_point(self.knee_point, label="Knee origin")
        vis.add_point(self.hip_point, label="Hip")

        vis.add_point(self.lig_top_pointA, label="Lateral")
        vis.add_point(self.lig_top_pointB, label="Medial")
        vis.add_point(self.lig_bottom_pointA, label=" ")
        vis.add_point(self.lig_bottom_pointB, label=" ")
        vis.add_point(self.application_point, label=" ")

        vis.add_circle(self.joint_ball_A, self.ball_radius_1)
        vis.add_circle(self.joint_ball_B, self.ball_radius_2)

        vis.add_line(self.knee_point, self.hip_point, label="Femur")
        vis.add_line(self.knee_point, self.lig_top_pointA)
        vis.add_line(self.knee_point, self.lig_top_pointB)
        vis.add_line(self.lig_top_pointA, self.lig_bottom_pointA, label="Lig", color="red")
        vis.add_line(self.lig_top_pointB, self.lig_bottom_pointB, label="Lig", color="red")

        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointA_flat, label="TibalPlataeuA")
        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointB_flat, label="TibalPlataeuB")
        vis.add_line(self.lig_bottom_pointA, self.lig_bottom_pointA_flat, label="BoneStructure")
        vis.add_line(self.lig_bottom_pointB, self.lig_bottom_pointB_flat, label="BoneStructure")
        vis.add_line(self.lig_on_tib_vis, self.application_point, label="Tibia")

        if show_forces:
            vis.add_force(self.constraint_force, associated_body_name="Constraint", label=self.constraint_force.name)
            vis.add_force(self.lig_springA.get_force_on_point2(), associated_body_name="LigamentA", label="LigSpringForceA")
            vis.add_force(self.lig_springB.get_force_on_point2(), associated_body_name="LigamentB", label="LigSpringForceB")
            vis.add_force(self.applied_force, associated_body_name="Applied", label="AppliedForce")


        vis.render(show_values=False, equal_aspect=True)


