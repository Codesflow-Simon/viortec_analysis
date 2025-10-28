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
    def __init__(self, data, log=True):
        self.log = log
        self.data = data
        self.geometry_built = False
        self.ligament_forces_built = False
        self.equations_assembled = False
        self.solutions = None

    def build_geometry(self):
        """Build the geometric structure (frames, points, joint) based on theta and config."""
        # Some constants
        femur_perp = self.data['femur_perp']
        femur_length = self.data['femur_length']
        tibia_perp = self.data['tibia_perp']
        tibia_para = self.data['tibia_para']
        application_length = self.data['application_length']
        # theta_val = self.data['theta']
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
        self.knee_joint.set_theta(None)

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
        self.geometry_built = True

    def build_ligament_forces(self, lig_function_left: BlankevoortFunction, lig_function_right: BlankevoortFunction):
        """Build ligament springs and applied forces."""

        if not self.geometry_built:
            raise ValueError("Geometry must be built before building ligament forces")

        self.lig_function_left = lig_function_left
        self.lig_function_right = lig_function_right

        # Springs
        self.lig_springA = BlankevoortSpring.from_ligament_function(self.lig_top_pointA, self.lig_bottom_pointA, "LigSpringA", lig_function_left)
        self.lig_springB = BlankevoortSpring.from_ligament_function(self.lig_top_pointB, self.lig_bottom_pointB, "LigSpringB", lig_function_right)

        self.ligament_forces_built = True

    def assemble_equations(self, theta: float):
        """Assemble the force and moment balance equations."""

        if not self.ligament_forces_built:
            raise ValueError("Ligament forces must be built before assembling equations")

        self.knee_joint.set_theta(theta)

        # Applied force: Unknown force, no torques transferred
        self.app_Fx = self.data['app_Fx']
        app_Fx = self.app_Fx
        self.force_vec_sym = [app_Fx,0 ,0 ]
        self.force_vector = Point(self.force_vec_sym, self.tibia_frame)
        self.applied_force = Force("AppliedForce", self.force_vector, self.application_point)

        # Constraint forces
        self.constraint_force, self.constraint_unknowns = self.knee_joint.get_constraint_force(define_frame=self.tibia_frame)
        if self.log:
            print(f"Constraint force: {self.constraint_force}")

        self.tibia_body.clear()
        self.tibia_body.add_force_pair(self.constraint_force, self.femur_body)

        # Register spring forces on the bodies
        lig_bottom_pointA = self.lig_bottom_pointA.convert_to_frame(self.world_frame)
        lig_bottom_pointB = self.lig_bottom_pointB.convert_to_frame(self.world_frame)

        self.lig_springA.set_points(self.lig_top_pointA, lig_bottom_pointA)
        self.lig_springB.set_points(self.lig_top_pointB, lig_bottom_pointB)

        self.femur_body.add_external_force(self.lig_springA.get_force_on_point1())
        self.tibia_body.add_external_force(self.lig_springA.get_force_on_point2())

        self.femur_body.add_external_force(self.lig_springB.get_force_on_point1())
        self.tibia_body.add_external_force(self.lig_springB.get_force_on_point2())

        self.tibia_body.add_external_force(self.applied_force)

        self.equations_assembled = True

    def solve(self):
        if not self.equations_assembled:
            raise ValueError("Equations must be assembled before solving")

        """Solve the linear system using direct numpy linear algebra instead of sympy.solve()."""
        net_force_coords, net_moment_coords = self.tibia_body.get_net_forces()

        non_zero_force = net_force_coords[0:2, 0]
        non_zero_moment = net_moment_coords[2, 0]
        equations = sympy.Matrix([non_zero_force[0], non_zero_force[1], non_zero_moment])
        solutions = sympy.solve(equations, [self.constraint_unknowns[0], self.constraint_unknowns[1], self.app_Fx])
        
        if solutions is None or len(solutions) == 0:
            # Print debug information when no solutions found
            print("\nDEBUG: No solutions found for system of equations")
            print("Current model state:")
            print(f"- Theta: {self.knee_joint.theta}")
            print(f"- Equations:")
            print(equations)
            print("\nForces in system:")
            print(f"- Net forces: {net_force_coords}")
            print(f"- Net moment: {net_moment_coords}")
            print(f"- Constraint unknowns: {self.constraint_unknowns}")
            print(f"- Applied force unknown: {self.app_Fx}")
            raise ValueError("No solutions found")
        TwoBallConstraintForce_x = solutions[self.constraint_unknowns[0]]
        TwoBallConstraintForce_y = solutions[self.constraint_unknowns[1]]
        app_Fx_solved = list(solutions.values())[2]

        if self.log:
            print(f"Solved forces: constraint_x={TwoBallConstraintForce_x:.3f}, constraint_y={TwoBallConstraintForce_y:.3f}, app_Fx={app_Fx_solved:.3f}")

        # Substitute solutions back into forces
        self.constraint_force.substitute_solutions(solutions)
        self.applied_force.substitute_solutions(solutions)

        solutions.update({
            'lig_springA_length': self.lig_springA.get_spring_length(),
            'lig_springB_length': self.lig_springB.get_spring_length(),
            'lig_springA_force': self.lig_springA.get_force_on_point2(),
            'lig_springB_force': self.lig_springB.get_force_on_point2(),
            'applied_force': self.applied_force,
            'constraint_force': self.constraint_force
        })

        self.equations_assembled = False
        return solutions

    def reset(self):
        self.geometry_built = False
        self.ligament_forces_built = False
        self.equations_assembled = False
        self.solutions = None
        self.knee_joint.set_theta(None)
        self.lig_function_left = None
        self.lig_function_right = None
        self.lig_springA = None
        self.lig_springB = None
        self.applied_force = None
        self.constraint_force = None
        self.constraint_unknowns = None
        self.net_force_coords = None
        self.net_moment_coords = None
        self.non_zero_force = None
        self.non_zero_moment = None
        self.equations = None
        self.solutions = None

    def calculate_moment_arm(self, force_point, force_vector, pivot_point):
        """
        Calculate the moment arm of a force around a pivot point.
        """
        
        return (force_point - pivot_point).cross(force_vector).norm() / force_vector.norm()
    
   
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

    def calculate_thetas(self, thetas):
        """
        Calculate comprehensive results for a list of theta angles.
        
        This method encapsulates the common pattern of:
        1. Looping through theta values
        2. Assembling equations at each theta
        3. Solving the system
        4. Extracting all relevant data
        
        Args:
            thetas: List or array of knee angles in radians
            
        Returns:
            dict: Dictionary with lists of all calculated values:
                - 'applied_forces': List of applied force magnitudes
                - 'applied_moments': List of applied moment magnitudes
                - 'lig_springA_lengths': List of LCL spring lengths
                - 'lig_springB_lengths': List of MCL spring lengths
                - 'lig_springA_forces': List of LCL force magnitudes
                - 'lig_springB_forces': List of MCL force magnitudes
                - 'constraint_forces': List of constraint force magnitudes
        """
        results = {
            'applied_forces': [],
            'applied_moments': [],
            'lig_springA_lengths': [],
            'lig_springB_lengths': [],
            'lig_springA_forces': [],
            'lig_springB_forces': [],
            'constraint_forces': []
        }
        
        for theta in thetas:
            self.assemble_equations(theta)
            solutions = self.solve()
            
            # Extract all the useful data
            results['applied_forces'].append(float(solutions['applied_force'].get_force().norm()))
            results['applied_moments'].append(float(solutions['applied_force'].get_moment().norm()))
            results['lig_springA_lengths'].append(float(solutions['lig_springA_length']))
            results['lig_springB_lengths'].append(float(solutions['lig_springB_length']))
            results['lig_springA_forces'].append(float(solutions['lig_springA_force'].get_force().norm()))
            results['lig_springB_forces'].append(float(solutions['lig_springB_force'].get_force().norm()))
            results['constraint_forces'].append(float(solutions['constraint_force'].get_force().norm()))
        
        return results


