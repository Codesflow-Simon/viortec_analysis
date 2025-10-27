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
        self.build_model()

    def update_data(self, data):
        self.data = data
        self.build_model()

    def build_model(self):
        # Some constants
        femur_perp = self.data['femur_perp']
        femur_length = self.data['femur_length']
        tibia_perp = self.data['tibia_perp']
        tibia_para = self.data['tibia_para']
        application_length = self.data['application_length']
        theta_val = self.data['theta']
        app_Fx = self.data['app_Fx']
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

        # Springs
        self.lig_springA = BlankevoortSpring.from_ligament_function(self.lig_top_pointA, self.lig_bottom_pointA, "LigSpringA", self.lig_function_left)
        self.lig_springB = BlankevoortSpring.from_ligament_function(self.lig_top_pointB, self.lig_bottom_pointB, "LigSpringB", self.lig_function_right)

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

        # Applied force: Unknown force, no torques transferred
        self.force_vec_sym = [app_Fx,0 ,0 ]
        self.force_vector = Point(self.force_vec_sym, self.tibia_frame)
        self.applied_force = Force("AppliedForce", self.force_vector, self.application_point)
        self.tibia_body.add_external_force(self.applied_force)

    def solve(self):
        force_expression, torque_expression = self.tibia_body.get_net_forces()
        force_expression.simplify(trig=True)
        torque_expression.simplify(trig=True)
        if self.log:
            print(f"Force expression: {force_expression}")
            print(f"Torque expression: {torque_expression}")

        # Solving
        unknown_from_system = [x for x in self.constraint_unknowns + self.force_vec_sym]
        unknown_inputs = [v for k,v in self.data.items() if not isinstance(v, (int, float))]
        unknowns = unknown_from_system + unknown_inputs
        unknowns = [x for x in unknowns if not isinstance(x, (int, float))]
        unknowns = list(dict.fromkeys(unknowns))  # Preserves order while removing duplicates

        if self.log:
            print(f"Unknowns: {unknowns}")

        # solve forces equal to zero
        equations_to_solve = list(force_expression) + list(torque_expression)
        
        #Print each equation separately
        if self.log:
            print(f"\nNumber of equations: {len(equations_to_solve)}")
            print(f"Number of unknowns: {len(unknowns)}")
            print("\nEquations to solve:")
            for i, eq in enumerate(equations_to_solve):
                print(f"Equation {i+1}: {eq} = 0")
        
        solutions = sympy.solve(equations_to_solve, unknowns)

        if self.log:
            print(f"Spring A elongation: {self.lig_springA.get_spring_length()}")
            print(f"Spring B elongation: {self.lig_springB.get_spring_length()}")

        # Substitute solutions back into forces
        self.constraint_force.substitute_solutions(solutions)
        self.applied_force.substitute_solutions(solutions)

        # Add computed values to solutions dictionary
        solutions['lig_springA_length'] = self.lig_springA.get_spring_length()
        solutions['lig_springB_length'] = self.lig_springB.get_spring_length()

        solutions['lig_springA_force'] = self.lig_springA.get_force_on_point1()
        solutions['lig_springB_force'] = self.lig_springB.get_force_on_point1()

        contact_point = self.knee_joint.get_contact_point(theta=self.data['theta'])
        left_moment_arm = self.calculate_moment_arm(self.lig_bottom_pointA, self.applied_force.get_force(), contact_point.convert_to_frame(self.tibia_frame))
        right_moment_arm = self.calculate_moment_arm(self.lig_bottom_pointB, self.applied_force.get_force(), contact_point.convert_to_frame(self.tibia_frame))
        
        print(f"Left moment arm: {left_moment_arm}")
        print(f"Right moment arm: {right_moment_arm}")
        print(f"Theta: {self.data['theta']}")
        if self.data['theta'] > 0:
            solutions['estimated_lig_springA_force'] = self.applied_force.get_moment().norm() / (left_moment_arm + right_moment_arm)
            solutions['estimated_lig_springB_force'] = -self.applied_force.get_moment().norm() / (left_moment_arm + right_moment_arm)
        else:
            solutions['estimated_lig_springA_force'] = -self.applied_force.get_moment().norm() / (right_moment_arm + left_moment_arm)
            solutions['estimated_lig_springB_force'] = self.applied_force.get_moment().norm() / (right_moment_arm + left_moment_arm)

        solutions['applied_force'] = self.applied_force
        solutions['constraint_force'] = self.constraint_force

        # print(f"Solutions: {solutions}")
        
        return solutions

    def calculate_moment_arm(self, force_point, force_vector, pivot_point):
        """
        Calculate the moment arm of a force around a pivot point.
        """
        
        return (force_point - pivot_point).cross(force_vector).norm() / force_vector.norm()
            
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


