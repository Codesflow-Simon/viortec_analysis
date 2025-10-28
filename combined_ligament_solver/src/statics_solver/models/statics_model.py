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
        self.equations_assembled = False
        self.ligament_forces_built = False

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

    def get_moment_arms(self):
        spring_a_direction = self.lig_bottom_pointA - self.lig_top_pointA
        spring_b_direction = self.lig_bottom_pointB - self.lig_top_pointB
        self.moment_arms = {"MCL moment arm": self.calculate_moment_arm(spring_a_direction.convert_to_frame(self.world_frame), spring_a_direction.convert_to_frame(self.world_frame), self.constraint_force.application_point),
               "LCL moment arm": self.calculate_moment_arm(spring_b_direction.convert_to_frame(self.world_frame), spring_b_direction.convert_to_frame(self.world_frame), self.constraint_force.application_point),
               "Application Moment arm": self.calculate_moment_arm(self.application_point.convert_to_frame(self.world_frame), Point([1, 0, 0], self.tibia_frame).convert_to_frame(self.world_frame), self.constraint_force.application_point.convert_to_frame(self.world_frame))}
        return self.moment_arms

    def solve_applied(self, thetas, mcl_params, lcl_params):
        def blankevoort_func(length, params):
            k, alpha, l_0, f_ref = params
            # Calculate strain
            strain = (length - l_0) / l_0
            # Calculate force using Blankevoort equation
            if strain > 0:
                force = (k / (4 * alpha)) * (np.exp(alpha * strain) - 1)
            else:
                force = 0
            return force

        for theta in thetas:
            mcl_length = self.lig_top_pointA.distance(self.lig_bottom_pointA.convert_to_frame(self.world_frame))
            lcl_length = self.lig_top_pointB.distance(self.lig_bottom_pointB.convert_to_frame(self.world_frame))
            mcl_forces.append(blankevoort_func(mcl_length, mcl_params))
            lcl_forces.append(blankevoort_func(lcl_length, lcl_params))


                blankevoort_vec = np.vectorize(lambda l: blankevoort_func(l, mcl_params))
    
        mcl_forces = blankevoort_vec()
        blankevoort_vec = np.vectorize(lambda l: blankevoort_func(l, lcl_params))
        lcl_forces = blankevoort_vec(thetas)

        mcl_moments = mcl_forces * self.moment_arms["MCL moment arm"]
        lcl_moments = lcl_forces * self.moment_arms["LCL moment arm"]
        application_moments = (mcl_forces + lcl_forces) * self.moment_arms["Application Moment arm"]

        applied_forces = mcl_moments + lcl_moments + application_moments

        return applied_forces

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

