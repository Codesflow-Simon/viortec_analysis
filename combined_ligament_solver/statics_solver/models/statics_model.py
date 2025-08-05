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
from ..src.joint_models import TwoBallJoint, PivotJoint
from .base import AbstractModel

class KneeModel(AbstractModel):
    def __init__(self, data, log=True):
        self.log = log
        self.data = data

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
        theta_val = -self.data['theta']
        app_Fx = self.data['app_Fx']

        ligament_slack_length = self.data['ligament_slack_length']
        ligament_transition_point = self.data['ligament_transition_point']
        ligament_stiffness = self.data['ligament_stiffness']

        # Frames
        self.world_frame = ReferenceFrame("WorldFrame")
        self.world_frame.set_as_ground_frame()
        self.tibia_frame = ReferenceFrame("TibiaFrame")


        self.ball_distance = tibia_perp/2
        self.ball_radius = tibia_para/2
        self.knee_joint = TwoBallJoint(self.tibia_frame, self.world_frame, distance=self.ball_distance, radius=self.ball_radius)
        self.knee_joint.set_theta(theta_val)

        self.tibia_frame.add_parent(self.world_frame, self.knee_joint)

        # Rigid bodies
        self.femur_body = RigidBody("Femur", self.world_frame)
        self.tibia_body = RigidBody("Tibia", self.tibia_frame)

        # Points of interest
        self.hip_point = Point([0, femur_length, 0], self.world_frame)
        self.knee_point = Point([0, 0, 0], self.world_frame)
        self.joint_ball_A = Point([self.ball_distance, -self.ball_radius, 0], self.world_frame)
        self.joint_ball_B = Point([-self.ball_distance, -self.ball_radius, 0], self.world_frame)
        self.lig_top_pointA = Point([femur_perp, 0, 0], self.world_frame)
        self.lig_top_pointB = Point([-femur_perp, 0, 0], self.world_frame)
        
        self.lig_on_tib_vis = Point([0, -tibia_para, 0], self.tibia_frame)
        self.lig_bottom_pointA = Point([tibia_perp, -tibia_para, 0], self.tibia_frame)
        self.lig_bottom_pointB = Point([-tibia_perp, -tibia_para, 0], self.tibia_frame)
        self.application_point = Point([0, -application_length, 0], self.tibia_frame)

        # Springs
        self.lig_springA = BlankevoortSpring(self.lig_top_pointA, self.lig_bottom_pointA, "LigSpringA",
             ligament_transition_point, ligament_stiffness, ligament_slack_length)
        self.lig_springB = BlankevoortSpring(self.lig_top_pointB, self.lig_bottom_pointB, "LigSpringB",
             ligament_transition_point, ligament_stiffness, ligament_slack_length)

        print(f"LigSpringA: {self.lig_springA.get_force_on_point1()}")
        print(f"LigSpringB: {self.lig_springB.get_force_on_point1()}")

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
            print(f"\nSolutions: {solutions}")
            print(f"Spring A elongation: {self.lig_springA.get_spring_length()}")
            print(f"Spring B elongation: {self.lig_springB.get_spring_length()}")

        # Substitute solutions back into forces
        self.constraint_force.substitute_solutions(solutions)
        self.applied_force.substitute_solutions(solutions)
        return solutions
    
    def plot_model(self):
        vis = Visualiser2D(self.world_frame)
        vis.add_point(self.knee_point)
        vis.add_point(self.hip_point, label="Hip")

        vis.add_point(self.lig_top_pointA, label="LigTopA")
        vis.add_point(self.lig_top_pointB, label="LigTopB")
        vis.add_point(self.lig_bottom_pointA, label="LigBottomA")
        vis.add_point(self.lig_bottom_pointB, label="LigBottomB")
        vis.add_point(self.application_point, label="Application")

        vis.add_circle(self.joint_ball_A, self.ball_radius)
        vis.add_circle(self.joint_ball_B, self.ball_radius)

        vis.add_line(self.knee_point, self.hip_point, label="Femur")
        vis.add_line(self.knee_point, self.lig_top_pointA, label="FemurBottom")
        vis.add_line(self.knee_point, self.lig_top_pointB, label="FemurBottom")
        vis.add_line(self.lig_top_pointA, self.lig_bottom_pointA, label="Lig", color="red")
        vis.add_line(self.lig_top_pointB, self.lig_bottom_pointB, label="Lig", color="red")
        vis.add_line(self.lig_on_tib_vis, self.application_point, label="Tibia")
        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointA, label="TibalPlataeuA")
        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointB, label="TibalPlataeuB")

        
        vis.add_force(self.constraint_force, label=self.constraint_force.name)
        vis.add_force(self.lig_springA.get_force_on_point2(), label="LigSpringForceA")
        vis.add_force(self.lig_springB.get_force_on_point2(), label="LigSpringForceB")
        vis.add_force(self.applied_force, label="AppliedForce")


        vis.render(show_values=False, equal_aspect=True)

if __name__ == "__main__":
    with open('./statics_solver/mechanics_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data = config['input_data']
    for key, value in data.items():
        if isinstance(value, str) and '_sym' in value:
            data[key] = Symbol(key.replace('_sym', ''))

    model = KneeModel(data, log=False)
    model.build_model()
    model.solve()
    model.plot_model()
    plt.show()


