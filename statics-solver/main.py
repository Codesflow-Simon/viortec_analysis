from mappings import *
from reference_frame import *
from rigid_body import *
from springs import *
import sympy
from sympy import Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt
from visualiser import *

def main(data=dict()):
    # Some constants
    femur_perp = data['femur_perp']
    femur_length = data['femur_length']
    tibia_perp = data['tibia_perp']
    tibia_para = data['tibia_para']
    application_length = data['application_length']
    theta_val = data['theta']
    ligament_slack_length = data['ligament_slack_length']

    # Frames
    world_frame = ReferenceFrame("WorldFrame")
    world_frame.set_as_ground_frame()
    tibia_frame = ReferenceFrame("TibiaFrame")

    actual_translation_mapping = TranslationMapping(sympy.Matrix([0,0,0]))

    rotation_mapping = RotationalMapping.from_euler_angles(sympy.Matrix([0, 0, theta_val]))
    tibia_to_world_mapping = RigidBodyMapping(rotation_mapping, actual_translation_mapping)

    tibia_frame.add_parent(world_frame, tibia_to_world_mapping, body_to_world=True)

    # Rigid bodies
    femur_body = RigidBody("Femur", world_frame)
    tibia_body = RigidBody("Tibia", tibia_frame)

    # Points of interest
    hip_point = Point([0, femur_length, 0], world_frame)
    knee_point = Point([0, 0, 0], world_frame)
    lig_top_point = Point([femur_perp, 0, 0], world_frame)
    lig_on_tib_vis = Point([0, -tibia_para, 0], tibia_frame)
    lig_bottom_point = Point([tibia_perp, -tibia_para, 0], tibia_frame)
    application_point = Point([0, -application_length, 0], tibia_frame)

    # Springs
    lig_spring = Spring(lig_top_point, lig_bottom_point, "LigSpring", 50, ligament_slack_length)
    # Register spring forces on the bodies
    femur_body.add_external_force(lig_spring.get_force_on_point1())
    tibia_body.add_external_force(lig_spring.get_force_on_point2())

    # Forces

    # Knee joint forces: Unkown joint force, no torques transferred
    knee_vec_sym = [Symbol('Knee_Fx'), Symbol('Knee_Fy'), 0]
    force_vector = Point(knee_vec_sym, world_frame)
    knee_force = Force("KneeForce", force_vector, knee_point)
    tibia_body.add_force_pair(knee_force, femur_body)

    # Applied force: Unknown force, no torques transferred
    force_vec_sym = [Symbol('App_Fx'),0 ,0 ]
    force_vector = Point(force_vec_sym, tibia_frame)
    applied_force = Force("AppliedForce", force_vector, application_point)
    tibia_body.add_external_force(applied_force)

    force_expression, torque_expression = tibia_body.get_net_forces()
    force_expression.simplify()
    torque_expression.simplify()

    print(f"Force expression: {force_expression}")
    print(f"Torque expression: {torque_expression}")

    # Solving

    unknowns = [x for x in knee_vec_sym + force_vec_sym if not isinstance(x, (int, float))]
    print(f"Unknowns: {unknowns}")

    # solve forces equal to zero
    equations_to_solve = list(force_expression) + list(torque_expression)
    solutions = sympy.solve(equations_to_solve, unknowns)
    print(f"Solutions: {solutions}")

    # Substitute solutions back into forces
    knee_force.force.coordinates = knee_force.force.coordinates.subs(solutions)
    applied_force.force.coordinates = applied_force.force.coordinates.subs(solutions)

    print(f"Knee force: {knee_force.force.coordinates}")
    print(f"Applied force: {applied_force.force.coordinates}")
    print(f"Lig spring force: {lig_spring.get_force_on_point2().force.coordinates}")

    # Visualise the system

    vis = Visualiser2D(world_frame)
    vis.add_point(knee_point, label="Knee")
    vis.add_point(hip_point, label="Hip")

    vis.add_point(lig_top_point, label="LigTop")
    vis.add_point(lig_bottom_point, label="LigBottom")
    vis.add_point(application_point, label="Application")

    vis.add_line(knee_point, hip_point, label="Femur")
    vis.add_line(knee_point, lig_top_point, label="FemurBottom")
    vis.add_line(lig_top_point, lig_bottom_point, label="Lig", color="red")
    vis.add_line(knee_point, application_point, label="Tibia")
    vis.add_line(lig_on_tib_vis, lig_bottom_point, label="Lig")

    try:
        vis.add_force(knee_force, label="KneeForce")
        vis.add_force(applied_force, label="AppliedForce")
        vis.add_force(lig_spring.get_force_on_point2(), label="LigSpringForce")
    except Exception as e:
        print(f"Error adding force, probably non-")

    vis.render()

if __name__ == "__main__":
    theta_sym = Symbol('theta')

    data_symbolic = {
        'femur_length': 0.5, # Distance from hip to knee
        'femur_perp': 0.05, # Perpendicular distance from hip to ligament
        'tibia_perp': 0.05, # Perpendicular (to the tibia) distance from knee to ligament
        'tibia_para': 0.05, # Distance (down the tibia) from knee to ligament
        'application_length': 0.10, # Distance(down the tibia)
        'theta': 0,
        'ligament_slack_length': 0.01
    }

    main(data_symbolic)


