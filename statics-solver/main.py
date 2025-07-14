from mappings import *
from reference_frame import *
from rigid_body import *
from springs import *
import sympy
from sympy import Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt
from visualiser import *
from joint_models import TwoBallJoint, PivotJoint

def main_balls(data=dict(), plot=True):
    # Some constants
    femur_perp = data['femur_perp']
    femur_length = data['femur_length']
    tibia_perp = data['tibia_perp']
    tibia_para = data['tibia_para']
    application_length = data['application_length']
    theta_val = -data['theta']
    ligament_slack_length = data['ligament_slack_length']

    # Frames
    world_frame = ReferenceFrame("WorldFrame")
    world_frame.set_as_ground_frame()
    tibia_frame = ReferenceFrame("TibiaFrame")

    ball_distance = tibia_perp/2
    ball_radius = tibia_para/2
    knee_joint = TwoBallJoint(tibia_frame, world_frame, distance=ball_distance, radius=ball_radius)
    knee_joint.set_theta(theta_val)

    tibia_frame.add_parent(world_frame, knee_joint)

    # Rigid bodies
    femur_body = RigidBody("Femur", world_frame)
    tibia_body = RigidBody("Tibia", tibia_frame)

    # Points of interest
    hip_point = Point([0, femur_length, 0], world_frame)
    knee_point = Point([0, 0, 0], world_frame)
    joint_ball_A = Point([ball_distance, -ball_radius, 0], world_frame)
    joint_ball_B = Point([-ball_distance, -ball_radius, 0], world_frame)
    lig_top_pointA = Point([femur_perp, 0, 0], world_frame)
    lig_top_pointB = Point([-femur_perp, 0, 0], world_frame)
    
    lig_on_tib_vis = Point([0, -tibia_para, 0], tibia_frame)
    lig_bottom_pointA = Point([tibia_perp, -tibia_para, 0], tibia_frame)
    lig_bottom_pointB = Point([-tibia_perp, -tibia_para, 0], tibia_frame)
    application_point = Point([0, -application_length, 0], tibia_frame)

    # Springs
    lig_springA = TriLinearSpring(lig_top_pointA, lig_bottom_pointA, "LigSpring", 200_000, 500_000, 0.01, ligament_slack_length)
    lig_springB = TriLinearSpring(lig_top_pointB, lig_bottom_pointB, "LigSpring", 200_000, 500_000, 0.01, ligament_slack_length)

    # Constraint forces
    constraint_force, constraint_unknowns = knee_joint.get_constraint_force()
    if plot:
        print(f"Constraint force: {constraint_force}")

    tibia_body.add_force_pair(constraint_force, femur_body)


    # Register spring forces on the bodies
    femur_body.add_external_force(lig_springA.get_force_on_point1())
    tibia_body.add_external_force(lig_springA.get_force_on_point2())

    femur_body.add_external_force(lig_springB.get_force_on_point1())
    tibia_body.add_external_force(lig_springB.get_force_on_point2())

    # Forces

    # # Knee joint forces: Unkown joint force, no torques transferred
    # knee_vec_sym = [Symbol('Knee_Fx'), Symbol('Knee_Fy'), 0]
    # force_vector = Point(knee_vec_sym, world_frame)
    # knee_force = Force("KneeForce", force_vector, knee_point)
    # tibia_body.add_force_pair(knee_force, femur_body)

    # Applied force: Unknown force, no torques transferred
    force_vec_sym = [Symbol('App_Fx'),0 ,0 ]
    force_vector = Point(force_vec_sym, tibia_frame)
    applied_force = Force("AppliedForce", force_vector, application_point)
    tibia_body.add_external_force(applied_force)

    force_expression, torque_expression = tibia_body.get_net_forces()
    force_expression.simplify(trig=True)
    torque_expression.simplify(trig=True)
    if plot:
        print(f"Force expression: {force_expression}")
        print(f"Torque expression: {torque_expression}")

    # Solving
    unknown_from_system = [x for x in constraint_unknowns + force_vec_sym]
    unknown_inputs = [v for k,v in data.items() if not isinstance(v, (int, float))]
    unknowns = unknown_from_system + unknown_inputs
    unknowns = [x for x in unknowns if not isinstance(x, (int, float))]
    if plot:
        print(f"Unknowns: {unknowns}")

    # solve forces equal to zero
    equations_to_solve = list(force_expression) + list(torque_expression)
    
    #Print each equation separately
    if plot:
        print(f"\nNumber of equations: {len(equations_to_solve)}")
        print(f"Number of unknowns: {len(unknowns)}")
        print("\nEquations to solve:")
        for i, eq in enumerate(equations_to_solve):
            print(f"Equation {i+1}: {eq} = 0")
    
    solutions = sympy.solve(equations_to_solve, unknowns)
    
    if plot:
        print(f"\nSolutions: {solutions}")
        print(f"Spring A elongation: {lig_springA.get_spring_length()}")
        print(f"Spring B elongation: {lig_springB.get_spring_length()}")

    # Substitute solutions back into forces
    constraint_force.substitute_solutions(solutions)
    applied_force.substitute_solutions(solutions)

    # Visualise the system

    if plot:
        vis = Visualiser2D(world_frame)
        vis.add_point(knee_point)
        vis.add_point(hip_point, label="Hip")

        vis.add_point(lig_top_pointA, label="LigTopA")
        vis.add_point(lig_top_pointB, label="LigTopB")
        vis.add_point(lig_bottom_pointA, label="LigBottomA")
        vis.add_point(lig_bottom_pointB, label="LigBottomB")
        vis.add_point(application_point, label="Application")

        vis.add_circle(joint_ball_A, ball_radius)
        vis.add_circle(joint_ball_B, ball_radius)

        vis.add_line(knee_point, hip_point, label="Femur")
        vis.add_line(knee_point, lig_top_pointA, label="FemurBottom")
        vis.add_line(knee_point, lig_top_pointB, label="FemurBottom")
        vis.add_line(lig_top_pointA, lig_bottom_pointA, label="Lig", color="red")
        vis.add_line(lig_top_pointB, lig_bottom_pointB, label="Lig", color="red")
        vis.add_line(lig_on_tib_vis, application_point, label="Tibia")
        vis.add_line(lig_on_tib_vis, lig_bottom_pointA, label="TibalPlataeuA")
        vis.add_line(lig_on_tib_vis, lig_bottom_pointB, label="TibalPlataeuB")

        
        vis.add_force(constraint_force, label=constraint_force.name)
        vis.add_force(lig_springA.get_force_on_point2(), label="LigSpringForceA")
        vis.add_force(lig_springB.get_force_on_point2(), label="LigSpringForceB")
        vis.add_force(applied_force, label="AppliedForce")


        vis.render(show_values=False, equal_aspect=True)

    return solutions, lig_springA.get_spring_length(), lig_springB.get_spring_length()


def main_pivot(data=dict(), plot=True):
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

    ball_distance = tibia_perp/2
    ball_radius = tibia_para/2
    knee_joint = PivotJoint(tibia_frame, world_frame)
    knee_joint.set_theta(theta_val)

    tibia_frame.add_parent(world_frame, knee_joint)

    # Rigid bodies
    femur_body = RigidBody("Femur", world_frame)
    tibia_body = RigidBody("Tibia", tibia_frame)

    # Points of interest
    hip_point = Point([0, femur_length, 0], world_frame)
    knee_point = Point([0, 0, 0], world_frame)
    joint_ball_A = Point([ball_distance, -ball_radius, 0], world_frame)
    joint_ball_B = Point([-ball_distance, -ball_radius, 0], world_frame)
    lig_top_pointA = Point([femur_perp, 0, 0], world_frame)
    lig_top_pointB = Point([-femur_perp, 0, 0], world_frame)
    
    lig_on_tib_vis = Point([0, -tibia_para, 0], tibia_frame)
    lig_bottom_pointA = Point([tibia_perp, -tibia_para, 0], tibia_frame)
    lig_bottom_pointB = Point([-tibia_perp, -tibia_para, 0], tibia_frame)
    application_point = Point([0, -application_length, 0], tibia_frame)

    # Springs
    lig_springA = TriLinearSpring(lig_top_pointA, lig_bottom_pointA, "LigSpring", 200_000, 500_000, 0.01, ligament_slack_length)
    lig_springB = TriLinearSpring(lig_top_pointB, lig_bottom_pointB, "LigSpring", 200_000, 500_000, 0.01, ligament_slack_length)

    # Constraint forces
    constraint_force, constraint_unknowns = knee_joint.get_constraint_force()
    if plot:
        print(f"Constraint force: {constraint_force}")

    tibia_body.add_force_pair(constraint_force, femur_body)


    # Register spring forces on the bodies
    femur_body.add_external_force(lig_springA.get_force_on_point1())
    tibia_body.add_external_force(lig_springA.get_force_on_point2())

    femur_body.add_external_force(lig_springB.get_force_on_point1())
    tibia_body.add_external_force(lig_springB.get_force_on_point2())

    # Forces

    # # Knee joint forces: Unkown joint force, no torques transferred
    # knee_vec_sym = [Symbol('Knee_Fx'), Symbol('Knee_Fy'), 0]
    # force_vector = Point(knee_vec_sym, world_frame)
    # knee_force = Force("KneeForce", force_vector, knee_point)
    # tibia_body.add_force_pair(knee_force, femur_body)

    # Applied force: Unknown force, no torques transferred
    force_vec_sym = [Symbol('App_Fx'),0 ,0 ]
    force_vector = Point(force_vec_sym, tibia_frame)
    applied_force = Force("AppliedForce", force_vector, application_point)
    tibia_body.add_external_force(applied_force)

    force_expression, torque_expression = tibia_body.get_net_forces()
    force_expression.simplify(trig=True)
    torque_expression.simplify(trig=True)

    if plot:
        print(f"Force expression: {force_expression}")
        print(f"Torque expression: {torque_expression}")

    # Solving
    unknown_from_system = [x for x in constraint_unknowns + force_vec_sym]
    unknown_inputs = [v for k,v in data.items() if not isinstance(v, (int, float))]
    unknowns = unknown_from_system + unknown_inputs
    unknowns = [x for x in unknowns if not isinstance(x, (int, float))]
    if plot:
        print(f"Unknowns: {unknowns}")

    # solve forces equal to zero
    equations_to_solve = list(force_expression) + list(torque_expression)
    
    #Print each equation separately
    if plot:
        print(f"\nNumber of equations: {len(equations_to_solve)}")
        print(f"Number of unknowns: {len(unknowns)}")
        print("\nEquations to solve:")
        for i, eq in enumerate(equations_to_solve):
            print(f"Equation {i+1}: {eq} = 0")
    
    solutions = sympy.solve(equations_to_solve, unknowns)
    
    if plot:
        print(f"\nSolutions: {solutions}")
        print(f"Spring A elongation: {lig_springA.get_spring_length()}")
        print(f"Spring B elongation: {lig_springB.get_spring_length()}")

    # Substitute solutions back into forces
    constraint_force.substitute_solutions(solutions)
    applied_force.substitute_solutions(solutions)

    # Visualise the system

    if plot:
        vis = Visualiser2D(world_frame)
        vis.add_point(knee_point)
        vis.add_point(hip_point, label="Hip")

        vis.add_point(lig_top_pointA, label="LigTopA")
        vis.add_point(lig_top_pointB, label="LigTopB")
        vis.add_point(lig_bottom_pointA, label="LigBottomA")
        vis.add_point(lig_bottom_pointB, label="LigBottomB")
        vis.add_point(application_point, label="Application")

        vis.add_circle(joint_ball_A, ball_radius)
        vis.add_circle(joint_ball_B, ball_radius)

        vis.add_line(knee_point, hip_point, label="Femur")
        vis.add_line(knee_point, lig_top_pointA, label="FemurBottom")
        vis.add_line(knee_point, lig_top_pointB, label="FemurBottom")
        vis.add_line(lig_top_pointA, lig_bottom_pointA, label="Lig", color="red")
        vis.add_line(lig_top_pointB, lig_bottom_pointB, label="Lig", color="red")
        vis.add_line(knee_point, application_point, label="Tibia")
        vis.add_line(lig_on_tib_vis, lig_bottom_pointA, label="TibalPlataeuA")
        vis.add_line(lig_on_tib_vis, lig_bottom_pointB, label="TibalPlataeuB")

        
        vis.add_force(constraint_force, label=constraint_force.name)
        vis.add_force(lig_springA.get_force_on_point2(), label="LigSpringForceA")
        vis.add_force(lig_springB.get_force_on_point2(), label="LigSpringForceB")
        vis.add_force(applied_force, label="AppliedForce")


        vis.render(show_values=False, equal_aspect=True)

    return solutions, lig_springA.get_spring_length(), lig_springB.get_spring_length()

if __name__ == "__main__":
    # theta_sym = Symbol('theta')
    theta_sym = np.radians(40)

    data_symbolic = {
        'femur_length': 0.5, # Distance from hip to knee
        'femur_perp': 0.1, # Perpendicular distance from hip to ligament
        'tibia_perp': 0.1, # Perpendicular (to the tibia) distance from knee to ligament
        'tibia_para': 0.05, # Distance (down the tibia) from knee to ligament
        'application_length': 0.20, # Distance(down the tibia)
        'theta': theta_sym,
        'ligament_slack_length': 0.07
    }
    # Sweep theta values and calculate lengths
    theta_range = np.linspace(-np.pi/2, np.pi/2, 50)
    lengths_balls = []
    lengths_pivot = []

    for theta in theta_range:
        data_symbolic['theta'] = theta
        _, length_ball, _ = main_balls(data_symbolic, plot=False)
        _, length_piv, _ = main_pivot(data_symbolic, plot=False)
        lengths_balls.append(length_ball)
        lengths_pivot.append(length_piv)

    # Plot results
    plt.figure()
    plt.plot(np.degrees(theta_range), lengths_balls, label='Two-Ball Joint')
    plt.plot(np.degrees(theta_range), lengths_pivot, label='Pivot Joint')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Ligament Length (m)')
    plt.title('Ligament Length vs Joint Angle')
    plt.legend()
    plt.grid(True)
    # plt.show()

    # Run both joint types at 10 degrees with plotting
    data_10deg = {
        'femur_length': 0.5,
        'femur_perp': 0.1,
        'tibia_perp': 0.1, 
        'tibia_para': 0.05,
        'application_length': 0.20,
        'theta': np.radians(10),
        'ligament_slack_length': 0.07
    }

    print("\nResults for Two-Ball Joint at 10 degrees:")
    main_balls(data_10deg, plot=True)

    print("\nResults for Pivot Joint at 10 degrees:")  
    main_pivot(data_10deg, plot=True)
    plt.show()


