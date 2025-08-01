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

class StaticsSolver:
    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self.input_data = config['input_data']
        
    def 

def main(data=dict(), plot=True):
    # Some constants
    femur_perp = data['femur_perp']
    femur_length = data['femur_length']
    tibia_perp = data['tibia_perp']
    tibia_para = data['tibia_para']
    application_length = data['application_length']
    theta_val = -data['theta']
    ligament_slack_length = data['ligament_slack_length']
    app_Fx = data['app_Fx']

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
    lig_springA = BlankevoortSpring(lig_top_pointA, lig_bottom_pointA, "LigSpringA", 0.06, 2000, ligament_slack_length)
    lig_springB = BlankevoortSpring(lig_top_pointB, lig_bottom_pointB, "LigSpringB", 0.06, 2000, ligament_slack_length)

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


    # Applied force: Unknown force, no torques transferred
    force_vec_sym = [app_Fx,0 ,0 ]
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
    unknowns = list(dict.fromkeys(unknowns))  # Preserves order while removing duplicates

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

    results = {
        'theta': theta_val,
        'lig_springA_length': lig_springA.get_spring_length(),
        'lig_springB_length': lig_springB.get_spring_length(),
        'app_Fx': app_Fx,
        'constraint_force': constraint_force.get_force_in_frame(world_frame),
        'applied_force': applied_force.get_force_in_frame(world_frame),
        'solutions': solutions
    }
    return results

def main_sweep_theta(data_dict):
    # Sweep theta values and calculate lengths
    theta_range = np.linspace(-np.pi/4, np.pi/4, 20)
    output_dict = {}

    for theta in theta_range:
        data_dict['theta'] = theta
        result_dict = main(data_dict, plot=False)
        for k,v in result_dict.items():
            if k not in output_dict:
                output_dict[k] = []
            output_dict[k].append(v)

    return output_dict

if __name__ == "__main__":
    # theta_sym = Symbol('theta')
    theta_sym = np.radians(10)

    app_Fx_sym = Symbol('app_Fx')
    # app_Fx_sym = 1000

    
    data_dict = main_sweep_theta(input_data)
    # print(data_dict)

    print("\nResults for Two-Ball Joint at 10 degrees:")
    input_data['theta'] = np.radians(10)
    main(input_data, plot=True)
    plt.show()


