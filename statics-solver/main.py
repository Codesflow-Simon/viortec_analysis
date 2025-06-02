from mappings import *
from reference_frame import *
from rigid_body import *
import sympy
from sympy import Symbol, lambdify
import numpy as np
import matplotlib.pyplot as plt


def main(data=dict()):
    world_frame = ReferenceFrame("WorldFrame")
    world_frame.set_as_ground_frame()

    femur_perp_val = data.get('femur_perp', 0.05)
    top_point = Point([femur_perp_val, 0, 0], world_frame)

    body_frame = ReferenceFrame("BodyFrame")

    rotation = RotationalMapping.from_euler_angles([0, 0, data['theta']])
    translation = TranslationMapping(sympy.Matrix([0, 0, 0]))

    body_in_world = RigidBodyMapping(rotation, translation)
    body_frame.add_parent(world_frame, body_in_world, body_to_world=True)

    tibia_perp_val = data.get('tibia_perp', 0.04)
    tibia_para_val = data.get('tibia_para', 0.03)
    bottom_point = Point([tibia_perp_val, tibia_para_val, 0], body_frame)
    bottom_point_in_world = bottom_point.convert_to_frame(world_frame)

    distance_vector = bottom_point_in_world - top_point
    distance_expression = distance_vector.norm()

    return distance_expression

if __name__ == "__main__":
    theta_sym = Symbol('theta')

    data_symbolic = {
        'femur_perp': 0.05,
        'tibia_perp': 0.04,
        'tibia_para': 0.03,
        'theta': theta_sym
    }

    symbolic_distance = main(data_symbolic)
    print(f"Symbolic distance expression: {symbolic_distance}")

    numerical_distance_func = lambdify((theta_sym,), symbolic_distance, 'numpy')

    theta_num_values = np.linspace(-np.pi, np.pi, 100)
    
    distances_numeric = numerical_distance_func(theta_num_values)

    plt.figure()
    plt.plot(np.degrees(theta_num_values), distances_numeric)
    plt.xlabel('Theta (deg)')
    plt.ylabel('Distance (symbolic evaluation)')
    plt.title('Distance vs Theta (Symbolic)')
    plt.grid(True)
    plt.show()