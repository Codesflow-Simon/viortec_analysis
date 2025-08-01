import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "statics_solver"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "trilinear_solver"))

from sympy import Symbol
from statics_solver.statics_solver import main_sweep_theta

import yaml



if __name__ == "__main__":
    theta_sym = None
    app_Fx_sym = Symbol('app_Fx')
    
    with open('./statics_solver/mechanics_config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    input_data = config['input_data']

    data_sweep=main_sweep_theta(input_data)
    applied_force = [x.force.coordinates[0] for x in data_sweep['applied_force']]
    theta = data_sweep['theta']
    A_ligament_length = data_sweep['lig_springA_length']
    B_ligament_length = data_sweep['lig_springB_length']

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(A_ligament_length, applied_force, alpha=0.6, label='Ligament A')
    plt.scatter(B_ligament_length, applied_force, alpha=0.6, label='Ligament B')
    # Add theta labels to each point
    for i, (x1, x2, y, t) in enumerate(zip(A_ligament_length, B_ligament_length, applied_force, theta)):
        plt.annotate(f'θ={t:.2f}', (x1, y), xytext=(5, 5), textcoords='offset points')
        plt.annotate(f'θ={t:.2f}', (x2, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Ligament Length (m)')
    plt.legend()
    plt.ylabel('Applied Force (N)')
    plt.title('Applied Force vs Ligament Length')
    plt.grid(True)
    plt.show()