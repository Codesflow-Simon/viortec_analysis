from statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from ligament_reconstructor.ligament_optimiser import reconstruct_ligament
from ligament_reconstructor.utils import get_params_from_config
from ligament_models.blankevoort import BlankevoortFunction


if __name__ == "__main__":    
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    lig_left = BlankevoortFunction(config['blankevoort_mcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    model = KneeModel(config['mechanics'], lig_left, lig_right, log=False)
    
    solutions = model.solve()
    # model.plot_model()
    # plt.show()

    thetas = np.linspace(-np.radians(1), np.radians(1), 15)

    length_estimates_a = []
    force_estimates_a = []

    length_estimates_b = []
    force_estimates_b = []

    for theta in thetas:
        mechanics = config['mechanics'].copy()
        mechanics['theta'] = theta
        model = KneeModel(mechanics, lig_left, lig_right, log=False)
        solutions = model.solve()

        length_estimates_a.append(solutions['lig_springA_length'])
        force_estimates_a.append(solutions['lig_springA_force'].get_force().norm())

        # length_estimates_b.append(solutions['lig_springB_length'])
        # force_estimates_b.append(solutions['lig_springB_force'].get_force().norm())
        # if theta == thetas[-1]:
        #     model.plot_model()
        #     plt.show()

    length = np.array(length_estimates_a + length_estimates_b, dtype=np.float64)
    force = np.array(force_estimates_a + force_estimates_b, dtype=np.float64)

    force = force + np.random.normal(0, config['data']['y_noise'], len(force))

    reference_point = force[5]

    print(f"Reference force: {reference_point}")
    relative_force = force - reference_point # We only measure relative changes in force

    def sort_data(length, force, relative_force):
        sort_idx = np.argsort(length)
        length = length[sort_idx]
        force = force[sort_idx]
        relative_force = relative_force[sort_idx]
        return length, force, relative_force
    length, force, relative_force = sort_data(length, force, relative_force)

    result_obj = reconstruct_ligament(length, relative_force)
    function = result_obj['function']
    print(result_obj['params'])

    gt_params = config['blankevoort_mcl']
    gt_params['f_ref'] = reference_point

    from ligament_models.constraints import ConstraintManager
    from ligament_reconstructor.slide_search import slide_search
    constraint_manager = ConstraintManager()

    l_0_bounds = constraint_manager.get_constraints_list()[2]
    result = slide_search(function, length, relative_force, l_0_bounds)
    print(result)

    plt.figure()
    plt.scatter(length, relative_force, c='r', label='Data', s=8, alpha=0.5)

    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), np.max(length), 100)
    plt.plot(x_data, function(x_data), c='b', label='Model')

    print(gt_params)
    function.set_params(np.array(list(gt_params.values())))

    plt.plot(x_data, function(x_data), c='g', label='Ground Truth', linestyle='--')
    plt.legend()
    plt.xlabel('Ligament Length')
    plt.ylabel('Ligament Relative Force')

    plt.show()
