from statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from ligament_reconstructor.ligament_reconstructor import LigamentReconstructor

from ligament_reconstructor.utils import get_params_from_config
from ligament_models.blankevoort import BlankevoortFunction


if __name__ == "__main__":
    with open('./config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    lig_left = BlankevoortFunction(config['blankevoort_mcl'])
    lig_right = BlankevoortFunction(config['blankevoort_mcl'])
    model = KneeModel(config['mechanics'], lig_left, lig_right, log=False)
    
    solutions = model.solve()
    # model.plot_model()
    # plt.show()

    thetas = np.linspace(-np.radians(0), np.radians(2), 20)

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

        length_estimates_b.append(solutions['lig_springB_length'])
        force_estimates_b.append(solutions['lig_springB_force'].get_force().norm())
        # if theta == thetas[-1]:
        #     model.plot_model()
        #     plt.show()

    length = np.array(length_estimates_a + length_estimates_b, dtype=np.float64)
    force = np.array(force_estimates_a + force_estimates_b, dtype=np.float64)
    reference_point = force[0]

    print(f"Reference force: {reference_point}")
    relative_force = force - reference_point # We only measure relative changes in force

    sort_idx = np.argsort(length)
    length = length[sort_idx]
    force = force[sort_idx]
    relative_force = relative_force[sort_idx]

    gt_params = config['blankevoort_mcl']
    # gt_params['f_ref'] = reference_point
    gt_params['f_ref'] = reference_point/gt_params['k']
    gt_params['l_0'] = gt_params['l_0'] + gt_params['f_ref']

    reconstructor = LigamentReconstructor()
    reconstructor.setup_model('blankevoort', gt_params)
    result_obj = reconstructor.reconstruct(length, relative_force, gt_params)
    function = result_obj['function']
    print(result_obj['params'])


    # Run MCMC to reconstruct the ligament distribution
    from ligament_reconstructor.bayesian import SamplerFactory
    from ligament_reconstructor.bayesian import MCMCSampler

    from ligament_reconstructor.bayesian.mcmc_plots import plot_all_mcmc_diagnostics
    
    # Get the constraint manager from the reconstructor
    constraint_manager = reconstructor.constraint_manager

    sampler = SamplerFactory.create_sampler("mcmc", constraint_manager=constraint_manager)

    data_cov_matrix, std, samples, acceptance_rate = sampler.sample(
        result_obj['params'], length, relative_force, function,
        sigma_noise=1e-1, random_state=42,
        n_samples=1000, loss_hessian=result_obj['loss_hess']
    )

    print("Parameter standard deviations:", std)
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Get parameter names for plotting
    param_names = list(result_obj['params'].keys())
    
    # Generate comprehensive MCMC diagnostic plots (filtering invalid samples)
    plot_all_mcmc_diagnostics(samples, param_names, acceptance_rate, save_plots=True, constraint_manager=constraint_manager)

    plt.figure()
    plt.scatter(length, relative_force, c='r', label='Data', s=8, alpha=0.5)

    x_data = np.linspace(min(gt_params['l_0']*0.9, np.min(length)), np.max(length), 100)
    plt.plot(x_data, function(x_data), c='b', label='Model')

    # Plot a few random MCMC samples
    n_samples_to_plot = 50
    random_indices = np.random.choice(len(samples), n_samples_to_plot)

    for idx in random_indices:
        sample_params = samples[idx]
        function.set_params(sample_params)
        plt.plot(x_data, function(x_data), c='gray', alpha=0.1, label='MCMC Samples' if idx == random_indices[0] else None)

    print(gt_params)
    function.set_params(np.array(list(gt_params.values())))

    plt.plot(x_data, function(x_data), c='g', label='Ground Truth', linestyle='--')
    plt.legend()
    plt.xlabel('Ligament Length')
    plt.ylabel('Ligament Relative Force')

    plt.show()
