from statics_solver.models.statics_model import KneeModel
import yaml
from matplotlib import pyplot as plt
import numpy as np
from ligament_reconstructor.ligament_reconstructor import LigamentReconstructor
from ligament_reconstructor.bayesian.sampling_covariance import compute_sampling_covariance
from ligament_reconstructor.utils import get_params_from_config


if __name__ == "__main__":
    with open('./statics_solver/mechanics_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model = KneeModel(config['input_data'], log=False)

    thetas = np.linspace(-np.radians(0), np.radians(2), 20)

    length_estimates_a = []
    force_estimates_a = []

    length_estimates_b = []
    force_estimates_b = []

    for theta in thetas:
        config['input_data']['theta'] = theta
        model = KneeModel(config['input_data'], log=False)
        model.build_model()
        solutions = model.solve()
        length_estimates_a.append(solutions['lig_springA_length'])
        force_estimates_a.append(solutions['lig_springA_force'].get_force().norm())

        length_estimates_b.append(solutions['lig_springB_length'])
        force_estimates_b.append(solutions['lig_springB_force'].get_force().norm())

    # plt.figure()
    # plt.scatter(length_estimates_a, force_estimates_a, c='r', s=8, alpha=0.5)
    # plt.scatter(length_estimates_b, force_estimates_b, c='b', s=8, alpha=0.5)
    # plt.xlabel('Ligament Length')
    # plt.ylabel('Ligament Force')

    # plt.figure(figsize=(10,4))
    
    # # Plot length vs theta
    # plt.subplot(121)
    # plt.scatter(np.degrees(thetas), length_estimates_a, c='r', s=8, alpha=0.5, label='Ligament A')
    # plt.scatter(np.degrees(thetas), length_estimates_b, c='b', s=8, alpha=0.5, label='Ligament B')
    # plt.xlabel('Theta (degrees)')
    # plt.ylabel('Ligament Length (m)')
    # plt.legend()
    
    # # Plot force vs theta
    # plt.subplot(122)
    # plt.scatter(np.degrees(thetas), force_estimates_a, c='r', s=8, alpha=0.5, label='Ligament A')
    # plt.scatter(np.degrees(thetas), force_estimates_b, c='b', s=8, alpha=0.5, label='Ligament B')
    # plt.xlabel('Theta (degrees)')
    # plt.ylabel('Ligament Force (N)')
    # plt.legend()
    
    # plt.tight_layout()

    length = np.array(length_estimates_a + length_estimates_b, dtype=np.float64)
    force = np.array(force_estimates_a + force_estimates_b, dtype=np.float64)
    reference_point = force[0]

    print(f"Reference force: {reference_point}")
    relative_force = force - reference_point # We only measure relative changes in force

    sort_idx = np.argsort(length)
    length = length[sort_idx]
    force = force[sort_idx]
    relative_force = relative_force[sort_idx]

    with open('ligament_reconstructor/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    params = get_params_from_config(config, config['mode'])

    reconstructor = LigamentReconstructor()
    reconstructor.setup_model('blankevoort', params)
    result_obj = reconstructor.reconstruct(length, relative_force, params)
    function = result_obj['function']
    print(result_obj['params'])

    from ligament_reconstructor.bayesian.sampling_covariance import compute_sampling_covariance, check_constraint_violations
    from ligament_reconstructor.bayesian.mcmc_plots import plot_all_mcmc_diagnostics
    
    # Get the constraint manager from the reconstructor
    constraint_manager = reconstructor.constraint_manager

    data_cov_matrix, std, samples, acceptance_rate = compute_sampling_covariance(
        result_obj['params'], length, relative_force, function, n_samples=10000, sigma_noise=1e-3, constraint_manager=constraint_manager)
    print("Parameter standard deviations:", np.sqrt(np.diag(data_cov_matrix)))
    print(f"MCMC acceptance rate: {acceptance_rate:.3f}")
    
    # Get parameter names for plotting
    param_names = list(result_obj['params'].keys())
    
    # Generate comprehensive MCMC diagnostic plots (filtering invalid samples)
    plot_all_mcmc_diagnostics(samples, param_names, acceptance_rate, save_plots=True, constraint_manager=constraint_manager)

    with open('statics_solver/mechanics_config.yaml', 'r') as file:
        gt_config = yaml.safe_load(file)

    gt_params = {
        'alpha': float(gt_config['input_data']['ligament_transition_point']),
        'k': float(gt_config['input_data']['ligament_stiffness']),
        'l_0': float(gt_config['input_data']['ligament_slack_length']),
        'f_ref': float(reference_point)
    }

    plt.figure()
    plt.scatter(length, relative_force, c='r', label='Data', s=8, alpha=0.5)

    x_data = np.linspace(min(result_obj['params']['l_0']*0.9, np.min(length)), np.max(length), 100)
    plt.plot(x_data, function(x_data), c='b', label='Model')

    # Plot a few random MCMC samples
    n_samples_to_plot = 50
    random_indices = np.random.choice(len(samples), n_samples_to_plot)
    for idx in random_indices:
        sample_params = samples[idx]
        function.set_params(sample_params)
        plt.plot(x_data, function(x_data), c='gray', alpha=0.1)

    print(gt_params)
    function.set_params(np.array(list(gt_params.values())))

    plt.plot(x_data, function(x_data), c='g', label='Ground Truth', linestyle='--')
    plt.legend()
    plt.xlabel('Ligament Length')
    plt.ylabel('Ligament Relative Force')

    plt.show()
