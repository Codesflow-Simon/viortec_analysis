from src.statics_model import KneeModel
from src.ligament_optimiser import least_squares_optimize_complete_model
from src.mcmc_sampler import CompleteMCMCSampler, NUTSSampler
from src.visualization import visualize_ligament_curves, visualize_theta_force_curves, visualize_parameter_marginals
import numpy as np
import yaml
import matplotlib.pyplot as plt


def analyse_data(config, data, constraints_config, ref_strain_lcl=None, ref_strain_mcl=None):
    # Run least squares optimization first
    print("=" * 50)
    print("RUNNING LEAST SQUARES OPTIMIZATION")
    print("=" * 50)
    
    ls_result = least_squares_optimize_complete_model(
        data['thetas'], data['measured_forces'], 
        config['mechanics'],
        constraints_config, sigma_noise=float(config['data']['sigma_noise'])
    )
    
    print(f"Least squares RMSE: {ls_result['rmse']:.2f}")
    print(f"Least squares MAE: {ls_result['mae']:.2f}")
    print(f"MCL parameters: {ls_result['mcl_params']}")
    print(f"LCL parameters: {ls_result['lcl_params']}")
    
    # Run MCMC sampling
    print("\n" + "=" * 50)
    print("RUNNING SAMPLING")
    print("=" * 50)
    
    # NUTS sampling
    from src.mcmc_sampler import CompleteMCMCSampler
    mcmc_sampler = CompleteMCMCSampler(config['mechanics'], constraints_config)
    # # Previous emcee MCMC sampling (commented out)
    sampler = CompleteMCMCSampler(config['mechanics'], constraints_config)
    
    # Create gt_params dict for ground truth parameters
    gt_params_dict = {
        'blankevoort_mcl': config['blankevoort_mcl'],
        'blankevoort_lcl': config['blankevoort_lcl']
    }
    
    cov_matrix, std_params, samples, acceptance_rate = sampler.sample(
        data['thetas'], data['measured_forces'], sigma_noise=float(config['data']['sigma_noise']), ls_result=ls_result, gt_params=gt_params_dict
    )


    # nuts_sampler = NUTSSampler(config['mechanics'], constraints_config, n_samples=1000, n_tune=500, random_seed=42)
    # cov_matrix, std_params, samples, acceptance_rate = nuts_sampler.sample(
    #     data['thetas'], data['measured_forces'], sigma_noise=float(config['data']['sigma_noise'])
    # )
    
    print(f"Samples completed with {len(samples)} samples")
    print(f"Acceptance rate: {acceptance_rate:.3f}")
    
    # Compare results
    print("\n" + "=" * 50)
    print("COMPARISON: LEAST SQUARES vs samples")
    print("=" * 50)
    print(f"Least squares RMSE: {ls_result['rmse']:.2f}")
    print(f"Least squares MAE:  {ls_result['mae']:.2f}")
    
    # Calculate NUTS mean parameters for comparison
    nuts_mcl_params = np.mean(samples[:, :3], axis=0)
    nuts_lcl_params = np.mean(samples[:, 3:], axis=0)
    print(f"\nParameter comparison:")
    print(f"MCL - LS: {ls_result['mcl_params']}")
    print(f"MCL - NUTS mean: {nuts_mcl_params}")
    print(f"LCL - LS: {ls_result['lcl_params']}")
    print(f"LCL - NUTS mean: {nuts_lcl_params}")
    
    # Visualize results
    visualize_ligament_curves(config, samples, data, ls_result, ref_strain_lcl=ref_strain_lcl, ref_strain_mcl=ref_strain_mcl)
    visualize_theta_force_curves(config, samples, data, ls_result, ref_strain_lcl=ref_strain_lcl, ref_strain_mcl=ref_strain_mcl)
    visualize_parameter_marginals(samples, ls_result, constraints_config, ref_strain_lcl=ref_strain_lcl, ref_strain_mcl=ref_strain_mcl)
    # plt.show()
    
    return {
        'cov_matrix': cov_matrix,
        'std_params': std_params,
        'samples': samples,
        'acceptance_rate': acceptance_rate,
        'least_squares_result': ls_result
    }

def collect_data(config):
    # Initialize data collection lists
    data_lists = {
        'thetas': [],
        'applied_forces': [],
        'mcl_lengths': [],
        'lcl_lengths': [],
        'mcl_forces': [],
        'lcl_forces': [],
        'moment_arms': [],
        'mcl_moments': [],
        'lcl_moments': [],
        'application_moments': []
    }
    moment_limit = 12_000  # In N(mm)
    

    mcl_params = config['blankevoort_mcl']
    lcl_params = config['blankevoort_lcl']
    
    # Collect data in both directions from theta=0
    print("\nCollecting ligament data over theta range...")
    
    # Positive direction (increasing theta)
    theta = 0 * np.pi/180
    increment = 0.05 * np.pi/180
    knee_model = KneeModel(config['mechanics'], log=False)
    knee_model.build_geometry()
    # knee_model.plot_model(show_forces=True)
    # plt.show()

    while True:
        result = knee_model.solve_applied([theta], mcl_params, lcl_params)
        data_lists['applied_forces'].append(result['applied_forces'])
        data_lists['thetas'].append(theta)
        data_lists['mcl_lengths'].append(result['mcl_lengths'])
        data_lists['lcl_lengths'].append(result['lcl_lengths'])
        data_lists['mcl_forces'].append(result['mcl_forces'])
        data_lists['lcl_forces'].append(result['lcl_forces'])
        data_lists['moment_arms'].append(result['moment_arms'])
        data_lists['mcl_moments'].append(result['mcl_moments'])
        data_lists['lcl_moments'].append(result['lcl_moments'])
        data_lists['application_moments'].append(result['application_moments'])
        theta += increment
        # print(f"Theta: {np.degrees(theta)}, Force: {result['applied_forces']}, Moment: {result['application_moments']}")
        if abs(result['application_moments']) > moment_limit:
            break

    theta = -increment
    while True:
        result = knee_model.solve_applied([theta], mcl_params, lcl_params)
        data_lists['applied_forces'].append(result['applied_forces'])
        data_lists['thetas'].append(theta)
        data_lists['mcl_lengths'].append(result['mcl_lengths'])
        data_lists['lcl_lengths'].append(result['lcl_lengths'])
        data_lists['mcl_forces'].append(result['mcl_forces'])
        data_lists['lcl_forces'].append(result['lcl_forces'])
        data_lists['moment_arms'].append(result['moment_arms'])
        data_lists['mcl_moments'].append(result['mcl_moments'])
        data_lists['lcl_moments'].append(result['lcl_moments'])
        data_lists['application_moments'].append(result['application_moments'])
        # print(f"Theta: {np.degrees(theta)}, Force: {result['applied_forces']}, Moment: {result['application_moments']}")
        theta -= increment
        if abs(result['application_moments']) > moment_limit:
            break

    y_noise = float(config['data']['y_noise'])
    # Ensure applied_forces is a numpy array and reshape to match noise shape
    applied_forces = np.array(data_lists['applied_forces']).reshape(-1)
    noise = np.random.normal(0, y_noise, size=applied_forces.shape)
    measured_forces = applied_forces + noise
    data_lists['measured_forces'] = measured_forces

    return data_lists

if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Load constraints configuration
    with open('constraints.yaml', 'r') as f:
        constraints_config = yaml.safe_load(f)

    results = []

    reference_strains_lcl = [0.03, 0.06, 0.09]
    reference_strains_mcl = [0.03, 0.06, 0.09]
    
    for ref_strain_lcl, ref_strain_mcl in zip(reference_strains_lcl, reference_strains_mcl):  
            config['blankevoort_lcl']['l_0'] = config['mechanics']['right_length']/(ref_strain_lcl + 1)
            config['blankevoort_mcl']['l_0'] = config['mechanics']['left_length']/(ref_strain_mcl + 1)
            print(f"LCL length: {config['blankevoort_lcl']['l_0']}, MCL length: {config['blankevoort_mcl']['l_0']}")
            data = collect_data(config)
            result = analyse_data(config, data, constraints_config, ref_strain_lcl=ref_strain_lcl, ref_strain_mcl=ref_strain_mcl)
            results.append({
                'lcl_strain': ref_strain_lcl,
                'mcl_strain': ref_strain_mcl,
                'result': result
            })
