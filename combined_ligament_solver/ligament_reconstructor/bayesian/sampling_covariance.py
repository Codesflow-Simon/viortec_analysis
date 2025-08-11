import numpy as np
from scipy.optimize import minimize
from ..modelling.constraints import ConstraintManager
import emcee


def log_likelihood(params, x_data, y_data, func, sigma_noise=100):
    """
    Log-likelihood function for MCMC sampling.
    
    Args:
        params: Parameter vector
        x_data: Input data points
        y_data: Target data points
        func: Function to evaluate
        sigma_noise: Noise standard deviation
        
    Returns:
        log_likelihood: Log-likelihood value
    """
    from copy import deepcopy
    func = deepcopy(func)


    func.set_params(params)
    y_pred = np.array([func(x) for x in x_data])
    
    # Compute residuals
    residuals = y_data - y_pred
    
    # Log-likelihood (assuming Gaussian noise)
    log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
    
    return log_like

def log_prior(params, constraint_manager=None):
    """
    Log-prior function for MCMC sampling.
    
    Args:
        params: Parameter vector
        constraint_manager: Optional ConstraintManager for physical constraints
        
    Returns:
        log_prior: Log-prior value
    """
    # Basic physical constraints: parameters should be finite
    for param in params:
        if not np.isfinite(param):
            return -np.inf
    
    # Apply physical constraints if constraint manager is provided
    if constraint_manager is not None:
        try:
            constraints = constraint_manager.get_constraints()
            for constraint in constraints:
                if constraint['fun'](params) <= 0:
                    return -np.inf
        except Exception:
            # If constraint evaluation fails, reject the sample
            return -np.inf
    
    # Log of uninformative prior (constant)
    return 0.0

def log_probability(params, x_data, y_data, func, sigma_noise, constraint_manager=None):
    """
    Log-probability function (prior + likelihood) for MCMC sampling.
    
    Args:
        params: Parameter vector
        x_data: Input data points
        y_data: Target data points
        func: Function to evaluate
        sigma_noise: Noise standard deviation
        constraint_manager: Optional ConstraintManager for physical constraints
        
    Returns:
        log_prob: Log-probability value
    """
    lp = log_prior(params, constraint_manager)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(params, x_data, y_data, func, sigma_noise)
    return lp + ll

def estimate_noise_level(x_data, y_data, map_params, func):
    """
    Estimate noise level from residuals at MAP estimate.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        map_params: MAP estimate
        func: Function to evaluate
        
    Returns:
        sigma_noise: Estimated noise standard deviation
    """
    y_pred = np.array([func(x, map_params) for x in x_data])
    residuals = y_data - y_pred
    return np.std(residuals)

def initial_walkers(map_params, n_walkers, n_params, constraint_manager):
    """
    Initialize walkers around MAP estimate with some scatter.
    """
    # Parameters are ordered [alpha, k, l_0, f_ref]
    base_positions = np.array(list(map_params.values()), dtype=float)
    
    constraints_list = constraint_manager.get_constraints_list()

    initial_positions = np.zeros((n_walkers, len(base_positions)))

    for i, constraint in enumerate(constraints_list):
        lower, upper = constraint
        initial_positions[:, i] = np.random.uniform(lower, upper, n_walkers)
    
    # Ensure all parameters stay positive (constraint satisfaction)
    initial_positions = np.maximum(initial_positions, 1e-6)
    
    return initial_positions

def compute_mcmc_covariance(map_params, x_data, y_data, func,
                           n_walkers, n_steps, n_burnin, sigma_noise, random_state=None, constraint_manager=None):
    """
    Compute covariance matrix using MCMC sampling with emcee.
    
    Args:
        map_params: MAP estimate (optimal parameters)
        x_data: Input data points
        y_data: Target data points
        func: Function to evaluate
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        n_burnin: Number of burn-in steps to discard
        random_state: Random state for reproducibility
        sigma_noise: Noise standard deviation
        constraint_manager: Optional ConstraintManager for physical constraints
        
    Returns:
        cov_matrix: MCMC covariance matrix
        std_params: Standard deviations of parameters
        mcmc_samples: All MCMC samples
        acceptance_fraction: Acceptance fraction of the sampler
    """
    
    # print(f"Computing MCMC covariance with {n_walkers} walkers, {n_steps} steps...")
    
    n_params = len(map_params)
    
    # Set up the MCMC sampler with better proposal strategy
    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_probability,
        args=(x_data, y_data, func, sigma_noise, constraint_manager),
        moves=emcee.moves.StretchMove(a=2.0)  # More conservative stretch move
    )
    
    # Initialize walkers around MAP estimate with some scatter
    initial_positions = initial_walkers(map_params, n_walkers, n_params, constraint_manager)
    
    if random_state is not None:
        np.random.seed(random_state)
    
    # Run MCMC
    # print("Running MCMC...")
    sampler.run_mcmc(initial_positions, n_steps, progress=True)
    
    # Get samples after burn-in
    samples = sampler.get_chain(discard=n_burnin, flat=True)
    
    # Compute statistics
    mcmc_means = np.mean(samples, axis=0)
    cov_matrix = np.cov(samples, rowvar=False)
    std_params = np.sqrt(np.diag(cov_matrix))
    
    return cov_matrix, std_params, samples, np.mean(sampler.acceptance_fraction)

def check_constraint_violations(samples, constraint_manager):
    """
    Check how many MCMC samples violate the physical constraints.
    
    Args:
        samples: MCMC samples array
        constraint_manager: ConstraintManager instance
        
    Returns:
        violation_rate: Fraction of samples that violate constraints
        violation_details: Details about which constraints are violated
    """
    if constraint_manager is None:
        return 0.0, {}
    
    constraints = constraint_manager.get_constraints()
    n_samples = len(samples)
    violations = np.zeros(n_samples, dtype=bool)
    constraint_violations = {i: 0 for i in range(len(constraints))}
    
    for i, sample in enumerate(samples):
        for j, constraint in enumerate(constraints):
            try:
                if constraint['fun'](sample) <= 0:
                    violations[i] = True
                    constraint_violations[j] += 1
            except Exception:
                violations[i] = True
                constraint_violations[j] += 1
    
    violation_rate = np.mean(violations)
    violation_details = {f"constraint_{j}": count/n_samples for j, count in constraint_violations.items()}
    
    return violation_rate, violation_details

def compute_sampling_covariance(map_params, x_data, y_data, func, 
                               n_samples=1000, method='mcmc', sigma_noise=1e-3, constraint_manager=None):
    """
    Main function to compute covariance matrix using MCMC sampling.
    
    Args:
        map_params: MAP estimate (optimal parameters)
        x_data: Input data points
        y_data: Target data points
        func: Function to evaluate
        n_samples: Number of MCMC steps (total samples = n_walkers * n_steps)
        method: Sampling method ('mcmc')
        sigma_noise: Noise standard deviation
        constraint_manager: Optional ConstraintManager for physical constraints
        
    Returns:
        cov_matrix: Covariance matrix
        std_params: Standard deviations of parameters
        mcmc_samples: MCMC parameter samples
        acceptance_rate: MCMC acceptance rate
    """
    # print(f"Computing covariance using {method} method...")
    
    if method == 'mcmc':
        # Calculate number of walkers and steps based on total samples
        n_walkers = 16
        n_steps = 200
        n_burnin = 100
        
        cov_matrix, std_params, mcmc_samples, acceptance_rate = compute_mcmc_covariance(
            map_params, x_data, y_data, func,
            n_walkers=n_walkers, n_steps=n_steps, n_burnin=n_burnin,
            sigma_noise=sigma_noise,
            constraint_manager=constraint_manager
        )
        return cov_matrix, std_params, mcmc_samples, acceptance_rate
    else:
        raise ValueError(f"Unknown method: {method}")