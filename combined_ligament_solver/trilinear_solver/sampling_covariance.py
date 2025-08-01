import numpy as np
from scipy.optimize import minimize
from loss import loss, loss_jac, loss_hess
import constraints

try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    print("Warning: emcee not available. Install with: pip install emcee")

def log_likelihood(params, x_data, y_data, funct_tuple, sigma_noise=100):
    """
    Log-likelihood function for MCMC sampling.
    
    Args:
        params: Parameter vector
        x_data: Input data points
        y_data: Target data points
        funct_tuple: Tuple of (function, jacobian, hessian)
        sigma_noise: Noise standard deviation
        
    Returns:
        log_likelihood: Log-likelihood value
    """
    funct = funct_tuple[0]
    
    # Compute model predictions
    y_pred = np.array([funct(x, *params) for x in x_data])
    
    # Compute residuals
    residuals = y_data - y_pred
    
    # Log-likelihood (assuming Gaussian noise)
    log_like = -0.5 * np.sum(residuals**2) / (sigma_noise**2) - len(x_data) * np.log(sigma_noise * np.sqrt(2 * np.pi))
    
    return log_like

def log_prior(params, param_bounds):
    """
    Log-prior function for MCMC sampling.
    
    Args:
        params: Parameter vector
        param_bounds: List of (min, max) bounds for each parameter
        
    Returns:
        log_prior: Log-prior value
    """
    # Uniform prior within bounds
    for i, (param, (min_val, max_val)) in enumerate(zip(params, param_bounds)):
        if param < min_val or param > max_val:
            return -np.inf
    
    # Log of uniform prior (constant)
    return 0.0

def log_probability(params, x_data, y_data, funct_tuple, param_bounds, sigma_noise=100):
    """
    Log-probability function (prior + likelihood) for MCMC sampling.
    
    Args:
        params: Parameter vector
        x_data: Input data points
        y_data: Target data points
        funct_tuple: Tuple of (function, jacobian, hessian)
        param_bounds: List of (min, max) bounds for each parameter
        sigma_noise: Noise standard deviation
        
    Returns:
        log_prob: Log-probability value
    """
    lp = log_prior(params, param_bounds)
    if not np.isfinite(lp):
        return -np.inf
    
    ll = log_likelihood(params, x_data, y_data, funct_tuple, sigma_noise)
    return lp + ll

def estimate_noise_level(x_data, y_data, map_params, funct_tuple):
    """
    Estimate noise level from residuals at MAP estimate.
    
    Args:
        x_data: Input data points
        y_data: Target data points
        map_params: MAP estimate
        funct_tuple: Tuple of (function, jacobian, hessian)
        
    Returns:
        sigma_noise: Estimated noise standard deviation
    """
    funct = funct_tuple[0]
    y_pred = np.array([funct(x, *map_params) for x in x_data])
    residuals = y_data - y_pred
    return np.std(residuals)

def get_parameter_bounds(map_params, mode='trilinear'):
    """
    Get parameter bounds for MCMC sampling.
    
    Args:
        map_params: MAP estimate
        mode: Model mode ('trilinear' or 'blankevoort')
        
    Returns:
        param_bounds: List of (min, max) bounds for each parameter
    """
    bound_level = 0
    if mode == 'trilinear':
        # k_1, k_2, k_3, x_1, x_2
        # Use wide bounds around MAP estimate
        bounds = []
        for i, param in enumerate(map_params):
            if i < 3:  # k parameters
                min_val = max(0, param * bound_level)  # At least 10% of MAP, but non-negative
                max_val = param * 10.0  # Up to 10x MAP
            else:  # x parameters
                min_val = max(0.01, param * bound_level)  # At least 0.01, but 10% of MAP
                max_val = min(1.0, param * 10.0)  # At most 1.0, but 10x MAP
            bounds.append((min_val, max_val))
    else:  # blankevoort
        # e_t, k_1
        bounds = []
        for i, param in enumerate(map_params):
            if i == 0:  # e_t
                min_val = max(0.01, param * bound_level)
                max_val = min(1.0, param * 10.0)
            else:  # k_1
                min_val = max(0, param * bound_level)
                max_val = param * 10.0
            bounds.append((min_val, max_val))
    
    return bounds

def compute_mcmc_covariance(map_params, x_data, y_data, funct_tuple, mode='trilinear',
                           n_walkers=32, n_steps=1000, n_burnin=200, random_state=None, sigma_noise=100):
    """
    Compute covariance matrix using MCMC sampling with emcee.
    
    Args:
        map_params: MAP estimate (optimal parameters)
        x_data: Input data points
        y_data: Target data points
        funct_tuple: Tuple of (function, jacobian, hessian)
        mode: Model mode ('trilinear' or 'blankevoort')
        n_walkers: Number of MCMC walkers
        n_steps: Number of MCMC steps
        n_burnin: Number of burn-in steps to discard
        random_state: Random state for reproducibility
        
    Returns:
        cov_matrix: MCMC covariance matrix
        std_params: Standard deviations of parameters
        mcmc_samples: All MCMC samples
        acceptance_fraction: Acceptance fraction of the sampler
    """
    if not EMCEE_AVAILABLE:
        raise ImportError("emcee is required for MCMC sampling. Install with: pip install emcee")
    
    # print(f"Computing MCMC covariance with {n_walkers} walkers, {n_steps} steps...")
    
    n_params = len(map_params)
    
    # Get parameter bounds
    param_bounds = get_parameter_bounds(map_params, mode)
    # print(f"Parameter bounds: {param_bounds}")
    
    # Set up the MCMC sampler
    sampler = emcee.EnsembleSampler(
        n_walkers, n_params, log_probability,
        args=(x_data, y_data, funct_tuple, param_bounds, sigma_noise)
    )
    
    # Initialize walkers around MAP estimate with some scatter
    initial_positions = map_params + 1e-2 * np.random.randn(n_walkers, n_params)
    
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
    
    # print(f"MCMC completed successfully")
    # print(f"MCMC means: {mcmc_means}")
    # print(f"MCMC stds: {std_params}")
    # print(f"Acceptance fraction: {np.mean(sampler.acceptance_fraction):.3f}")
    
    return cov_matrix, std_params, samples, np.mean(sampler.acceptance_fraction)

def compute_sampling_covariance(map_params, x_data, y_data, funct_tuple, 
                               n_samples=1000, method='mcmc', mode='trilinear', sigma_noise=100):
    """
    Main function to compute covariance matrix using MCMC sampling.
    
    Args:
        map_params: MAP estimate (optimal parameters)
        x_data: Input data points
        y_data: Target data points
        funct_tuple: Tuple of (function, jacobian, hessian)
        n_samples: Number of MCMC steps (total samples = n_walkers * n_steps)
        method: Sampling method ('mcmc')
        mode: Model mode ('trilinear' or 'blankevoort')
        
    Returns:
        cov_matrix: Covariance matrix
        std_params: Standard deviations of parameters
        mcmc_samples: MCMC parameter samples
        acceptance_rate: MCMC acceptance rate
    """
    # print(f"Computing covariance using {method} method...")
    
    if method == 'mcmc':
        # Calculate number of walkers and steps based on total samples
        n_walkers = 32
        n_steps = max(100, n_samples // n_walkers)
        n_burnin = max(50, n_steps // 5)
        
        cov_matrix, std_params, mcmc_samples, acceptance_rate = compute_mcmc_covariance(
            map_params, x_data, y_data, funct_tuple, mode=mode,
            n_walkers=n_walkers, n_steps=n_steps, n_burnin=n_burnin
        )
        return cov_matrix, std_params, mcmc_samples, acceptance_rate
    else:
        raise ValueError(f"Unknown method: {method}")