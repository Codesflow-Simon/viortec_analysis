import time
import numpy as np
import cProfile
import pstats
from src.ligament_reconstructor.mcmc_sampler import CompleteMCMCSampler
from src.ligament_models.constraints import ConstraintManager
import yaml

print("Loading configuration...")
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open('constraints.yaml', 'r') as f:
    constraints_config = yaml.safe_load(f)

# Set up constraint managers
mcl_constraints = constraints_config['blankevoort_mcl']
lcl_constraints = constraints_config['blankevoort_lcl']

mcl_constraint_manager = ConstraintManager(mcl_constraints)
lcl_constraint_manager = ConstraintManager(lcl_constraints)

# Create sample data
print("Generating sample data...")
thetas = np.linspace(0, 0.3, 10)  # 10 data points
applied_forces = np.random.randn(10) * 10 + 50

# Generate ligament lengths (placeholder values)
pre_compute_lcl_lengths = np.ones(10) * 60
pre_compute_mcl_lengths = np.ones(10) * 85

# Create sampler
knee_config = config['mechanics']
constraint_manager = [mcl_constraint_manager, lcl_constraint_manager]

sampler = CompleteMCMCSampler(
    knee_config, 
    constraint_manager,
    n_walkers=32,
    n_steps=2
)

# Set the ligament lengths that are needed for the likelihood
sampler.lcl_lengths = pre_compute_lcl_lengths
sampler.mcl_lengths = pre_compute_mcl_lengths

print("Testing log_likelihood with profiling...")
print("=" * 60)

# Create some test parameters
test_params = np.array([
    33.5, 0.06, 86.2, 0.0,  # MCL params
    42.8, 0.06, 57.36, 0.0  # LCL params
])

# Profile the log_likelihood function
profiler = cProfile.Profile()
profiler.enable()

start = time.time()
for i in range(3):
    ll = sampler.log_likelihood(test_params, thetas, applied_forces, sigma_noise=1e2)
    print(f"Trial {i+1}: log_likelihood = {ll:.2f}")
elapsed = time.time() - start

profiler.disable()

print(f"\nAverage time per call: {elapsed/3:.3f} seconds")
print("=" * 60)
print("PROFILING RESULTS")
print("=" * 60)

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
print("\nTop 20 functions by cumulative time:")
stats.print_stats(20)

stats.sort_stats('tottime')
print("\nTop 20 functions by total time:")
stats.print_stats(20)
