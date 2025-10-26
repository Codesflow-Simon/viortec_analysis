# Ligament Analysis Guide

## Overview

The analysis system now processes **both MCL and LCL ligaments** separately in each experiment. All results are stored together with a `ligament_type` column to distinguish between them.

## Changes Made

### 1. Main Analysis (`main.py`)
- **New function**: `process_ligament()` - Handles optimization and MCMC for a single ligament
- **Updated `main()`**: Now processes both LCL and MCL separately
- **CSV output**: Added `ligament_type` column (first column)
- **Data structure**: Each experiment now returns `{'LCL': result_lcl, 'MCL': result_mcl}`

### 2. Visualization Scripts
Both `visualize_diagonal.py` and `plot_diagonal_fits.py` now accept a ligament type argument:

```bash
# Default: MCL
python visualize_diagonal.py results/summary_YYYYMMDD_HHMMSS.csv

# Specify LCL
python visualize_diagonal.py results/summary_YYYYMMDD_HHMMSS.csv LCL

# Specify MCL explicitly
python visualize_diagonal.py results/summary_YYYYMMDD_HHMMSS.csv MCL
```

## Running the Analysis

### Step 1: Generate Data

```bash
python main.py
```

This will:
- Run experiments for all LCL/MCL strain combinations
- Process **both ligaments** in each experiment
- Save results to:
  - `results/results_TIMESTAMP.pkl` - Complete results
  - `results/summary_TIMESTAMP.csv` - CSV with ligament_type column
  - `results/samples_TIMESTAMP/` - Individual MCMC samples

### Step 2: Visualize Results

#### Option A: Generate Summary Plots for MCL
```bash
python visualize_diagonal.py results/summary_TIMESTAMP.csv MCL
```

Generates:
- `parameter_estimates_diagonal.png` - Parameter recovery plots
- `parameter_errors_diagonal.png` - Relative error plots
- `parameter_uncertainties_diagonal.png` - MCMC uncertainty
- `loss_comparison_diagonal.png` - Optimization performance
- `mcmc_metrics_diagonal.png` - MCMC diagnostics
- `theta_statistics_diagonal.png` - Theta range statistics

#### Option B: Generate Summary Plots for LCL
```bash
python visualize_diagonal.py results/summary_TIMESTAMP.csv LCL
```

#### Option C: Generate Individual Fit Plots for MCL
```bash
python plot_diagonal_fits.py results/results_TIMESTAMP.pkl MCL
```

Generates:
- `fit_strain_X.XX.png` - Individual fit plots for each strain value
- `all_fits_combined.png` - All fits in one figure

#### Option D: Generate Individual Fit Plots for LCL
```bash
python plot_diagonal_fits.py results/results_TIMESTAMP.pkl LCL
```

## CSV Structure

The summary CSV now has the following structure:

```csv
ligament_type,lcl_strain,mcl_strain,acceptance_rate,initial_loss,final_loss,...
MCL,0.02,0.02,0.4321,35123.45,145.67,...
LCL,0.02,0.02,0.4156,32456.78,152.34,...
MCL,0.02,0.04,0.4289,36789.12,149.23,...
LCL,0.02,0.04,0.4201,33567.89,155.67,...
...
```

Each strain combination now has **two rows**: one for MCL and one for LCL.

## Output Directories

Visualization outputs are organized by ligament type:

```
results/
├── summary_20251009_HHMMSS.csv
├── summary_20251009_HHMMSS_mcl_diagonal_plots/
│   ├── parameter_estimates_diagonal.png
│   ├── parameter_errors_diagonal.png
│   └── ...
├── summary_20251009_HHMMSS_lcl_diagonal_plots/
│   ├── parameter_estimates_diagonal.png
│   ├── parameter_errors_diagonal.png
│   └── ...
├── results_20251009_HHMMSS_mcl_diagonal_fits/
│   ├── fit_strain_0.02.png
│   ├── fit_strain_0.04.png
│   └── all_fits_combined.png
└── results_20251009_HHMMSS_lcl_diagonal_fits/
    ├── fit_strain_0.02.png
    ├── fit_strain_0.04.png
    └── all_fits_combined.png
```

## Quick Examples

### Compare MCL vs LCL performance
```bash
# Generate MCL plots
python visualize_diagonal.py results/summary_20251009_120000.csv MCL

# Generate LCL plots
python visualize_diagonal.py results/summary_20251009_120000.csv LCL

# Compare the parameter_estimates_diagonal.png files side by side
```

### Generate all visualizations for both ligaments
```bash
TIMESTAMP="20251009_120000"

# MCL visualizations
python visualize_diagonal.py results/summary_${TIMESTAMP}.csv MCL
python plot_diagonal_fits.py results/results_${TIMESTAMP}.pkl MCL

# LCL visualizations
python visualize_diagonal.py results/summary_${TIMESTAMP}.csv LCL
python plot_diagonal_fits.py results/results_${TIMESTAMP}.pkl LCL
```

## Notes

- **Default behavior**: If no ligament type is specified, **MCL** is used by default
- **Diagonal cases**: Visualization scripts filter for cases where `lcl_strain == mcl_strain`
- **Monte Carlo samples**: Grey lines show MCMC uncertainty (100 random samples)
- **Reference forces**: Now calculated separately for LCL and MCL at theta=0

## Data Structure in Pickle File

```python
results = [
    {
        'lcl_strain': 0.02,
        'mcl_strain': 0.02,
        'result': {
            'LCL': {
                'samples': array(...),
                'gt_params': {...},
                'optimized_params': {...},
                ...
            },
            'MCL': {
                'samples': array(...),
                'gt_params': {...},
                'optimized_params': {...},
                ...
            }
        }
    },
    ...
]
```



