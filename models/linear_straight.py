import numpy as np
import matplotlib.pyplot as plt

# Parameter variations
Ls_values = [0.5]     # Different rest lengths
L_perp_values = [0.05,0.10,0.25] # Different perpendicular offsets
sigma2 = 0.01   # Measurement noise variance

# Theta range
theta = np.linspace(-np.pi, np.pi, 500)

# Create a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Spring Analysis Metrics for Different Parameters', fontsize=16)

# Colors for different parameter combinations
colors = plt.cm.viridis(np.linspace(0, 1, len(Ls_values) * len(L_perp_values)))

# Run experiments for each parameter combination
color_idx = 0
for L0 in Ls_values:
    for L_perp in L_perp_values:
        # Anchor‐difference components in ground frame
        Px = L_perp*(np.cos(theta) - 1) + (L0/2)*np.sin(theta)
        Py = L_perp*np.sin(theta) - (L0/2)*(np.cos(theta) + 1)

        # Derivatives of components w.r.t. theta
        dPx = -L_perp*np.sin(theta) + (L0/2)*np.cos(theta)
        dPy =  L_perp*np.cos(theta) + (L0/2)*np.sin(theta)

        # Spring length L(theta)
        L = np.sqrt(Px**2 + Py**2)

        # Derivative dL/dtheta
        dL = (Px*dPx + Py*dPy) / L

        # Sensitivity S(theta) = |dL/dtheta|
        sensitivity = np.abs(dL)

        # Fisher Information I(theta) = (dL/dtheta)^2 / sigma2
        Fisher = (dL**2) / sigma2

        # Plot each metric with a label indicating parameters
        label = f'L0={L0}, L_perp={L_perp}'
        
        # 1) Spring length vs. theta
        axs[0, 0].plot(theta, L, color=colors[color_idx], label=label)
        axs[0, 0].set_xlabel(r'$\theta$ (rad)')
        axs[0, 0].set_ylabel(r'$L(\theta)$')
        axs[0, 0].set_title(r'Spring Length $L(\theta)$ vs. Angle $\theta$')
        axs[0, 0].grid(True)

        # 2) Derivative dL/dtheta vs. theta
        axs[0, 1].plot(theta, dL, color=colors[color_idx], label=label)
        axs[0, 1].set_xlabel(r'$\theta$ (rad)')
        axs[0, 1].set_ylabel(r'$\dfrac{dL}{d\theta}$')
        axs[0, 1].set_title(r'Derivative $\dfrac{dL}{d\theta}$ vs. Angle $\theta$')
        axs[0, 1].grid(True)

        # 3) Sensitivity |dL/dtheta| vs. theta
        axs[1, 0].plot(theta, sensitivity, color=colors[color_idx], label=label)
        axs[1, 0].set_xlabel(r'$\theta$ (rad)')
        axs[1, 0].set_ylabel(r'$|dL/d\theta|$')
        axs[1, 0].set_title(r'Sensitivity $|dL/d\theta|$ vs. Angle $\theta$')
        axs[1, 0].grid(True)

        # 4) Fisher Information vs. theta
        axs[1, 1].plot(theta, Fisher, color=colors[color_idx], label=label)
        axs[1, 1].set_xlabel(r'$\theta$ (rad)')
        axs[1, 1].set_ylabel(r'$I(\theta)$')
        axs[1, 1].set_title(r'Fisher Information $I(\theta)$ vs. Angle $\theta$')
        axs[1, 1].grid(True)

        color_idx += 1

# Add legends to each subplot
for ax in axs.flat:
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()
