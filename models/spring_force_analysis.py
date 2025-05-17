import numpy as np
import matplotlib.pyplot as plt

class SpringJointSystem:
    def __init__(self, k, L0, LS, L_perp):
        """
        Initialize the spring-joint system
        
        Parameters:
        k: float - Spring constant
        L0: float - Rest length of spring
        LS: float - Straight leg length
        L_perp: float - Perpendicular offset
        """
        self.k = k
        self.L0 = L0
        self.LS = LS
        self.L_perp = L_perp
    
    def calculate_spring_force(self, theta):
        """
        Calculate the spring force vector and its application point
        
        Parameters:
        theta: float - Joint angle in radians
        
        Returns:
        F_spring: np.array - Spring force vector [Fx, Fy]
        P_spring: np.array - Spring force application point [Px, Py]
        """
        # Calculate spring attachment points
        Px = self.L_perp*(np.cos(theta) - 1) + (self.LS/2)*np.sin(theta)
        Py = self.L_perp*np.sin(theta) - (self.LS/2)*(np.cos(theta) + 1)
        
        # Calculate spring displacement vector
        delta_P = np.array([Px, Py])
        delta_P_mag = np.linalg.norm(delta_P)
        
        # Calculate spring force magnitude
        F_mag = self.k * (self.L0 - delta_P_mag)
        
        # Calculate spring force vector
        F_spring = F_mag * delta_P / delta_P_mag
        
        # Spring force application point
        P_spring = np.array([
            self.L_perp*np.cos(theta) + (self.LS/2)*np.sin(theta),
            self.L_perp*np.sin(theta) - (self.LS/2)*np.cos(theta)
        ])
        
        return F_spring, P_spring, delta_P_mag
    
    def calculate_balancing_forces(self, theta):
        """
        Calculate the applied force and joint reaction force
        
        Parameters:
        theta: float - Joint angle in radians
        
        Returns:
        F_applied: np.array - Applied force at joint [Fx, Fy]
        F_joint: np.array - Joint reaction force [Fx, Fy]
        """
        # Get spring force and its application point
        F_spring, P_spring, _ = self.calculate_spring_force(theta)
        
        # Calculate torque from spring force
        torque_spring = np.cross(P_spring, F_spring)
        # This cross returns a scalar since vectors are 2D
        
        # Calculate applied force to balance torque
        # Applied force is at [LS/2 * sin(theta), -LS/2 * cos(theta)]
        P_applied = np.array([
            (self.LS/2)*np.sin(theta),
            -(self.LS/2)*np.cos(theta)
        ])
        
        torque_applied = -torque_spring
        
        # We want to solve for F_applied such that F_applied * P_applied = torque_applied
        # For a 2D system, the cross product of two vectors [x1,y1] and [x2,y2] is x1*y2 - x2*y1
        # We want: P_applied × F_applied = torque_applied
        # This means: P_applied[0]*F_applied[1] - P_applied[1]*F_applied[0] = torque_applied
        
        # The solution is not unique, but we can find a particular solution
        # by setting F_applied perpendicular to P_applied
        # This means F_applied = k * [-P_applied[1], P_applied[0]]
        
        # To find k, we solve:
        # P_applied[0]*(k*P_applied[0]) - P_applied[1]*(k*-P_applied[1]) = torque_applied
        # k*(P_applied[0]^2 + P_applied[1]^2) = torque_applied
        # k = torque_applied / |P_applied|^2
        
        P_applied_mag_squared = np.dot(P_applied, P_applied)
        k = torque_applied / P_applied_mag_squared
        
        # Calculate F_applied
        F_applied = k * np.array([-P_applied[1], P_applied[0]])
        
        # Calculate joint reaction force to balance linear forces
        F_joint = -F_spring - F_applied
        
        return F_applied, F_joint

def analyze_forces(k, L0, LS, L_perp, theta_range=(-np.pi/2, np.pi/2), n_points=100):
    """
    Analyze forces for a given set of parameters
    
    Parameters:
    k: float - Spring constant
    L0: float - Rest length
    LS: float - Straight leg length
    L_perp: float - Perpendicular offset
    theta_range: tuple - Range of angles to analyze
    n_points: int - Number of points in the analysis
    """
    # Create system
    system = SpringJointSystem(k, L0, LS, L_perp)
    
    # Calculate forces for a range of angles
    theta = np.linspace(theta_range[0], theta_range[1], n_points)
    F_applied = np.zeros((len(theta), 2))
    F_joint = np.zeros((len(theta), 2))
    F_spring = np.zeros((len(theta), 2))
    delta_P_mags = np.zeros(len(theta))
    
    for i, t in enumerate(theta):
        F_spring[i], _, delta_P_mags[i] = system.calculate_spring_force(t)
        F_applied[i], F_joint[i] = system.calculate_balancing_forces(t)
    
    # Calculate spring energy (E = 1/2 * k * (L0 - L)^2)
    spring_energy = 0.5 * k * (L0 - delta_P_mags)**2
    
    # Create figure with four subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Force Analysis (k={k}, L0={L0}, LS={LS}, L_perp={L_perp})', fontsize=16)
    
    # Plot 1: Spring force components
    ax1 = axs[0, 0]
    ax1.plot(theta, F_spring[:, 0], 'b-', label='Fx')
    ax1.plot(theta, F_spring[:, 1], 'b--', label='Fy')
    ax1.set_xlabel('θ (rad)')
    ax1.set_ylabel('Force Components (N)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    ax1.legend(loc='upper right')
    ax1.set_title('Spring Force Components')
    
    # Plot 2: Applied force components
    ax2 = axs[0, 1]
    ax2.plot(theta, F_applied[:, 0], 'r-', label='Fx')
    ax2.plot(theta, F_applied[:, 1], 'r--', label='Fy')
    ax2.set_xlabel('θ (rad)')
    ax2.set_ylabel('Force Components (N)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.grid(True)
    ax2.legend(loc='upper right')
    ax2.set_title('Applied Force Components')
    
    # Plot 3: Joint reaction force components
    ax3 = axs[1, 0]
    ax3.plot(theta, F_joint[:, 0], 'g-', label='Fx')
    ax3.plot(theta, F_joint[:, 1], 'g--', label='Fy')
    ax3.set_xlabel('θ (rad)')
    ax3.set_ylabel('Force Components (N)', color='g')
    ax3.tick_params(axis='y', labelcolor='g')
    ax3.grid(True)
    ax3.legend(loc='upper right')
    ax3.set_title('Joint Reaction Force Components')
    
    # Plot 4: Spring Energy
    ax4 = axs[1, 1]
    ax4.plot(theta, spring_energy)
    ax4.set_xlabel('θ (rad)')
    ax4.set_ylabel('Spring Energy (J)')
    ax4.set_title('Spring Energy vs. Angle')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Example parameters
    k = 1000.0  # Spring constant (N/m)
    L0 = 0.08    # Rest length (m)
    LS = 0.1    # Straight leg length (m)
    L_perp = 0.05  # Perpendicular offset (m)
    
    # Run analysis
    analyze_forces(k, L0, LS, L_perp) 