import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_types import RawData, ProcessedData
from math_utils import cartesian_to_spherical, spherical_to_cartesian
from collections import deque
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation

class LiveVisualizer:
    def __init__(self, history_length=100):
        """Initialize the live visualizer"""
        # Create figure with two subplots
        self.fig = plt.figure(figsize=(15, 6))
        
        # 2D plot for theta vs phi
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_2d.set_xlabel('Azimuth angle (radians)')
        self.ax_2d.set_ylabel('Elevation angle (radians)')
        self.ax_2d.set_title('Spherical Coordinates View')
        self.ax_2d.grid(True)
        
        # 3D plot for unit sphere visualization
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.ax_3d.set_xlabel('X')
        self.ax_3d.set_ylabel('Y')
        self.ax_3d.set_zlabel('Z')
        self.ax_3d.set_title('3D Unit Sphere View')
        
        # Set fixed limits for 3D plot
        self.ax_3d.set_xlim(-1, 1)
        self.ax_3d.set_ylim(-1, 1)
        self.ax_3d.set_zlim(-1, 1)
        
        # Pre-allocate arrays for better performance
        self.history_length = history_length
        self.theta_history = np.zeros(history_length)
        self.phi_history = np.zeros(history_length)
        self.x_history = np.zeros(history_length)
        self.y_history = np.zeros(history_length)
        self.z_history = np.zeros(history_length)
        self.current_idx = 0
        
        # Initialize plots
        self.scatter_2d, = self.ax_2d.plot([], [], 'bo-', alpha=0.5, label='Data')
        self.scatter_3d, = self.ax_3d.plot([], [], [], 'bo-', alpha=0.5, label='Data')
        self.regression_2d, = self.ax_2d.plot([], [], 'r-', label='Regression')
        
        # Set fixed limits for 2D plot
        self.ax_2d.set_xlim(-np.pi, np.pi)
        self.ax_2d.set_ylim(-np.pi/2, np.pi/2)
        
        # Draw unit sphere
        self._draw_unit_sphere()
        
        # Add legends
        self.ax_2d.legend()
        self.ax_3d.legend()
        
        # Tight layout
        self.fig.tight_layout()
        plt.ion()
        self.fig.show()

    def _draw_unit_sphere(self):
        """Draw wireframe unit sphere with reduced resolution"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))
        self.ax_3d.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)

    def set_regression_line(self, regression_model):
        """Add regression line to 2D plot
        
        Args:
            regression_model: Dictionary containing 'normal' and 'basis' vectors for the great circle
        """
        # Generate points along the great circle
        t = np.linspace(0, 2*np.pi, 100)
        circle_points = np.zeros((len(t), 3))
        for i, theta in enumerate(t):
            # Combine basis vectors with angle
            p = np.cos(theta) * regression_model['basis'][0] + np.sin(theta) * regression_model['basis'][1]
            circle_points[i] = p
        
        # Convert 3D points to spherical coordinates for 2D plot
        thetas = []
        phis = []
        for point in circle_points:
            r, theta, phi = cartesian_to_spherical(point[0], point[1], point[2])
            thetas.append(theta)
            phis.append(phi)
        
        # Sort points by theta for continuous line
        sorted_indices = np.argsort(thetas)
        thetas = np.array(thetas)[sorted_indices]
        phis = np.array(phis)[sorted_indices]
        
        # Update 2D regression line
        self.regression_2d.set_data(thetas, phis)
        
        # Force redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_plot(self, frame):
        """Animation update function"""
        # Update 2D plot
        valid_data = self.theta_history[:self.current_idx]
        valid_phi = self.phi_history[:self.current_idx]
        self.scatter_2d.set_data(valid_data, valid_phi)
        
        # Update 3D plot
        valid_x = self.x_history[:self.current_idx]
        valid_y = self.y_history[:self.current_idx]
        valid_z = self.z_history[:self.current_idx]
        self.scatter_3d.set_data_3d(valid_x, valid_y, valid_z)
        
        return self.scatter_2d, self.scatter_3d

    def update(self, sample: RawData):
        """Update visualization with new sample"""
        # Get unit vector projection
        unit_projection = np.array([1, 0, 0])
        relative_rotation = sample.get_tibia_in_femur_frame()
        
        if relative_rotation is None:
            return
            
        x_projection = np.matmul(relative_rotation, unit_projection)
        
        # Convert to spherical coordinates
        r, theta, phi = cartesian_to_spherical(x_projection[0], x_projection[1], x_projection[2])
        
        # Update circular buffer
        idx = self.current_idx % self.history_length
        self.theta_history[idx] = theta
        self.phi_history[idx] = phi
        self.x_history[idx] = x_projection[0]
        self.y_history[idx] = x_projection[1]
        self.z_history[idx] = x_projection[2]
        
        self.current_idx += 1
        
        # Get valid data range
        valid_range = slice(0, min(self.current_idx, self.history_length))
        
        # Update plots
        self.scatter_2d.set_data(self.theta_history[valid_range], self.phi_history[valid_range])
        self.scatter_3d.set_data_3d(
            self.x_history[valid_range],
            self.y_history[valid_range],
            self.z_history[valid_range]
        )
        
        # Draw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _is_valid_rotation_matrix(self, R):
        """Check if matrix is a valid rotation matrix"""
        # Check dimensions
        if R.shape != (3, 3):
            return False
            
        # Check orthogonality (R * R^T = I)
        I = np.eye(3)
        if not np.allclose(R @ R.T, I, rtol=1e-3, atol=1e-3):
            return False
            
        # Check proper rotation (det = 1)
        if not np.isclose(np.linalg.det(R), 1.0, rtol=1e-3):
            return False
            
        return True

    def close(self):
        """Close the visualization window"""
        plt.close(self.fig) 

class RotationVisualizer:
    def __init__(self):
        """Initialize the rotation visualizer"""
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Rotation Matrix Unit Vectors')
        
        # Set fixed limits
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.set_zlim(-1, 1)
        
        # Initialize unit vector plots (x=red, y=green, z=blue)
        self.x_vector, = self.ax.plot([0, 1], [0, 0], [0, 0], 'r-', linewidth=2, label='X axis')
        self.y_vector, = self.ax.plot([0, 0], [0, 1], [0, 0], 'g-', linewidth=2, label='Y axis')
        self.z_vector, = self.ax.plot([0, 0], [0, 0], [0, 1], 'b-', linewidth=2, label='Z axis')
        
        # Draw reference frame
        self._draw_reference_frame()
        
        # Set 3D plot aspects
        self.ax.set_box_aspect([1,1,1])
        
        # Add legend
        self.ax.legend()
        
        # Set up plot for better performance
        plt.ion()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        # Pre-allocate arrays
        self.x_data = np.zeros((2, 3))
        self.y_data = np.zeros((2, 3))
        self.z_data = np.zeros((2, 3))
        
        # Remove animation for better performance
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
            del self.ani

    def _draw_reference_frame(self):
        """Draw light gray reference frame"""
        # Draw light gray reference axes
        self.ax.plot([0, 1], [0, 0], [0, 0], 'gray', alpha=0.3, linestyle='--')
        self.ax.plot([0, 0], [0, 1], [0, 0], 'gray', alpha=0.3, linestyle='--')
        self.ax.plot([0, 0], [0, 0], [0, 1], 'gray', alpha=0.3, linestyle='--')

    def _update_plot(self, frame):
        return self.x_vector, self.y_vector, self.z_vector

    def update(self, sample: RawData):
        """Update visualization with new sample"""
        rotation = sample.get_tibia_in_femur_frame()
        if rotation is None:
            return
            
        # Update pre-allocated arrays
        self.x_data[1] = rotation[:, 0]
        self.y_data[1] = rotation[:, 1]
        self.z_data[1] = rotation[:, 2]
        
        # Restore background
        self.fig.canvas.restore_region(self.background)
        
        # Update vectors
        self.x_vector.set_data_3d(self.x_data[:, 0], self.x_data[:, 1], self.x_data[:, 2])
        self.y_vector.set_data_3d(self.y_data[:, 0], self.y_data[:, 1], self.y_data[:, 2])
        self.z_vector.set_data_3d(self.z_data[:, 0], self.z_data[:, 1], self.z_data[:, 2])
        
        # Redraw just the vectors
        self.ax.draw_artist(self.x_vector)
        self.ax.draw_artist(self.y_vector)
        self.ax.draw_artist(self.z_vector)
        
        # Blit the display
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization window"""
        plt.close(self.fig) 

class LegVisualizer:
    def __init__(self, femur_length=0.4, tibia_length=0.4):
        """Initialize the leg visualizer
        
        Args:
            femur_length: Length of femur segment in arbitrary units
            tibia_length: Length of tibia segment in arbitrary units
        """
        self.fig = plt.figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set labels and title
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Leg Orientation')
        
        # Store segment lengths
        self.femur_length = femur_length
        self.tibia_length = tibia_length
        
        # Set fixed limits with padding
        limit = (femur_length + tibia_length) * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_zlim(-limit, limit)
        
        # Initialize segment plots
        # Femur in red, tibia in blue
        self.femur_line, = self.ax.plot([], [], [], 'r-', linewidth=3, label='Femur')
        self.tibia_line, = self.ax.plot([], [], [], 'b-', linewidth=3, label='Tibia')
        
        # Draw reference frame
        self._draw_reference_frame()
        
        # Set 3D plot aspects
        self.ax.set_box_aspect([1,1,1])
        
        # Add legend
        self.ax.legend()
        
        # Set up plot for better performance
        plt.ion()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        
        # Pre-allocate arrays
        self.femur_data = np.zeros((2, 3))
        self.tibia_data = np.zeros((2, 3))
        
        # Remove animation for better performance
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
            del self.ani

    def _draw_reference_frame(self):
        """Draw light gray reference frame"""
        # Draw light gray reference axes
        self.ax.plot([0, 1], [0, 0], [0, 0], 'gray', alpha=0.3, linestyle='--')
        self.ax.plot([0, 0], [0, 1], [0, 0], 'gray', alpha=0.3, linestyle='--')
        self.ax.plot([0, 0], [0, 0], [0, 1], 'gray', alpha=0.3, linestyle='--')

    def _update_plot(self, frame):
        return self.femur_line, self.tibia_line

    def update(self, sample: RawData):
        """Update visualization with new sample"""
        # Get rotation matrices
        femur_rotation = sample.femur_rotation
        tibia_rotation = sample.tibia_rotation
        if femur_rotation is None or tibia_rotation is None:
            return
            
        # Calculate endpoints
        femur_direction = femur_rotation @ np.array([0, 0, 1])
        femur_end = femur_direction * self.femur_length
        
        tibia_direction = tibia_rotation @ np.array([0, 0, 1])
        tibia_end = femur_end + (tibia_direction * self.tibia_length)
        
        # Update pre-allocated arrays
        self.femur_data[1] = femur_end
        self.tibia_data[0] = femur_end
        self.tibia_data[1] = tibia_end
        
        # Restore background
        self.fig.canvas.restore_region(self.background)
        
        # Update lines
        self.femur_line.set_data_3d(self.femur_data[:, 0],
                                   self.femur_data[:, 1],
                                   self.femur_data[:, 2])
        self.tibia_line.set_data_3d(self.tibia_data[:, 0],
                                   self.tibia_data[:, 1],
                                   self.tibia_data[:, 2])
        
        # Redraw just the lines
        self.ax.draw_artist(self.femur_line)
        self.ax.draw_artist(self.tibia_line)
        
        # Blit the display
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization window"""
        plt.close(self.fig)

class AngleVisualizer:
    def __init__(self, history_length=200):
        """Initialize the angle visualizer with deque for efficient data storage"""
        self.fig, (self.ax_flex, self.ax_var) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Setup flexion/extension plot
        self.ax_flex.set_xlabel('Samples')
        self.ax_flex.set_ylabel('Angle (degrees)')
        self.ax_flex.set_title('Flexion/Extension Angle')
        self.ax_flex.grid(True)
        
        # Setup varus/valgus plot
        self.ax_var.set_xlabel('Samples')
        self.ax_var.set_ylabel('Angle (degrees)')
        self.ax_var.set_title('Varus/Valgus Angle')
        self.ax_var.grid(True)
        
        # Initialize deques with maxlen for automatic size management
        self.history_length = history_length
        self.flexion_data = deque([0] * history_length, maxlen=history_length)
        self.varus_data = deque([0] * history_length, maxlen=history_length)
        self.x_data = np.arange(history_length)
        
        # Initialize plots
        self.flex_line, = self.ax_flex.plot(self.x_data, self.flexion_data, 'b-', 
                                          label='Flexion/Extension')
        self.var_line, = self.ax_var.plot(self.x_data, self.varus_data, 'r-', 
                                        label='Varus/Valgus')
        
        # Add legends
        self.ax_flex.legend()
        self.ax_var.legend()
        
        # Set fixed limits
        self.ax_flex.set_xlim(0, history_length)
        self.ax_var.set_xlim(0, history_length)
        self.ax_flex.set_ylim(-90, 90)
        self.ax_var.set_ylim(-45, 45)
        
        # Setup plot for better performance
        plt.ion()
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def update(self, processed_data: ProcessedData):
        """Update visualization with new processed angles using deque"""
        if processed_data is None:
            return
            
        # Update deques (automatically handles rolling)
        if not np.isnan(processed_data.flexion_angle):
            self.flexion_data.append(processed_data.flexion_angle)
        if not np.isnan(processed_data.varus_angle):
            self.varus_data.append(processed_data.varus_angle)
        
        # Restore background
        self.fig.canvas.restore_region(self.background)
        
        # Update line data
        self.flex_line.set_ydata(self.flexion_data)
        self.var_line.set_ydata(self.varus_data)
        
        # Redraw just the lines
        self.ax_flex.draw_artist(self.flex_line)
        self.ax_var.draw_artist(self.var_line)
        
        # Blit the display
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def close(self):
        """Close the visualization window"""
        plt.close(self.fig)