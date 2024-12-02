from typing import List
from data_types import RawData, ProcessedData
from math_utils import cartesian_to_spherical, spherical_to_cartesian
from sklearn.linear_model import LinearRegression
import numpy as np

class DataProcessor:
    """Provides means of creating desired data values through regression"""

    def __init__(self):
        """Initialize empty regression dataset"""
        self.regression_samples: List[RawData] = []
        self.regression_model = None
        self.unit_projection = np.array([1, 0, 0])
    
    def set_extension_point(self, sample: RawData) -> None:
        """Define the extension point of the regression"""
        self.extension_point = sample
        
    def add_to_regression(self, sample: RawData) -> None:
        """Add a sample to the regression dataset
        
        Args:
            sample: RawData sample to add to regression set
        """
        self.regression_samples.append(sample)

    def relative_to_initial(self, sample: RawData) -> RawData:
        """Get the relative rotation of the sample to the initial extension point"""
        # Need to multiply by inverse of extension point to get relative rotation
        return sample @ ~self.extension_point

    def fit_regression(self) -> None:
        """Fit regression model to the provided dataset"""
        if len(self.regression_samples) == 0:
            raise ValueError("Cannot fit regression with no samples")
        
        relative_rotation = [self.relative_to_initial(sample).get_tibia_in_femur_frame() for sample in self.regression_samples]
        
        # Validate all matrices are proper rotation matrices
        for i, rot_matrix in enumerate(relative_rotation):
            # Check dimensions
            if rot_matrix.shape != (3,3):
                print(f"Matrix value:\n{rot_matrix}")
                raise ValueError(f"Matrix {i} is not 3x3: shape = {rot_matrix.shape}")
            
            # Check orthogonality (R * R^T = I)
            RRT = np.matmul(rot_matrix, rot_matrix.T)
            if not np.allclose(RRT, np.eye(3), rtol=1e-2, atol=1e-4):
                print(f"Matrix value:\n{rot_matrix}")
                print(f"R*R^T value:\n{RRT}")
                raise ValueError(f"Matrix {i} is not orthogonal")
                
            # Check determinant = 1 (proper rotation)
            det = np.linalg.det(rot_matrix)
            if not np.allclose(det, 1.0, rtol=1e-3):
                print(f"Matrix value:\n{rot_matrix}")
                print(f"Determinant value: {det}")
                raise ValueError(f"Matrix {i} is not a proper rotation matrix: det = {det}")

        x_projection = [np.matmul(relative_rotation, self.unit_projection) for relative_rotation in relative_rotation]        
        
        # Verify all x_projections are unit vectors
        for i, proj in enumerate(x_projection):
            if not np.allclose(np.linalg.norm(proj), 1.0, rtol=1e-3):
                raise ValueError(f"x_projection {i} is not a unit vector: magnitude = {np.linalg.norm(proj)}")

        # Get spherical coordinates where:
        # phi (elevation) is measured from +z axis (phi=0 at north pole, phi=pi/2 at equator)
        # theta (azimuth) is measured in x-y plane from +x axis
        spherical_coords = [cartesian_to_spherical(x, y, z) for x, y, z in x_projection]

        # Validate all radii are 1 (unit vectors)
        for i, (r, theta, phi) in enumerate(spherical_coords):
            if not np.allclose(r, 1.0, rtol=1e-3):
                raise ValueError(f"Vector {i} is not unit length: radius = {r}")

        # Remove radius since all vectors are unit length
        theta_phi_coords = [(theta, phi) for r, theta, phi in spherical_coords]

        # Fit regression model to predict elevation (phi) from azimuth (theta)
        # phi=0 is at north pole, phi=pi/2 is at equator
        X = np.array([theta for theta, phi in theta_phi_coords]).reshape(-1, 1)  
        y = np.array([phi for theta, phi in theta_phi_coords])  # elevation angle
        
        # Shift y values by pi/2 to force regression through equator
        # y_shifted = y - np.pi/2
        self.regression_model = LinearRegression()
        self.regression_model.intercept_ = np.pi/2
        self.regression_model.fit(X, y)
        # Add pi/2 back to the model's intercept

        # Plot regression results
        import matplotlib.pyplot as plt
        
        # Create scatter plot of actual data points
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', alpha=0.5, label='Data points')
        
        # Plot regression line
        X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred = self.regression_model.predict(X_plot)
        plt.plot(X_plot, y_pred, color='red', label='Regression line')
        
        # Add horizontal line at y=pi/2 (equator)
        # plt.axhline(y=np.pi/2, color='grey', linestyle='--', label='Equator (π/2)')
        
        plt.xlabel('Azimuth angle (radians)')
        plt.ylabel('Elevation angle (radians from north pole)') 
        plt.title('Regression of Elevation vs Azimuth')
        plt.legend()
        plt.grid(True)

        # Create 3D visualization of regression points and line
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot original data points in 3D
        # Convert theta,phi coordinates back to 3D unit vectors
        data_points_3d = [spherical_to_cartesian(1.0, theta, phi) 
                         for theta, phi in theta_phi_coords]
        data_x = [p[0] for p in data_points_3d]
        data_y = [p[1] for p in data_points_3d]
        data_z = [p[2] for p in data_points_3d]
        ax.scatter(data_x, data_y, data_z, color='blue', alpha=0.5, label='Data points')

        # Plot regression line in 3D
        # Convert predicted points to 3D coordinates
        line_points_3d = [spherical_to_cartesian(1.0, theta[0], phi) 
                         for theta, phi in zip(X_plot, y_pred)]
        line_x = [p[0] for p in line_points_3d]
        line_y = [p[1] for p in line_points_3d]
        line_z = [p[2] for p in line_points_3d]
        ax.plot(line_x, line_y, line_z, color='red', label='Regression line')

        # Add labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Regression (Unit Sphere)')
        ax.legend()

        # Set equal aspect ratio to make sphere look spherical
        ax.set_box_aspect([1,1,1])
        plt.show()

    def process(self, sample: RawData) -> ProcessedData:
        """Transform a RawData sample to ProcessedData using the regression model
        
        Args:
            sample: RawData sample to process
            
        Returns:
            ProcessedData containing calculated joint angles
            
        Raises:
            RuntimeError: If regression model hasn't been fit yet
        """
        if not self.regression_model:
            raise RuntimeError("Must fit regression model before processing samples")

        # Get the relative rotation matrix
        relative_rotation = self.relative_to_initial(sample).get_tibia_in_femur_frame()

        # Project the relative rotation onto the unit sphere
        x_projection = np.matmul(relative_rotation, self.unit_projection)
        r, theta, phi = cartesian_to_spherical(x_projection[0], x_projection[1], x_projection[2])

        # Get the regression line parameters (in radians)
        m = self.regression_model.coef_[0]  # slope
        b = 0  # intercept (forced through origin)

        # Calculate signed perpendicular distance (varus angle in radians)
        # Using the signed point-to-line formula: (m*x - y + b) / sqrt(m^2 + 1)
        varus_rad = (m*theta - phi + b) / np.sqrt(m*m + 1)

        # Calculate projection onto regression line (flexion angle)
        # Project point (theta, phi) onto line y = mx + b
        theta_proj = (theta + m*phi)/(m*m + 1)  # x coordinate of projection
        phi_proj = m*theta_proj  # y coordinate of projection (since b=0)

        # Calculate signed flexion angle in radians
        # Use atan2 to get signed angle including quadrant information
        flexion_rad = np.arctan2(phi_proj, theta_proj)

        # Calculate internal rotation angle from the relative rotation matrix (in radians)
        # internal_rad = np.arctan2(relative_rotation[1,2], relative_rotation[2,2])

        # Convert all angles to degrees for output
        return ProcessedData(
            flexion_angle=np.rad2deg(flexion_rad),
            varus_angle=np.rad2deg(varus_rad),
            internal_angle=None,
            timestamp=sample.timestamp,
            raw_data=sample
        )
        
        