from typing import List, Tuple
from data_types import RawData, ProcessedData
from math_utils import cartesian_to_spherical, spherical_to_cartesian
import numpy as np
import warnings

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
        try:
            return sample @ ~self.extension_point
        except ValueError as e:
            print(f"Failed to get relative rotation: {e}")
            print(f"Sample: {sample}")
            print(f"Extension point: {self.extension_point}")
            return sample

    def fit_regression(self) -> None:
        """Fit regression by finding the axis of rotation that best fits the data
        
        The axis of rotation is found by minimizing the squared dot product with all data points.
        This is equivalent to finding the eigenvector corresponding to the smallest eigenvalue
        of the data covariance matrix.
        """
        if len(self.regression_samples) == 0:
            raise ValueError("Cannot fit regression with no samples")
        
        # Remove outliers before fitting
        self._remove_outliers()
        
        # Get relative rotations and validate them
        relative_rotations = [
            self.relative_to_initial(sample).get_tibia_in_femur_frame() 
            for sample in self.regression_samples
        ]
        
        self._validate_rotation_matrices(relative_rotations)
        
        # Project onto unit sphere
        x_projections = [rot @ self.unit_projection for rot in relative_rotations]
        self._validate_unit_vectors(x_projections)
        
        # Stack points into matrix
        points = np.array(x_projections)
        
        # Compute covariance matrix
        covariance = points.T @ points
        
        # Find eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        
        # The axis of rotation is the eigenvector with smallest eigenvalue
        # (direction of minimum variance)
        self.plane_normal = eigenvectors[:, 0].T
        
        # The other two eigenvectors form a basis for the plane perpendicular to the axis
        self.plane_basis = eigenvectors[:, 1:].T
        
        self.regression_model = {
            'normal': self.plane_normal,
            'basis': self.plane_basis
        }

        # Plot regression results if requested
        self._plot_regression_results(points)

    def _validate_rotation_matrices(self, matrices: List[np.ndarray]) -> None:
        """Validate that all matrices are proper rotation matrices"""
        for i, matrix in enumerate(matrices):
            # Check dimensions
            if matrix.shape != (3,3):
                warnings.warn(f"Matrix {i} is not 3x3: shape = {matrix.shape}")
                continue
            
            # Check orthogonality (R * R^T = I)
            RRT = matrix @ matrix.T
            if not np.allclose(RRT, np.eye(3), rtol=1e-1, atol=1e-3):
                warnings.warn(f"Matrix {i} is not orthogonal")
            
            # Check determinant = 1 (proper rotation)
            det = np.linalg.det(matrix)
            if not np.allclose(det, 1.0, rtol=1e-3):
                warnings.warn(f"Matrix {i} is not a proper rotation matrix: det = {det}")

    def _validate_unit_vectors(self, vectors: List[np.ndarray]) -> None:
        """Validate that all vectors have unit length"""
        for i, vector in enumerate(vectors):
            if not np.allclose(np.linalg.norm(vector), 1.0, rtol=1e-3):
                print(f"WARNING: Vector {i} is not a unit vector: magnitude = {np.linalg.norm(vector)}")

    def _plot_regression_results(self, points: np.ndarray) -> None:
        """Plot regression results for visualization
        
        Args:
            points: Nx3 array of points on unit sphere
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points
        ax.scatter(points[:,0], points[:,1], points[:,2], c='blue', alpha=0.5, label='Data points')
        
        # Plot great circle
        t = np.linspace(0, 2*np.pi, 100)
        circle_points = np.zeros((len(t), 3))
        for i, theta in enumerate(t):
            # Create point on great circle at angle theta
            circle_point = np.cos(theta) * self.plane_basis[0] + np.sin(theta) * self.plane_basis[1]
            circle_points[i] = circle_point
        ax.plot(circle_points[:,0], circle_points[:,1], circle_points[:,2], 
                'r-', label='Best-fit great circle')
        
        # Plot normal vector
        ax.quiver(0, 0, 0, *self.plane_normal, color='g', label='Plane normal')
        
        # Set fixed limits
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # Set equal aspect ratio to ensure sphere appears spherical
        ax.set_box_aspect([1,1,1])
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Geodesic Regression on Unit Sphere')
        ax.legend()
        plt.show()

    def process(self, sample: RawData) -> ProcessedData:
        """Transform a RawData sample to ProcessedData using the geodesic regression
        
        Args:
            sample: RawData sample to process
            
        Returns:
            ProcessedData containing calculated joint angles
        """
        if not self.regression_model:
            raise RuntimeError("Must fit regression model before processing samples")

        # Get the relative rotation matrix
        if sample.error:
            return ProcessedData(
                flexion_angle=None,
                varus_angle=None,
                internal_angle=None,
                timestamp=sample.timestamp,
                raw_data=sample
            )
        relative_rotation = self.relative_to_initial(sample).get_tibia_in_femur_frame()

        # Project onto unit sphere (out of SO3)
        point = np.matmul(relative_rotation, self.unit_projection)
        
        # Calculate varus angle (angle between point and best-fit plane)
        # This is the angle between the point and its projection onto the plane
        varus_rad = np.arcsin(np.dot(point, self.plane_normal))
        
        # Project point onto plane of great circle
        point_proj = point - np.dot(point, self.plane_normal) * self.plane_normal
        point_proj = point_proj / np.linalg.norm(point_proj)  # normalize
        
        # Calculate flexion angle (angle in the plane from reference position)
        # Convert projected point to coordinates in plane basis
        # Get the relative rotation matrix
        extention_matrix = self.relative_to_initial(self.extension_point).get_tibia_in_femur_frame()

        # Project onto unit sphere (out of SO3)
        extension_point = np.matmul(extention_matrix, self.unit_projection)

        flexion_rad = np.arccos(np.dot(point_proj, extension_point))

        return ProcessedData(
            flexion_angle=np.rad2deg(flexion_rad),
            varus_angle=np.rad2deg(varus_rad),
            internal_angle=None,
            timestamp=sample.timestamp,
            raw_data=sample
        )
        
    def _remove_outliers(self, iqr_multiplier: float = 1.5) -> None:
        """Remove outliers from regression samples using the IQR method
        
        Args:
            iqr_multiplier: Multiplier for IQR to determine outlier threshold (default=1.5)
            
        Raises:
            ValueError: If no regression samples exist
        """
        if len(self.regression_samples) == 0:
            raise ValueError("No regression samples to process")
            
        # Get relative rotations and project onto unit sphere
        relative_rotations = [
            self.relative_to_initial(sample).get_tibia_in_femur_frame() 
            for sample in self.regression_samples
        ]
        x_projections = [rot @ self.unit_projection for rot in relative_rotations]
        
        # Convert to spherical coordinates
        spherical_coords = [
            cartesian_to_spherical(x, y, z) 
            for x, y, z in x_projections
        ]
        
        # Extract theta (azimuth) and phi (elevation)
        thetas = np.array([theta for r, theta, phi in spherical_coords])
        phis = np.array([phi for r, theta, phi in spherical_coords])
        
        # Find outliers using IQR method
        theta_mask = self._get_inlier_mask(thetas, iqr_multiplier)
        phi_mask = self._get_inlier_mask(phis, iqr_multiplier)
        
        # Combined mask (point must be inlier in both dimensions)
        inlier_mask = theta_mask & phi_mask
        
        # Keep only the inlier samples
        self.regression_samples = [
            sample for sample, is_inlier in zip(self.regression_samples, inlier_mask)
            if is_inlier
        ]
        
        print(f"Removed {len(inlier_mask) - sum(inlier_mask)} outliers from {len(inlier_mask)} total samples")

    def _get_inlier_mask(self, data: np.ndarray, iqr_multiplier: float) -> np.ndarray:
        """Get boolean mask indicating which points are inliers using IQR method
        
        Args:
            data: 1D numpy array of values to check for outliers
            iqr_multiplier: Multiplier for IQR to determine outlier threshold
            
        Returns:
            Boolean numpy array where True indicates inlier points
        """
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        return (data >= lower_bound) & (data <= upper_bound)
        
        