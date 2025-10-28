import sympy
import numpy as np
import matplotlib.pyplot as plt
import yaml

def blankevoort_func(lengths, params):
    # Handle both dictionary and array parameter formats
    if isinstance(params, dict):
        k, alpha, l_0, f_ref = params['k'], params['alpha'], params['l_0'], params['f_ref']
    else:
        k, alpha, l_0, f_ref = params
    # Calculate strain for all lengths
    strain = (lengths - l_0) / l_0
    
    # Calculate force using piecewise Blankevoort equation (vectorized)
    # f(x) = 0 for x < l0 (strain < 0)
    # f(x) = k(x-l0)^2/(2*alpha*l0) for l0 < x < alpha*l0 (0 < strain < alpha)
    # f(x) = k(x-l0-alpha*l0/2) for x > alpha*l0 (strain > alpha)
    quadratic_region = (strain > 0) & (strain <= alpha)
    linear_region = strain > alpha
    
    force = np.zeros_like(strain)
    force[quadratic_region] = k * (lengths[quadratic_region] - l_0)**2 / (2 * alpha * l_0)
    force[linear_region] = k * (lengths[linear_region] - l_0 - (alpha * l_0)/2)
    
    return force

class KneeModel():
    def __init__(self, data, log=True):
        self.log = log
        self.data = data

    def build_geometry(self):
        """Build the geometric structure (frames, points, joint) based on theta and config."""
        # Some constants
        femur_perp = self.data['femur_perp']
        femur_length = self.data['femur_length']
        tibia_perp = self.data['tibia_perp']
        tibia_para = self.data['tibia_para']
        application_length = self.data['application_length']
        # theta_val = self.data['theta']
        left_length = self.data['left_length']
        right_length = self.data['right_length']
        ball_radius_1 = self.data['ball_radius_1']
        ball_radius_2 = self.data['ball_radius_2']
        ball_distance_1 = self.data['ball_distance_1']
        ball_distance_2 = self.data['ball_distance_2']

        self.ball_distance_1 = ball_distance_1
        self.ball_distance_2 = ball_distance_2
        self.ball_radius_1 = ball_radius_1
        self.ball_radius_2 = ball_radius_2

        left_attatchment_height = left_length - tibia_para
        right_attatchment_height = right_length - tibia_para

        # Points of interest
        self.hip_point_world = np.array([0, femur_length, 0])
        self.knee_point_world = np.array([0, 0, 0])
        self.joint_ball_A_world = np.array([self.ball_distance_1, -self.ball_radius_1, 0])
        self.joint_ball_B_world = np.array([-self.ball_distance_2, -self.ball_radius_2, 0])
        self.lig_top_pointA_world = np.array([femur_perp, 0, 0])
        self.lig_top_pointB_world = np.array([-femur_perp, 0, 0])

        self.lig_on_tib_vis_tibia = np.array([0, -tibia_para, 0])
        self.lig_bottom_pointA_tibia = np.array([tibia_perp, -tibia_para - right_attatchment_height, 0])
        self.lig_bottom_pointB_tibia = np.array([-tibia_perp, -tibia_para - left_attatchment_height, 0])
        self.lig_bottom_pointA_flat_tibia = np.array([tibia_perp, -tibia_para, 0])
        self.lig_bottom_pointB_flat_tibia = np.array([-tibia_perp, -tibia_para, 0])
        self.application_point_tibia = np.array([0, -tibia_para-application_length, 0])

    def tibia_to_world(self, points, thetas):
        """
        Transform points from tibia coordinate system to world coordinate system.
        
        Based on the mathematical equations:
        - d(θ) and r(θ) are piecewise functions based on tan(θ) condition
        - c(θ) is the center/offset vector
        - R(θ) is the rotation matrix (2D rotation around Z-axis)
        - t(θ) is the translation vector
        - T(x, θ) = R(θ)x + t(θ) is the final transformation
        
        Args:
            points: numpy array of shape (N, 3) or (3,) representing points in tibia coordinates
            thetas: numpy array of shape (M,) or scalar representing joint angles in radians
            
        Returns:
            numpy array of shape (M, N, 3) representing points in world coordinates
            If both inputs are scalars, returns shape (3,)
        """
        # Convert inputs to numpy arrays and ensure proper shapes
        points = np.asarray(points)
        thetas = np.asarray(thetas)
        
        # Handle scalar inputs
        if points.ndim == 1:
            points = points.reshape(1, 3)
            single_point = True
        else:
            single_point = False
            
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
            single_theta = True
        else:
            single_theta = False
        
        # Extract constants
        d1 = self.ball_distance_1
        d2 = self.ball_distance_2
        r1 = self.ball_radius_1
        r2 = self.ball_radius_2
        
        # Calculate piecewise functions d(θ) and r(θ) for all thetas
        threshold = 2 * (r1 - r2) / (d1 + d2)
        conditions = np.tan(thetas) < threshold
        
        d_theta = np.where(conditions, -d1, d2)
        r_theta = np.where(conditions, r1, r2)
        
        # Calculate center/offset vector c(θ) for all thetas
        sin_theta = np.sin(thetas)
        cos_theta = np.cos(thetas)
        
        c_theta = np.column_stack([
            d_theta + r_theta * sin_theta,
            -r_theta * (1 + cos_theta),
            np.zeros_like(thetas)
        ])  # Shape: (M, 3)
        
        # Subtract center vector from all points
        # Broadcasting: (N, 3) - (M, 1, 3) -> (M, N, 3)
        points_centered = points - c_theta[:, np.newaxis, :]

        # Calculate rotation matrices R(θ) for all thetas
        # Shape: (M, 3, 3)
        # Counter-clockwise rotation from positive y-axis (downwards)
        R_theta = np.array([
            [cos_theta, -sin_theta, np.zeros_like(thetas)],
            [sin_theta, cos_theta, np.zeros_like(thetas)],
            [np.zeros_like(thetas), np.zeros_like(thetas), np.ones_like(thetas)]
        ]).transpose(2, 0, 1)
        
        # Apply transformation T(x, θ) = R(θ)(x - c(θ)) + c(θ) for all combinations
        # Broadcasting: (M, 3, 3) @ (M, N, 3) -> (M, N, 3)
        world_points = np.einsum('mij,mnj->mni', R_theta, points_centered) + c_theta[:, np.newaxis, :]
        
        # Handle scalar inputs by squeezing dimensions
        if single_theta and single_point:
            return world_points.squeeze()
        elif single_theta:
            return world_points.squeeze(0)  # Remove theta dimension
        elif single_point:
            return world_points.squeeze(1)  # Remove point dimension
        else:
            return world_points

    def calculate_moment_arm(self, force_point, force_vector, pivot_point):
        """
        Calculate the moment arm of a force around a pivot point.
        
        Args:
            force_point: numpy array of shape (3,) or (M, 3), point where force is applied
            force_vector: numpy array of shape (3,) or (M, 3), direction of force
            pivot_point: numpy array of shape (3,) or (M, 3), pivot point for moment calculation
            
        Returns:
            numpy array: moment arm magnitude(s)
        """
        # Calculate vector from pivot to force point
        r = force_point - pivot_point
        
        # Calculate moment arm as |r × force_vector| / |force_vector|
        # Cross product: r × force_vector
        cross_product = np.cross(r, force_vector)
        
        # Moment arm magnitude
        moment_arm = np.linalg.norm(cross_product, axis=-1) / np.linalg.norm(force_vector, axis=-1)
        
        return moment_arm

    def _get_pivot_points(self, thetas):
        """
        Get the pivot points (c_theta) for the given thetas.
        
        Args:
            thetas: numpy array of shape (M,) representing joint angles in radians
            
        Returns:
            numpy array of shape (M, 3): pivot points for each theta
        """
        # Extract constants
        d1 = self.ball_distance_1
        d2 = self.ball_distance_2
        r1 = self.ball_radius_1
        r2 = self.ball_radius_2
        
        # Calculate piecewise functions d(θ) and r(θ) for all thetas
        threshold = 2 * (r1 - r2) / (d1 + d2)
        conditions = np.tan(thetas) < threshold
        
        d_theta = np.where(conditions, -d1, d2)
        r_theta = np.where(conditions, r1, r2)
        
        # Calculate center/offset vector c(θ) for all thetas
        sin_theta = np.sin(thetas)
        cos_theta = np.cos(thetas)
        
        c_theta = np.column_stack([
            d_theta + r_theta * sin_theta,
            -r_theta * (1 + cos_theta),
            np.zeros_like(thetas)
        ])  # Shape: (M, 3)
        
        return c_theta

    def get_moment_arms(self, thetas):
        """
        Calculate moment arms for the current configuration.
        
        Args:
            thetas: float or array, joint angle(s) in radians
            
        Returns:
            list: List of dictionaries containing moment arm values for each theta
        """
        # Convert to numpy array
        thetas = np.asarray(thetas)
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
            single_theta = True
        else:
            single_theta = False
        
        # Get the pivot points (c_theta) for all thetas
        pivot_points = self._get_pivot_points(thetas)
        
        # Transform tibia points to world coordinates for all thetas
        lig_bottom_pointA_world = self.tibia_to_world(self.lig_bottom_pointA_tibia, thetas)
        lig_bottom_pointB_world = self.tibia_to_world(self.lig_bottom_pointB_tibia, thetas)
        application_point_world = self.tibia_to_world(self.application_point_tibia, thetas)
        
        # Calculate spring directions (vectors from top to bottom points)
        spring_a_direction = lig_bottom_pointA_world - self.lig_top_pointA_world
        spring_b_direction = lig_bottom_pointB_world - self.lig_top_pointB_world
        
        # Calculate moment arms using the pivot points
        mcl_moment_arms = self.calculate_moment_arm(lig_bottom_pointB_world, spring_b_direction, pivot_points)
        lcl_moment_arms = self.calculate_moment_arm(lig_bottom_pointA_world, spring_a_direction, pivot_points)
        
        # For application point, we need to define the force direction
        # Assuming it's along the negative x-axis (rightward)
        application_force_direction = np.array([1, 0, 0])
        if not single_theta:
            application_force_direction = np.tile(application_force_direction, (len(thetas), 1))
        application_moment_arms = self.calculate_moment_arm(application_point_world, application_force_direction, pivot_points)
        
        # Create list of dictionaries
        moment_arms_list = []
        moment_arms_dict = {
            "MCL moment arm": mcl_moment_arms,
            "LCL moment arm": lcl_moment_arms,
            "Application Moment arm": application_moment_arms
        }
        
        if single_theta:
            return {k: v[0] for k, v in moment_arms_dict.items()}
        else:
            return moment_arms_dict

    def get_ligament_lengths(self, thetas):
        """
        Calculate ligament lengths for the current configuration.
        
        Args:
            thetas: float or array, joint angle(s) in radians
            
        Returns:
            tuple: (MCL_lengths, LCL_lengths) - arrays if multiple thetas, scalars if single theta
        """
        # Convert to numpy array
        thetas = np.asarray(thetas)
        if thetas.ndim == 0:
            thetas = thetas.reshape(1)
            single_theta = True
        else:
            single_theta = False
        
        # Transform tibia points to world coordinates for all thetas
        lig_bottom_pointA_world = self.tibia_to_world(self.lig_bottom_pointA_tibia, thetas)
        lig_bottom_pointB_world = self.tibia_to_world(self.lig_bottom_pointB_tibia, thetas)
        
        # Calculate ligament lengths for all thetas
        # MCL = left = B, LCL = right = A
        mcl_lengths = np.linalg.norm(lig_bottom_pointB_world - self.lig_top_pointB_world, axis=-1)
        lcl_lengths = np.linalg.norm(lig_bottom_pointA_world - self.lig_top_pointA_world, axis=-1)
        
        if single_theta:
            return mcl_lengths[0], lcl_lengths[0]
        else:
            return mcl_lengths, lcl_lengths



    def solve_applied(self, thetas, mcl_params, lcl_params):
        # Convert thetas to numpy array if it's not already
        thetas = np.asarray(thetas)
        
        mcl_lengths, lcl_lengths = self.get_ligament_lengths(thetas)
        moment_arms_list = self.get_moment_arms(thetas)

        # Calculate forces using vectorized blankevoort function
        mcl_forces = blankevoort_func(mcl_lengths, mcl_params)
        lcl_forces = blankevoort_func(lcl_lengths, lcl_params)

        # Calculate moments for each theta
        mcl_moments = mcl_forces * np.array(moment_arms_list["MCL moment arm"])
        lcl_moments = lcl_forces * np.array(moment_arms_list["LCL moment arm"])
        application_moments = mcl_moments - lcl_moments
        applied_forces = np.array(application_moments) / np.array(moment_arms_list["Application Moment arm"])
        applied_forces = np.array(applied_forces)

        return {
            'thetas': thetas,
            'applied_forces': applied_forces,
            'mcl_lengths': mcl_lengths,
            'lcl_lengths': lcl_lengths,
            'mcl_forces': mcl_forces,
            'lcl_forces': lcl_forces,
            'moment_arms': moment_arms_list,
            'mcl_moments': mcl_moments,
            'lcl_moments': lcl_moments,
            'application_moments': application_moments
        }

    
    def plot_model(self, show_forces=False):
        vis = Visualiser2D(self.world_frame)
        vis.add_point(self.knee_point, label="Knee origin")
        vis.add_point(self.hip_point, label="Hip")

        vis.add_point(self.lig_top_pointA, label="Lateral")
        vis.add_point(self.lig_top_pointB, label="Medial")
        vis.add_point(self.lig_bottom_pointA, label=" ")
        vis.add_point(self.lig_bottom_pointB, label=" ")
        vis.add_point(self.application_point, label=" ")

        vis.add_circle(self.joint_ball_A, self.ball_radius_1)
        vis.add_circle(self.joint_ball_B, self.ball_radius_2)

        vis.add_line(self.knee_point, self.hip_point, label="Femur")
        vis.add_line(self.knee_point, self.lig_top_pointA)
        vis.add_line(self.knee_point, self.lig_top_pointB)
        vis.add_line(self.lig_top_pointA, self.lig_bottom_pointA, label="Lig", color="red")
        vis.add_line(self.lig_top_pointB, self.lig_bottom_pointB, label="Lig", color="red")

        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointA_flat, label="TibalPlataeuA")
        vis.add_line(self.lig_on_tib_vis, self.lig_bottom_pointB_flat, label="TibalPlataeuB")
        vis.add_line(self.lig_bottom_pointA, self.lig_bottom_pointA_flat, label="BoneStructure")
        vis.add_line(self.lig_bottom_pointB, self.lig_bottom_pointB_flat, label="BoneStructure")
        vis.add_line(self.lig_on_tib_vis, self.application_point, label="Tibia")

        if show_forces:
            vis.add_force(self.constraint_force, associated_body_name="Constraint", label=self.constraint_force.name)
            vis.add_force(self.lig_springA.get_force_on_point2(), associated_body_name="LigamentA", label="LigSpringForceA")
            vis.add_force(self.lig_springB.get_force_on_point2(), associated_body_name="LigamentB", label="LigSpringForceB")
            vis.add_force(self.applied_force, associated_body_name="Applied", label="AppliedForce")


        vis.render(show_values=False, equal_aspect=True)

