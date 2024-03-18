import math
import numpy as np

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a 3x3 rotation matrix.
    
    :param roll: Rotation around the X-axis (in radians)
    :param pitch: Rotation around the Y-axis (in radians)
    :param yaw: Rotation around the Z-axis (in radians)
    :return: 3x3 rotation matrix
    """
    # Define the rotation matrix for roll, pitch, and yaw
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]])
    
    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]])
    
    R_z = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                    [math.sin(yaw), math.cos(yaw), 0],
                    [0, 0, 1]])
    
    # Combine the individual rotation matrices in the specified order (ZYX)
    rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
    
    return rotation_matrix

def rotation_matrix_to_euler_angles(R):
  """Converts a rotation matrix to Euler angles.

  Args:
    R: A 3x3 rotation matrix.

  Returns:
    A tuple of three Euler angles in radians.
  """

  # Calculate the pitch angle.
  pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))

  # Calculate the roll angle.
  roll = np.arctan2(R[2, 1], R[2, 2])

  # Calculate the yaw angle.
  yaw = np.arctan2(R[1, 0], R[0, 0])

  return pitch, roll, yaw


## Transform to spherical coordinates ##

def spherical_transform(x,y,z):
    azimuth = np.arctan2(y,x)
    horizontal = np.square(x) + np.square(x)
    elevation = np.arctan2(z,horizontal)
    return azimuth, elevation

def inverse_sphertical_transform(azimuth, elevation):
    x = np.cos(elevation) * np.cos(azimuth)
    y = np.cos(elevation) * np.sin(azimuth)
    z = np.sin(elevation)
    return x,y,z

def df_to_vectors(df):
    out = []
    for index, row in df.iterrows():
        femur_matrix = euler_to_rotation_matrix(row['femur_x'], row['femur_y'], -row['femur_z'])
        tibia_matrix = euler_to_rotation_matrix(row['tibia_x'], row['tibia_y'], -row['tibia_z'])
        relative = np.linalg.inv(femur_matrix) * tibia_matrix;
        test_vector = np.array([1,0,0]).transpose()
        vector = (relative*test_vector)[:,0]
        if (np.linalg.norm(vector) == 0): continue
        out.append((tibia_matrix*test_vector)[:,0])
    return out