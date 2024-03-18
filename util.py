import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

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

def euler_multiply(euler1, euler0):
    """
    Multiply two sets of Euler angles.
    
    :param euler1: First set of Euler angles
    :param euler0: Second set of Euler angles
    :return: Euler angles that represent the combined rotation
    """
    R0 = euler_to_rotation_matrix(euler0[0], euler0[1], euler0[2])
    R1 = euler_to_rotation_matrix(euler1[0], euler1[1], euler1[2])
    R = np.dot(R1, R0)
    return rotation_matrix_to_euler(R)

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
        femur_matrix = euler_to_rotation_matrix(row['femur_x'], row['femur_y'], row['femur_z'])
        tibia_matrix = euler_to_rotation_matrix(row['tibia_x'], row['tibia_y'], row['tibia_z'])
        relative = np.linalg.inv(femur_matrix) * tibia_matrix;
        test_vector = np.array([0,0,-1]).transpose()
        vector = (np.dot(relative,test_vector))
        if (np.linalg.norm(vector) == 0): continue
        out.append(vector)
    return out

def vectors_spherical(vectors):
    out = []
    for vector in vectors:
        print(vector)
        azimuth, elevation = spherical_transform(vector[0], vector[1], vector[2])
        out.append((azimuth, elevation))
    return out

def rotate_to_line_matrix(theta):
    return np.array([[math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]])

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    out = np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
    return out

def quaternion_conjugate(quaternion):
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z], dtype=np.float64)
                    
def quaternion_inverse(quaternion):
    return quaternion_conjugate(quaternion) / np.dot(quaternion, quaternion)

def quaternion_to_euler(quaternion):
    w, x, y, z = quaternion
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = 1 if t2 > 1 else t2
    t2 = -1 if t2 < -1 else t2
    pitch_y = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    yaw_z = (yaw_z - math.pi/2) % (2*math.pi) - math.pi
    return np.array([roll_x, pitch_y, yaw_z])

def quat_to_euler(quaternion_series):
    # Assume columns are w,x,y,z
    quaternion_series = np.array(quaternion_series)
    out = []
    for row in quaternion_series:
        out.append(quaternion_to_euler(row))
    return pd.DataFrame(out, columns=['roll', 'pitch', 'yaw'])

def plot_euler(camera_df, imu_df, camera_time=None, imu_time=None):
    # Assume columns are w,x,y,z
    euler_camera = quat_to_euler(camera_df)
    euler_imu = quat_to_euler(imu_df)

    print(euler_camera)

    # Plot data overlayed on 2x2
    fig, axs = plt.subplots(2, 2)
    if camera_time is not None and imu_time is not None:
        axs[0, 0].plot(camera_time, euler_camera['roll'], label='Camera')
        axs[0, 0].plot(imu_time, euler_imu['roll'], label='IMU')
        axs[0, 0].set_title('Roll')

        axs[0, 1].plot(camera_time, euler_camera['pitch'], label='Camera')
        axs[0, 1].plot(imu_time, euler_imu['pitch'], label='IMU')
        axs[0, 1].set_title('Pitch')

        axs[1, 0].plot(camera_time, euler_camera['yaw'], label='Camera')
        axs[1, 0].plot(imu_time, euler_imu['yaw'], label='IMU')   
        axs[1, 0].set_title('Yaw')
    else:
        axs[0, 0].plot(euler_camera['roll'], label='Camera')
        axs[0, 0].plot(euler_imu['roll'], label='IMU')
        axs[0, 0].set_title('Roll')

        axs[0, 1].plot(euler_camera['pitch'], label='Camera')
        axs[0, 1].plot(euler_imu['pitch'], label='IMU')
        axs[0, 1].set_title('Pitch')

        axs[1, 0].plot(euler_camera['yaw'], label='Camera')
        axs[1, 0].plot(euler_imu['yaw'], label='IMU')   
        axs[1, 0].set_title('Yaw')
    

def plot_quaternion(camera_df, imu_df):
    # Plot data overlayed on 2x2
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(camera_df['time'], camera_df['W_rot'], label='Camera')
    axs[0, 0].plot(imu_df['time'], imu_df['femur_w'], label='IMU')
    axs[0, 0].set_title('W_rot')
    axs[0, 0].legend()

    axs[0, 1].plot(camera_df['time'], camera_df['X_rot'], label='Camera')
    axs[0, 1].plot(imu_df['time'], imu_df['femur_x'], label='IMU')
    axs[0, 1].set_title('X_rot')
    axs[0, 1].legend()

    axs[1, 0].plot(camera_df['time'], camera_df['Y_rot'], label='Camera')
    axs[1, 0].plot(imu_df['time'], imu_df['femur_y'], label='IMU')
    axs[1, 0].set_title('Y_rot')
    axs[1, 0].legend()

    axs[1, 1].plot(camera_df['time'], camera_df['Z_rot'], label='Camera')
    axs[1, 1].plot(imu_df['time'], imu_df['femur_z'], label='IMU') 
    axs[1, 1].set_title('Z_rot')
    axs[1, 1].legend()

def quaternion_visualisation(quaternion, ax, colour='b'):
    pitch, roll, yaw = quat_to_euler([quaternion]).values[0]
    x,y,z = np.transpose(euler_to_rotation_matrix(pitch, roll, yaw))
    ax.quiver(0,0,0,x[0],x[1],x[2], color='g')
    ax.quiver(0,0,0,y[0],y[1],y[2], color=colour)
    ax.quiver(0,0,0,z[0],z[1],z[2], color=colour)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    
