import numpy as np

def cartesian_to_spherical(x: float, y: float, z: float) -> tuple[float, float, float]:
    """Convert Cartesian coordinates to spherical coordinates (r, theta, phi)
    
    Args:
        x: X coordinate
        y: Y coordinate 
        z: Z coordinate
        
    Returns:
        Tuple of (r, theta, phi) where:
            r: radius (distance from origin)
            theta: azimuthal angle in x-y plane from x-axis (in radians) 
            phi: polar angle from z-axis (in radians)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z/r)
    return r, theta, phi

def spherical_to_cartesian(r: float, theta: float, phi: float) -> tuple[float, float, float]:
    """Convert spherical coordinates to Cartesian coordinates
    
    Args:
        r: radius (distance from origin)
        theta: azimuthal angle in x-y plane from x-axis (in radians)
        phi: polar angle from z-axis (in radians)
        
    Returns:
        Tuple of (x, y, z) Cartesian coordinates
    """
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta) 
    z = r * np.cos(phi)
    return x, y, z

def check_unit_vector(vector: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if a vector has unit length (magnitude = 1)
    
    Args:
        vector: NumPy array representing the vector
        tolerance: Maximum allowed deviation from 1.0
        
    Returns:
        True if vector magnitude is within tolerance of 1.0, False otherwise
    """
    magnitude = np.sqrt(np.sum(vector**2))
    return abs(magnitude - 1.0) < tolerance

def euler_to_rotation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix using XYZ convention
    
    Args:
        x: Rotation around X axis in degrees
        y: Rotation around Y axis in degrees
        z: Rotation around Z axis in degrees
        
    Returns:
        3x3 rotation matrix
    """
    # Convert to radians if input is in degrees
    x = np.deg2rad(x)
    y = np.deg2rad(y)
    z = np.deg2rad(z)
    
    # Rotation matrix around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(x), -np.sin(x)],
        [0, np.sin(x), np.cos(x)]
    ])
    
    # Rotation matrix around Y axis
    Ry = np.array([
        [np.cos(y), 0, np.sin(y)],
        [0, 1, 0],
        [-np.sin(y), 0, np.cos(y)]
    ])
    
    # Rotation matrix around Z axis
    Rz = np.array([
        [np.cos(z), -np.sin(z), 0],
        [np.sin(z), np.cos(z), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix (XYZ convention)
    R = Rz @ Ry @ Rx
    return R

def rotation_matrix_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to Euler angles using XYZ convention
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        Tuple of (x, y, z) angles in degrees
    """
    # Check if we're in gimbal lock
    if abs(R[2,0]) >= 1.0:
        # Gimbal lock case
        z = 0.0  # Set arbitrarily
        if R[2,0] < 0:
            y = np.pi/2
            x = z + np.arctan2(R[0,1], R[0,2])
        else:
            y = -np.pi/2
            x = -z + np.arctan2(-R[0,1], -R[0,2])
    else:
        # Regular case
        y = -np.arcsin(R[2,0])
        x = np.arctan2(R[2,1]/np.cos(y), R[2,2]/np.cos(y))
        z = np.arctan2(R[1,0]/np.cos(y), R[0,0]/np.cos(y))
    
    # Convert to degrees
    return np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)

def is_rotation_matrix(R: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if a matrix is a valid rotation matrix
    
    Args:
        R: Matrix to check
        tolerance: Numerical tolerance for floating point comparisons
        
    Returns:
        True if R is a valid rotation matrix, False otherwise
    """
    if R.shape != (3, 3):
        return False
    
    # Check if determinant is 1
    if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
        return False
    
    # Check if transpose is inverse (orthogonal)
    identity = np.eye(3)
    if not np.allclose(R @ R.T, identity, atol=tolerance):
        return False
    
    return True 

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix
    
    Args:
        q: Quaternion as numpy array [w,x,y,z]
        
    Returns:
        3x3 rotation matrix
    """
    w, x, y, z = q
    
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])