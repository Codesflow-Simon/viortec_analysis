import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class RawData:
    """Contains raw IMU data for a single timepoint with tibia and femur rotation matrices"""
    tibia_rotation: np.ndarray  # 3x3 rotation matrix
    femur_rotation: np.ndarray  # 3x3 rotation matrix 
    timestamp: float  # Time in seconds

    calibration: np.ndarray  # Vector [gf, mg, gt, mt] containing calibration readings

    def __post_init__(self):
        """Validate rotation matrix shapes"""
        if self.tibia_rotation.shape != (3,3) or self.femur_rotation.shape != (3,3):
            raise ValueError("Rotation matrices must be 3x3")

    def get_tibia_in_femur_frame(self) -> np.ndarray:
        """
        Calculate the orientation of the tibia in the femur's reference frame
        Returns:
            3x3 rotation matrix representing tibia orientation relative to femur
        """
        # To get tibia orientation in femur frame, multiply inverse of femur rotation
        # by tibia rotation: R_femur^(-1) * R_tibia
        return np.matmul(self.femur_rotation.T, self.tibia_rotation)

    def __invert__(self) -> 'RawData':
        """
        Return a RawData object with inverted rotations
        Returns:
            New RawData with inverted rotation matrices
        """
        return RawData(
            tibia_rotation=self.tibia_rotation.T,
            femur_rotation=self.femur_rotation.T,
            timestamp=float('nan'),
            calibration=self.calibration
        )

    def __matmul__(self, other: 'RawData') -> 'RawData':
        """
        Compose two RawData objects by composing their rotation matrices
        Args:
            other: RawData to compose with
        Returns:
            New RawData with composed rotation matrices
        """
        return RawData(
            tibia_rotation=np.matmul(self.tibia_rotation, other.tibia_rotation),
            femur_rotation=np.matmul(self.femur_rotation, other.femur_rotation), 
            timestamp=float('nan'),
            calibration=self.calibration
        )

    def __str__(self):
        return f"RawData(timestamp={self.timestamp}, tibia_rotation={self.tibia_rotation}, femur_rotation={self.femur_rotation})"

@dataclass
class ProcessedData:
    """Contains processed joint angles calculated from IMU data"""
    flexion_angle: float  # Positive indicates flexion, negative indicates extension (degrees)
    varus_angle: float   # Positive indicates varus, negative indicates valgus (degrees)
    internal_angle: float  # Positive indicates internal rotation, negative indicates external rotation (degrees)
    timestamp: float     # Time in seconds
    raw_data: RawData    # Reference to original raw data



