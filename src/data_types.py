import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class MessageType(Enum):
    DATA = auto()
    CALIBRATION = auto()
    INFO = auto()
    ERROR = auto()

@dataclass
class RawData:
    """Contains raw IMU data for a single timepoint"""
    tibia_rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    femur_rotation: Optional[np.ndarray] = None  # 3x3 rotation matrix
    tibia_accel: Optional[np.ndarray] = None    # 3D acceleration vector
    femur_accel: Optional[np.ndarray] = None    # 3D acceleration vector
    timestamp: Optional[float] = None           # Time in milliseconds
    calibration: Optional[list[int]] = None    # Calibration status
    error: Optional[bool] = None                # Error message
    
    def get_tibia_in_femur_frame(self) -> Optional[np.ndarray]:
        """
        Calculate the orientation of the tibia in the femur's reference frame
        Returns:
            3x3 rotation matrix representing tibia orientation relative to femur
            Returns None if data not available
        """
        if self.tibia_rotation is None or self.femur_rotation is None:
            return None
            
        return np.matmul(self.femur_rotation.T, self.tibia_rotation)

    def __invert__(self) -> Optional['RawData']:
        """Return a RawData object with inverted rotations"""
        if self.tibia_rotation is None or self.femur_rotation is None:
            return None
            
        return RawData(
            tibia_rotation=self.tibia_rotation.T,
            femur_rotation=self.femur_rotation.T,
            tibia_accel=self.tibia_accel,
            femur_accel=self.femur_accel,
            timestamp=float('nan'),
            calibration=self.calibration
        )

    def __matmul__(self, other: 'RawData') -> Optional['RawData']:
        """Compose two RawData objects by composing their rotation matrices"""
        if (self.tibia_rotation is None or self.femur_rotation is None or
            other.tibia_rotation is None or other.femur_rotation is None):
            return None
            
        return RawData(
            tibia_rotation=np.matmul(self.tibia_rotation, other.tibia_rotation),
            femur_rotation=np.matmul(self.femur_rotation, other.femur_rotation),
            tibia_accel=np.matmul(self.tibia_rotation, other.tibia_accel) if other.tibia_accel is not None else None,
            femur_accel=np.matmul(self.femur_rotation, other.femur_accel) if other.femur_accel is not None else None,
            timestamp=float('nan'),
            calibration=np.minimum(self.calibration, other.calibration) if self.calibration is not None and other.calibration is not None else None
        )

@dataclass
class SerialMessage:
    """Container for all types of serial messages"""
    msg_type: MessageType
    message: Optional[str] = None
    data: Optional[RawData] = None

    def __str__(self):
        if self.msg_type == MessageType.DATA or self.msg_type == MessageType.CALIBRATION:
            return f"SerialMessage(type={self.msg_type}, data={self.data})"
        else:
            return f"SerialMessage(type={self.msg_type}, message={self.message})"

@dataclass
class ProcessedData:
    """Contains processed joint angles calculated from IMU data"""
    flexion_angle: float  # Positive indicates flexion, negative indicates extension (degrees)
    varus_angle: float   # Positive indicates varus, negative indicates valgus (degrees)
    internal_angle: float  # Positive indicates internal rotation, negative indicates external rotation (degrees)
    timestamp: float     # Time in seconds
    raw_data: RawData    # Reference to original raw data



