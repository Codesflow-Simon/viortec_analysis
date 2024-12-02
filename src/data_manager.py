import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, List
from data_types import RawData
from math_utils import euler_to_rotation_matrix, quaternion_to_rotation_matrix
import serial



class DataSource(ABC):
    """Abstract base class representing a source of IMU data"""
    
    @abstractmethod
    def sample(self) -> Optional[RawData]:
        """Get the next sample from the data source
        
        Returns:
            RawData containing the next sample, or None if no more samples available
        """
        raise NotImplementedError("Subclasses must implement the sample method")

class DataStream(DataSource):
    """Implementation of DataSource for live hardware data collection"""
    
    def __init__(self, port: str):
        """Initialize connection to hardware"""
        try:
            self.port = port
            self.serial = serial.Serial(port, baudrate=115200, timeout=1)
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
            self.partial_line = ""  # Store incomplete lines between calls
            self.test_connection()
        except serial.SerialException as e:
            raise RuntimeError(f"{type(e)}: {e} \n when attempting to connect to '{port}'") from e
        
    def _parse_line(self, line: str) -> Optional[RawData]:
        """Internal method to parse a single line of data
        
        Args:
            line: CSV formatted line of sensor data
            Format: time,femur_w,femur_x,femur_y,femur_z,tibia_w,tibia_x,tibia_y,tibia_z,
                    femur_acc_x,femur_acc_y,femur_acc_z,tibia_acc_x,tibia_acc_y,tibia_acc_z,
                    gf,mf,gt,mt
            
        Returns:
            RawData object if parsing successful, None if parsing failed
        """
        try:
            # Parse CSV format
            values = [float(x) for x in line.split(',')]
            
            # Extract quaternions
            femur_quat = np.array([values[1], values[2], values[3], values[4]])  # w,x,y,z
            tibia_quat = np.array([values[5], values[6], values[7], values[8]])  # w,x,y,z
            
            # Convert quaternions to rotation matrices
            # Check if quaternions are all zeros and return identity matrix if so
            def get_rotation_safe(quat):
                return np.eye(3) if np.allclose(quat, 0) else quaternion_to_rotation_matrix(quat)
                
            femur_rotation = get_rotation_safe(femur_quat)
            tibia_rotation = get_rotation_safe(tibia_quat)

            # Create calibration vector from last 4 values
            calibration = np.array([
                values[-4],  # gf (gyro femur)
                values[-3],  # mf (mag femur)
                values[-2],  # gt (gyro tibia)
                values[-1]   # mt (mag tibia)
            ])
            
            return RawData(
                tibia_rotation=tibia_rotation,
                femur_rotation=femur_rotation,
                timestamp=values[0],
                calibration=calibration
            )
            
        except (ValueError, IndexError) as e:
            print(f"Error parsing line '{line}': {e}")
            return None
            
    def _read_serial(self) -> str:
        """Internal method to read available data from serial port
        
        Returns:
            String containing all available data from serial port
        """
        try:
            return self.serial.read(self.serial.in_waiting).decode('utf-8', errors='replace')
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            return ""
    
    def sample(self) -> Optional[RawData]:
        """Get the next live sample from the hardware
        
        Returns:
            RawData containing the latest sensor readings
            None if there was a timeout or error reading data
        """
        try:
            # Flush buffer to get most recent data
            self.serial.reset_input_buffer()
            
            # Read until we get a complete line
            line = self.serial.readline().decode('utf-8').strip()
            if not line:
                return None
            
            return self._parse_line(line)
            
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            return None
        
    def sample_buffer(self) -> List[RawData]:
        """Get all complete samples currently in the buffer
        
        Returns:
            List of RawData samples from the buffer
            Empty list if no complete samples available
        """
        # Read all available data
        data = self._read_serial()
        
        # Combine with any partial line from previous call
        data = self.partial_line + data
        
        # Split into lines
        lines = data.split('\n')
        
        # Save last line if incomplete (no newline)
        self.partial_line = lines[-1] if not data.endswith('\n') else ""
        
        # Process complete lines
        samples = []
        for line in lines[:-1]:  # Skip last line (either incomplete or empty)
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            sample = self._parse_line(line)
            if sample:
                samples.append(sample)
        
        return samples

    def test_connection(self) -> bool:
        """Test connection to hardware"""
        try:
            # Try to read one line to verify connection
            line = self.serial.readline()
            decoded_line = line.decode('utf-8').strip()
            print(f"Connection to '{self.port}' successful: '{decoded_line}'")
            return True
            
        except (serial.SerialException, ValueError) as e:
            print(f"Failed to connect to '{self.port}': {e}")
            return False
            
    def __del__(self):
        """Clean up serial connection when object is destroyed"""
        if hasattr(self, 'serial'):
            self.serial.close()

class DataSet(DataSource):
    """Implementation of DataSource for reading from saved data files"""
    
    def __init__(self, filepath: str):
        """Initialize dataset from file
        
        Args:
            filepath: Path to the data file to load
        """
        import pandas as pd
        import numpy as np
        from math_utils import euler_to_rotation_matrix
        
        # Read CSV file
        self.data = pd.read_csv(filepath)
        self.current_index = 0
        
    def sample(self) -> Optional[RawData]:
        """Get the next sample from the loaded dataset
        
        Returns:
            RawData containing the next sample from the file,
            or None if end of dataset reached
        """
        # Check if we've reached the end of the dataset
        if self.current_index >= len(self.data):
            return None
        
        # Get current row
        row = self.data.iloc[self.current_index]
        
        # Convert Euler angles to rotation matrices
        femur_rotation = euler_to_rotation_matrix(
            row['femur_x'], 
            row['femur_y'], 
            row['femur_z']
        )
        
        tibia_rotation = euler_to_rotation_matrix(
            row['tibia_x'],
            row['tibia_y'], 
            row['tibia_z']
        )
        
        # Create calibration vector
        calibration = np.array([
            row['gf'],
            row['mg'],
            row['gt'],
            row['mt'] 
        ])
        
        # Increment counter
        self.current_index += 1
        
        # Return RawData sample
        return RawData(
            tibia_rotation=tibia_rotation,
            femur_rotation=femur_rotation,
            timestamp=row['time'],
            calibration=calibration
        )
    def sample_buffer(self) -> List[RawData]:
        return [self.sample()]