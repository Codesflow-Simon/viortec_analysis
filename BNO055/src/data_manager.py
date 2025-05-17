import pandas as pd
import numpy as np
import time

from abc import ABC, abstractmethod
from typing import Optional, List
from data_types import RawData, MessageType, SerialMessage
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
            
            # Add command definitions matching Arduino
            self.COMMAND_SAVE_CAL = "SAVE_CAL\n"
            self.COMMAND_PAUSE = "PAUSE\n"
            self.COMMAND_RESUME = "RESUME\n"
        except serial.SerialException as e:
            raise RuntimeError(f"{type(e)}: {e} \n when attempting to connect to '{port}'") from e
        
    def clear_buffer(self):
        """Clear the serial buffer"""
        self.serial.reset_input_buffer()
        
    def _parse_line(self, line: str) -> Optional[SerialMessage]:
        """Internal method to parse a single line of data"""
        # try:
        if line.startswith('D,'):
            # Parse CSV format
            values = [float(x) for x in line.split(',')[1:]]
            
            # Create RawData object
            data = RawData(
                tibia_rotation=quaternion_to_rotation_matrix(np.array([values[5:9]])),
                femur_rotation=quaternion_to_rotation_matrix(np.array([values[1:5]])),
                tibia_accel=np.array(values[12:15]),
                femur_accel=np.array(values[9:12]),
                timestamp=values[0],
                calibration=np.array(values[-8:]),
            )

            if data.tibia_rotation is None or data.femur_rotation is None:
                data.error = True

            return SerialMessage(
                msg_type=MessageType.DATA,
                data=data
            )

        elif line.startswith('C,'):
            values = [float(x) for x in line.split(',')[1:]]
            return SerialMessage(
                msg_type=MessageType.CALIBRATION,
                data=RawData(calibration=np.array(values))
            )

        elif line.startswith('I,'):
            return SerialMessage(
                msg_type=MessageType.INFO,
                message=line[2:]
            )

        elif line.startswith('E,'):
            raise RuntimeError(f"Error: {line[2:]}")
                
        else:
            raise RuntimeError(f"Unknown message type: {line}")
        
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
    
    def sample(self) -> Optional[SerialMessage]:
        """Get the most recent complete sample from the hardware"""
        try:
            # Read all available data
            while self.serial.in_waiting > 0:
                # Read until the last complete line
                data = self.serial.read(self.serial.in_waiting).decode('utf-8', errors='replace')
                
                # Split into lines and filter empty lines
                lines = [line.strip() for line in data.splitlines() if line.strip()]
                
                # If we got any complete lines
                if lines:
                    # Take only the last complete line that ends with calibration values
                    for line in reversed(lines):
                        # Quick check for complete data line (should have all calibration values)
                        parts = line.split(',')
                        if line.startswith('D,') and len(parts) == 24:  # 1 prefix + 23 values
                            try:
                                sample = self._parse_line(line)
                                if sample:
                                    return sample
                            except (ValueError, IndexError):
                                continue
                        elif line.startswith(('C,', 'I,', 'E,')):
                            try:
                                sample = self._parse_line(line)
                                if sample:
                                    return sample
                            except (ValueError, IndexError):
                                continue
                else:
                    print(f"No complete lines found in {data}")
                            
                # Small delay to prevent busy-waiting
                time.sleep(0.001)
                    
        except serial.SerialException as e:
            print(f"Error reading from serial port: {e}")
            raise e
        except UnicodeDecodeError:
            print("Warning: Invalid UTF-8 sequence received, skipping...")
            return None

        print("No data available")
        time.sleep(0.1)
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
            print(f"Connection to '{self.port}' successful")
            return True
            
        except (serial.SerialException, ValueError) as e:
            print(f"Failed to connect to '{self.port}': {e}")
            return False
            
    def __del__(self):
        """Clean up serial connection when object is destroyed"""
        if hasattr(self, 'serial'):
            self.serial.close()

    def save_calibration(self):
        """Send command to save current calibration to EEPROM"""
        try:
            self.serial.write(self.COMMAND_SAVE_CAL.encode('utf-8'))
            self.serial.flush()
            # Wait for confirmation message
            response = self.sample()
            if response and response.msg_type == MessageType.INFO:
                print(f"Calibration save response: {response.message}")
                return True
            return False
        except serial.SerialException as e:
            print(f"Error sending save calibration command: {e}")
            return False

    def pause_streaming(self):
        """Send command to pause data streaming"""
        try:
            self.serial.write(self.COMMAND_PAUSE.encode('utf-8'))
            self.serial.flush()
            # Wait for confirmation message
            response = self.sample()
            if response and response.msg_type == MessageType.INFO:
                print(f"Pause response: {response.message}")
                return True
            return False
        except serial.SerialException as e:
            print(f"Error sending pause command: {e}")
            return False

    def resume_streaming(self):
        """Send command to resume data streaming"""
        try:
            self.serial.write(self.COMMAND_RESUME.encode('utf-8'))
            self.serial.flush()
            # Wait for confirmation message
            response = self.sample()
            if response and response.msg_type == MessageType.INFO:
                print(f"Resume response: {response.message}")
                return True
            return False
        except serial.SerialException as e:
            print(f"Error sending resume command: {e}")
            return False

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

