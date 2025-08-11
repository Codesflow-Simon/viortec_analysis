import numpy as np
from .low_pass import LowPassFilter


class DerivativeCalculator:
    """
    A class to calculate first and second derivatives from noisy data using low pass filtering.
    
    This implementation uses finite difference methods combined with low pass filtering
    to reduce noise in the derivative calculations.
    """
    
    def __init__(self, cutoff_frequency=10.0, sample_rate=100.0):
        """
        Initialize the derivative calculator.
        
        Args:
            cutoff_frequency (float): Cutoff frequency for the low pass filter (Hz)
            sample_rate (float): Sampling rate of the data (Hz)
        """
        self.cutoff_frequency = cutoff_frequency
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        
        # Initialize low pass filters for different derivative orders
        self.first_deriv_filter = LowPassFilter(cutoff_frequency, sample_rate)
        self.second_deriv_filter = LowPassFilter(cutoff_frequency, sample_rate)
        
    def calculate_first_derivative(self, data, method='central'):
        """
        Calculate the first derivative using finite differences and low pass filtering.
        
        Args:
            data (np.ndarray): Input data array
            method (str): Differentiation method ('forward', 'backward', 'central')
            
        Returns:
            np.ndarray: First derivative array
        """
        if len(data) < 2:
            raise ValueError("Data must have at least 2 points for derivative calculation")
            
        # Calculate raw derivatives using finite differences
        if method == 'forward':
            raw_deriv = np.diff(data) / self.dt
            # Pad with the last value to maintain array length
            raw_deriv = np.append(raw_deriv, raw_deriv[-1])
        elif method == 'backward':
            raw_deriv = np.diff(data) / self.dt
            # Pad with the first value to maintain array length
            raw_deriv = np.insert(raw_deriv, 0, raw_deriv[0])
        elif method == 'central':
            # Central difference for interior points
            central_deriv = np.diff(data[1:]) / self.dt
            # Forward difference for first point
            first_deriv = (data[1] - data[0]) / self.dt
            # Backward difference for last point
            last_deriv = (data[-1] - data[-2]) / self.dt
            raw_deriv = np.concatenate([[first_deriv], central_deriv, [last_deriv]])
        else:
            raise ValueError("Method must be 'forward', 'backward', or 'central'")
        
        # Apply low pass filter to reduce noise
        filtered_deriv = np.zeros_like(raw_deriv)
        for i, value in enumerate(raw_deriv):
            filtered_deriv[i] = self.first_deriv_filter.filter(value)
            
        return filtered_deriv
    
    def calculate_second_derivative(self, data, method='central'):
        """
        Calculate the second derivative using finite differences and low pass filtering.
        
        Args:
            data (np.ndarray): Input data array
            method (str): Differentiation method ('forward', 'backward', 'central')
            
        Returns:
            np.ndarray: Second derivative array
        """
        if len(data) < 3:
            raise ValueError("Data must have at least 3 points for second derivative calculation")
            
        # Calculate raw second derivatives using finite differences
        if method == 'forward':
            # Second forward difference
            raw_deriv = (data[2:] - 2*data[1:-1] + data[:-2]) / (self.dt**2)
            # Pad to maintain array length
            raw_deriv = np.append(raw_deriv, raw_deriv[-1])
            raw_deriv = np.insert(raw_deriv, 0, raw_deriv[0])
        elif method == 'backward':
            # Second backward difference
            raw_deriv = (data[2:] - 2*data[1:-1] + data[:-2]) / (self.dt**2)
            # Pad to maintain array length
            raw_deriv = np.append(raw_deriv, raw_deriv[-1])
            raw_deriv = np.insert(raw_deriv, 0, raw_deriv[0])
        elif method == 'central':
            # Central difference for interior points
            central_deriv = (data[2:] - 2*data[1:-1] + data[:-2]) / (self.dt**2)
            # Handle boundary points
            first_deriv = (data[2] - 2*data[1] + data[0]) / (self.dt**2)
            last_deriv = (data[-1] - 2*data[-2] + data[-3]) / (self.dt**2)
            raw_deriv = np.concatenate([[first_deriv], central_deriv, [last_deriv]])
        else:
            raise ValueError("Method must be 'forward', 'backward', or 'central'")
        
        # Apply low pass filter to reduce noise
        filtered_deriv = np.zeros_like(raw_deriv)
        for i, value in enumerate(raw_deriv):
            filtered_deriv[i] = self.second_deriv_filter.filter(value)
            
        return filtered_deriv
    
    def calculate_derivatives(self, data, method='central'):
        """
        Calculate both first and second derivatives in one call.
        
        Args:
            data (np.ndarray): Input data array
            method (str): Differentiation method ('forward', 'backward', 'central')
            
        Returns:
            tuple: (first_derivative, second_derivative)
        """
        first_deriv = self.calculate_first_derivative(data, method)
        second_deriv = self.calculate_second_derivative(data, method)
        
        return first_deriv, second_deriv
    
    def reset_filters(self):
        """Reset the low pass filters to their initial state."""
        self.first_deriv_filter = LowPassFilter(self.cutoff_frequency, self.sample_rate)
        self.second_deriv_filter = LowPassFilter(self.cutoff_frequency, self.sample_rate)


def calculate_derivatives_with_filter(data, time=None, cutoff_frequency=10.0, 
                                    sample_rate=100.0, method='central'):
    """
    Convenience function to calculate derivatives with automatic time array generation.
    
    Args:
        data (np.ndarray): Input data array
        time (np.ndarray, optional): Time array. If None, will be generated based on sample_rate
        cutoff_frequency (float): Cutoff frequency for the low pass filter (Hz)
        sample_rate (float): Sampling rate of the data (Hz)
        method (str): Differentiation method ('forward', 'backward', 'central')
        
    Returns:
        tuple: (time_array, first_derivative, second_derivative)
    """
    if time is None:
        time = np.arange(len(data)) / sample_rate
    
    calculator = DerivativeCalculator(cutoff_frequency, sample_rate)
    first_deriv, second_deriv = calculator.calculate_derivatives(data, method)
    
    return time, first_deriv, second_deriv


def smooth_derivatives(data, time=None, cutoff_frequency=10.0, sample_rate=100.0, 
                      method='central', window_size=5):
    """
    Calculate derivatives with additional smoothing using a moving average window.
    
    Args:
        data (np.ndarray): Input data array
        time (np.ndarray, optional): Time array. If None, will be generated based on sample_rate
        cutoff_frequency (float): Cutoff frequency for the low pass filter (Hz)
        sample_rate (float): Sampling rate of the data (Hz)
        method (str): Differentiation method ('forward', 'backward', 'central')
        window_size (int): Size of the moving average window for additional smoothing
        
    Returns:
        tuple: (time_array, first_derivative, second_derivative)
    """
    if time is None:
        time = np.arange(len(data)) / sample_rate
    
    # Calculate derivatives
    calculator = DerivativeCalculator(cutoff_frequency, sample_rate)
    first_deriv, second_deriv = calculator.calculate_derivatives(data, method)
    
    # Apply additional smoothing with moving average
    if window_size > 1:
        kernel = np.ones(window_size) / window_size
        first_deriv = np.convolve(first_deriv, kernel, mode='same')
        second_deriv = np.convolve(second_deriv, kernel, mode='same')
    
    return time, first_deriv, second_deriv
