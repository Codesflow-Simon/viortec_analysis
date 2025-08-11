import numpy as np

class LowPassFilter:
    def __init__(self, cutoff_frequency, sample_rate):
        self.cutoff_frequency = cutoff_frequency
        self.sample_rate = sample_rate
        self.alpha = 2 * np.pi * cutoff_frequency / sample_rate
        self.y_prev = 0

    def filter(self, x):
        y = self.alpha * x + (1 - self.alpha) * self.y_prev
        self.y_prev = y
        return y