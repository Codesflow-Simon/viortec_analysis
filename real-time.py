from serial import Serial
import pandas as pd
import time
import msvcrt
from util import *
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# Open serial port
ser = Serial('COM3', 115200)

# Parse CSV header
header = ser.readline().decode('utf-8').strip().split(',')
print(header)

axis_data = pd.DataFrame(columns=header)

print("Sensor calibration, press a key to continue...")

# Pause until a key is pressed
while True:
    line = ser.readline().decode('utf-8').strip().split(',')
    print(line[-4:])
    if msvcrt.kbhit():
        ser.readline()
        break

# Wait until key is released
while msvcrt.kbhit():
    msvcrt.getch()

print("Axis calibration, press a key to continue...")

# Visualisation setup
fig, ax = plt.subplots()

# Read data loop, including buffer handling
partial_line = ""
while True:
    # Wait until data is available
    while ser.in_waiting == 0:
        pass

    # Read whole buffer
    buffer = ser.read(ser.in_waiting).decode('utf-8')
    lines = (partial_line + buffer).split('\n')
    lines = [line for line in lines if line != '']

    # If the last line is not complete, save it for the next iteration
    # Experimentally, partial lines are very infrequent
    if not buffer.endswith('\n'):
        partial_line = lines.pop()
    else:
        partial_line = ""

    for line in lines:
        line = line.strip().split(',')

        # Parse line and convert to numerical values
        data = dict(zip(header, line))
        data = {key: pd.to_numeric(value) for key, value in data.items()}

        # Append to dataframe
        axis_data.loc[len(axis_data)] = data

    if len(axis_data) > 10:
        # Update the scatter plot
        ax.clear()
        vectors = df_to_vectors(axis_data)
        # print(vectors)
        spheric = vectors_spherical(vectors)
        spheric_T = np.array(spheric).transpose()
        ax.scatter(spheric_T[0], spheric_T[1])
        plt.pause(0.01)  # Pause to update the plot

    # Check if user pressed a key
    if msvcrt.kbhit():
        break

vectors = df_to_vectors(axis_data)
spheric = vectors_spherical(vectors)

# Plot spherical coordinates
spheric_T = np.array(spheric).transpose()
plt.scatter(spheric_T[0], spheric_T[1])
plt.show()

