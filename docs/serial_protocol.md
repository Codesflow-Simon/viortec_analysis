# Serial Communication Protocol

## Overview
The system uses a serial connection at 115200 baud rate to communicate between the Arduino (sensors) and the host computer. Messages are sent as ASCII text with different prefixes to identify message types.

## Message Types
All messages start with a prefix followed by a comma:

- `D,` - Data messages (sensor readings)
- `C,` - Calibration status messages
- `I,` - Information/status messages
- `E,` - Error messages

## Message Formats

### Data Messages (`D,`)
Format: `D,timestamp,fw,fx,fy,fz,tw,tx,ty,tz,fax,fay,faz,tax,tay,taz,fs,fg,fa,fm,ts,tg,ta,tm`

Where:
- `timestamp`: milliseconds since start
- `fw,fx,fy,fz`: femur quaternion (w,x,y,z)
- `tw,tx,ty,tz`: tibia quaternion (w,x,y,z)
- `fax,fay,faz`: femur accelerometer (x,y,z)
- `tax,tay,taz`: tibia accelerometer (x,y,z)
- `fs,fg,fa,fm`: femur calibration status (system, gyro, accel, mag)
- `ts,tg,ta,tm`: tibia calibration status (system, gyro, accel, mag)

Example: `D,1000,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,9.81,0.0,0.0,9.81,3,3,3,3,3,3,3,3`

### Calibration Status Messages (`C,`)
Format: `C,fs,fg,fa,fm,ts,tg,ta,tm`

Where:
- `fs`: femur system calibration (0-3)
- `fg`: femur gyro calibration (0-3)
- `fa`: femur accelerometer calibration (0-3)
- `fm`: femur magnetometer calibration (0-3)
- `ts`: tibia system calibration (0-3)
- `tg`: tibia gyro calibration (0-3)
- `ta`: tibia accelerometer calibration (0-3)
- `tm`: tibia magnetometer calibration (0-3)

Example: `C,3,3,3,3,3,3,3,3` (fully calibrated)

### Information Messages (`I,`)
Format: `I,message`

Used for status updates and general information.

Examples:
- `I,Starting data stream`
- `I,Checking calibration status...`
- `I,Calibration loaded successfully!`

### Error Messages (`E,`)
Format: `E,error_message`

Used for error reporting. These are converted to Python RuntimeErrors.

Example: `E,No BNO055 detected ... Check your wiring or I2C ADDR!`

## Message Processing

### Python RawData Object
Messages are parsed into a RawData object with:
- `msg_type`: MessageType enum (DATA, CALIBRATION, INFO, ERROR)
- `message`: Optional string for INFO messages
- `tibia_rotation`: 3x3 rotation matrix (DATA only)
- `femur_rotation`: 3x3 rotation matrix (DATA only)
- `timestamp`: Time in milliseconds (DATA only)
- `calibration`: Numpy array of calibration values
  - For DATA: All 8 calibration values
  - For CALIBRATION: All 8 calibration values
  - For INFO/ERROR: None

### Processing Rules
1. Data messages (`D,`):
   - Quaternions are converted to rotation matrices
   - Accelerometer data is stored
   - Calibration status is stored
   - All fields of RawData are populated

2. Calibration messages (`C,`):
   - Only calibration and msg_type fields are set
   - Other fields remain None

3. Info messages (`I,`):
   - Only message and msg_type fields are set
   - Other fields remain None

4. Error messages (`E,`):
   - Raise RuntimeError with the error message

## Notes
- All floating-point values use decimal point (.)
- Values in data messages are comma-separated
- Messages end with newline character (\n)
- Timestamps are in milliseconds since Arduino startup
- Invalid messages are logged and skipped
- Empty lines are ignored 