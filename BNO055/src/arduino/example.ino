#include <Wire.h>
#include <Arduino.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <math.h>
#include <MatrixMath.h>
#include <EEPROM.h>


/***** BNO055 setup *****/
// For Arduino Uno/Nano:
// Connect SCL to Analog pin 5 (A5)
// Connect SDA to Analog pin 4 (A4)

#define BNO055_SAMPLERATE_DELAY_MS (50)
// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
Adafruit_BNO055 bno_femur = Adafruit_BNO055(-1, 0x29, &Wire);
Adafruit_BNO055 bno_tibia = Adafruit_BNO055(-1, 0x28, &Wire);
long int start_time;

// EEPROM addresses for calibration data
#define EEPROM_FEMUR_ADDR 0
#define EEPROM_TIBIA_ADDR (EEPROM_FEMUR_ADDR + sizeof(adafruit_bno055_offsets_t))

// Message type prefixes
#define DATA_PREFIX "D,"        // Regular data
#define CAL_STATUS_PREFIX "C,"  // Calibration status
#define INFO_PREFIX "I,"        // Information messages
#define ERROR_PREFIX "E,"       // Error messages
#define COMMAND_SAVE_CAL "SAVE_CAL"    // Command to save calibration
#define COMMAND_PAUSE "PAUSE"          // Command to pause data streaming
#define COMMAND_RESUME "RESUME"        // Command to resume data streaming

/***** BNO055 setup *****/
// Forward declarations - remove default argument here
void vector_print(imu::Vector<3> euler, bool convert_to_radians);
void quaternion_print(imu::Quaternion quaternion);
double* quaternion_multiply(double q1[4], double q2[4]);

// Add with other global variables at the top
bool streaming_enabled = true;

void setup(void)
{
  Serial.begin(115200);
  start_time = millis();
  while (!Serial) delay(10);  // wait for serial port to open!

  /* Initialise the sensors */
  if(!bno_femur.begin() || !bno_tibia.begin())
  {
    Serial.print(ERROR_PREFIX);
    Serial.println("No BNO055 detected ... Check your wiring or I2C ADDR!");
    while(1);
  }

  Serial.println("cc");


  // Set initial mode to NDOF (9-DOF fusion)
  bno_femur.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  bno_tibia.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  
  // Add these lines to set external crystal use
  bno_femur.setExtCrystalUse(true);
  bno_tibia.setExtCrystalUse(true);
  
  delay(1000);  // Wait for sensors to stabilize
  
  // Try to load calibration
  loadCalibration();
  
  // Now we're calibrated, start normal operation
  Serial.print(INFO_PREFIX);
  Serial.println("Starting data stream");
}

void print_calibration_status() {
  uint8_t system_femur, gyro_femur, accel_femur, mag_femur;
  uint8_t system_tibia, gyro_tibia, accel_tibia, mag_tibia;
  bno_femur.getCalibration(&system_femur, &gyro_femur, &accel_femur, &mag_femur);
  bno_tibia.getCalibration(&system_tibia, &gyro_tibia, &accel_tibia, &mag_tibia);
  Serial.print(system_femur);
  Serial.print(",");
  Serial.print(gyro_femur);
  Serial.print(",");
  Serial.print(accel_femur);
  Serial.print(",");
  Serial.print(mag_femur);
  Serial.print(",");
  Serial.print(system_tibia);
  Serial.print(",");
  Serial.print(gyro_tibia);
  Serial.print(",");
  Serial.print(accel_tibia);
  Serial.print(",");
  Serial.println(mag_tibia);
}

void loop(void)
{
  // Check for incoming commands
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();  // Remove any whitespace/newlines
    
    if (command == COMMAND_SAVE_CAL) {
      saveCalibration();
      Serial.print(INFO_PREFIX);
      Serial.println("Calibration saved!");
    }
    else if (command == COMMAND_PAUSE) {
      streaming_enabled = false;
      Serial.print(INFO_PREFIX);
      Serial.println("Data streaming paused");
    }
    else if (command == COMMAND_RESUME) {
      streaming_enabled = true;
      Serial.print(INFO_PREFIX);
      Serial.println("Data streaming resumed");
    }
  }

  if (!streaming_enabled) {
    delay(100);  // Small delay when paused to prevent busy-waiting
    return;
  }

  // Get quaternion and acceleration data
  imu::Quaternion femur = bno_femur.getQuat();
  imu::Quaternion tibia = bno_tibia.getQuat();
  imu::Vector<3> femur_acc = bno_femur.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3> tibia_acc = bno_tibia.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);

  // Print data with prefix
  Serial.print(DATA_PREFIX);
  Serial.print(millis()-start_time);
  Serial.print(",");
  quaternion_print(femur);
  Serial.print(",");
  quaternion_print(tibia);
  Serial.print(",");
  vector_print(femur_acc, false);
  Serial.print(",");
  vector_print(tibia_acc, false);
  Serial.print(",");
  print_calibration_status();
  Serial.println();

  delay(BNO055_SAMPLERATE_DELAY_MS);
}

void vector_print(imu::Vector<3> euler, bool convert_to_radians=false) {
  double x = euler.x();
  double y = euler.y();
  double z = euler.z();

  if (convert_to_radians) {
    x = x*180./PI;
    y = y*180./PI;
    z = z*180./PI;
  }

  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(",");
  Serial.print(z);
}

void quaternion_print(imu::Quaternion quaternion) {
  // Serial.print("Quaternion: ");
  Serial.print(quaternion.w(), 4);
  Serial.print(",");
  Serial.print(quaternion.x(), 4);
  Serial.print(",");
  Serial.print(quaternion.y(), 4);
  Serial.print(",");
  Serial.print(quaternion.z(), 4);
}

void saveCalibration()
{
  adafruit_bno055_offsets_t femur_offsets;
  adafruit_bno055_offsets_t tibia_offsets;
  
  // Get the sensor's calibration data
  bool femur_success = bno_femur.getSensorOffsets(femur_offsets);
  bool tibia_success = bno_tibia.getSensorOffsets(tibia_offsets);

  if (femur_success && tibia_success) {
    // Save femur calibration
    EEPROM.put(EEPROM_FEMUR_ADDR, femur_offsets);
    // Save tibia calibration
    EEPROM.put(EEPROM_TIBIA_ADDR, tibia_offsets);
    Serial.print(INFO_PREFIX);
    Serial.println("Calibration data saved successfully");
  } else {
    Serial.print(ERROR_PREFIX);
    Serial.println("Failed to get calibration data from sensors");
  }
}

void loadCalibration()
{
  adafruit_bno055_offsets_t femur_offsets;
  adafruit_bno055_offsets_t tibia_offsets;

  // Read calibration data from EEPROM
  EEPROM.get(EEPROM_FEMUR_ADDR, femur_offsets);
  EEPROM.get(EEPROM_TIBIA_ADDR, tibia_offsets);

  // Reset the sensors
  bno_femur.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_CONFIG);
  bno_tibia.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_CONFIG);
  delay(25);
  
  // Use public method for system reset instead of private write8
  bno_femur.setExtCrystalUse(true);  // This triggers a reset
  bno_tibia.setExtCrystalUse(true);
  delay(50);

  // Apply offsets
  bno_femur.setSensorOffsets(femur_offsets);
  bno_tibia.setSensorOffsets(tibia_offsets);

  // Set back to NDOF mode
  bno_femur.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  bno_tibia.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  
  delay(25);  // Wait for mode switch
}
