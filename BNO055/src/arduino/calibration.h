#ifndef CALIBRATION_H
#define CALIBRATION_H

void waitForCalibration() {
  uint8_t system_femur, gyro_femur, accel_femur, mag_femur;
  uint8_t system_tibia, gyro_tibia, accel_tibia, mag_tibia;
  
  Serial.println("Checking calibration status...");
  unsigned long startTime = millis();
  const unsigned long QUICK_CALIBRATION_TIMEOUT = 2000; // 2 seconds to check if calibration loaded
  bool quickCalibration = true;
  
  while (true) {
    bno_femur.getCalibration(&system_femur, &gyro_femur, &accel_femur, &mag_femur);
    bno_tibia.getCalibration(&system_tibia, &gyro_tibia, &accel_tibia, &mag_tibia);

    // Print current calibration status
    Serial.print("Femur (sys=");
    Serial.print(system_femur);
    Serial.print(" gyro=");
    Serial.print(gyro_femur);
    Serial.print(" acc=");
    Serial.print(accel_femur);
    Serial.print(" mag=");
    Serial.print(mag_femur);
    Serial.print(") Tibia (sys=");
    Serial.print(system_tibia);
    Serial.print(" gyro=");
    Serial.print(gyro_tibia);
    Serial.print(" acc=");
    Serial.print(accel_tibia);
    Serial.print(" mag=");
    Serial.print(mag_tibia);
    Serial.println(")");

    // Check if well calibrated
    if (system_femur >= 2 && system_tibia >= 2 && 
        gyro_femur >= 2 && gyro_tibia >= 2 &&
        accel_femur >= 2 && accel_tibia >= 2 &&
        mag_femur >= 2 && mag_tibia >= 2) {
      
      if (quickCalibration) {
        Serial.println("Calibration loaded successfully!");
      } else {
        saveCalibration();
        Serial.println("New calibration complete and saved!");
      }
      return;
    }

    if (quickCalibration && (millis() - startTime > QUICK_CALIBRATION_TIMEOUT)) {
      quickCalibration = false;
      Serial.println("\nSaved calibration not detected or invalid");
      Serial.println("Starting full calibration process...");
      Serial.println("Please follow calibration steps:");
      Serial.println("1. Keep sensors still for gyro");
      Serial.println("2. Move through 6 positions for accelerometer");
      Serial.println("3. Move in figure-8 pattern for magnetometer");
    }

    delay(quickCalibration ? 50 : 200);
  }
}

void saveCalibration() {
  adafruit_bno055_offsets_t femur_offsets;
  adafruit_bno055_offsets_t tibia_offsets;
  
  bool femur_success = bno_femur.getSensorOffsets(femur_offsets);
  bool tibia_success = bno_tibia.getSensorOffsets(tibia_offsets);

  if (femur_success && tibia_success) {
    EEPROM.put(EEPROM_FEMUR_ADDR, femur_offsets);
    EEPROM.put(EEPROM_TIBIA_ADDR, tibia_offsets);
  }
}

void loadCalibration() {
  adafruit_bno055_offsets_t femur_offsets;
  adafruit_bno055_offsets_t tibia_offsets;

  EEPROM.get(EEPROM_FEMUR_ADDR, femur_offsets);
  EEPROM.get(EEPROM_TIBIA_ADDR, tibia_offsets);

  // Set mode to CONFIG
  bno_femur.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_CONFIG);
  bno_tibia.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_CONFIG);
  delay(25);

  // Apply offsets
  bno_femur.setSensorOffsets(femur_offsets);
  bno_tibia.setSensorOffsets(tibia_offsets);

  // Set back to NDOF mode
  bno_femur.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  bno_tibia.setMode(adafruit_bno055_opmode_t::OPERATION_MODE_NDOF);
  
  delay(25);
}

#endif // CALIBRATION_H 