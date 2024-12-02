#include <Wire.h>
#include <Arduino.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include <math.h>
#include <MatrixMath.h>


/***** BNO055 setup *****/
#define BNO055_SAMPLERATE_DELAY_MS (50)
// Check I2C device address and correct line below (by default address is 0x29 or 0x28)
Adafruit_BNO055 bno_femur = Adafruit_BNO055(-1, 0x28, &Wire);
Adafruit_BNO055 bno_tibia = Adafruit_BNO055(-1, 0x29, &Wire);
long int start_time;

/***** BNO055 setup *****/
void vector_print(imu::Vector<3> euler, bool convert_to_radians=false);
void quaternion_print(imu::Quaternion quaternion);
double* quaternion_multiply(double q1[4], double q2[4]);

void setup(void)
{
  Serial.begin(115200);
  start_time = millis();
  while (!Serial) delay(10);  // wait for serial port to open!

  // Serial.println("\nBNO055\n");
  Serial.print("time,femur_w,femur_x,femur_y,femur_z,femur_acc_x,femur_acc_y,femur_acc_z,tibia_w,tibia_x,tibia_y,tibia_z,tibia_acc_x,tibia_acc_y,tibia_acc_z,gf,mf,gt,mt\n");

  /* Initialise the sensor */
  if(!bno_femur.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 on femur not detected ... Check your wiring or I2C ADDR!\n");
    while(1);
  }

  if(!bno_tibia.begin())
  {
    /* There was a problem detecting the BNO055 ... check your connections */
    Serial.print("Ooops, no BNO055 on tibia not detected ... Check your wiring or I2C ADDR!\n");
  }

  delay(1000);
}

void loop(void)
{
  /* Display calibration status for each sensor. */
  uint8_t system_femur, gyro_femur, accel_femur, mag_femur = 0;
  uint8_t system_tibia, gyro_tibia, accel_tibia, mag_tibia = 0;
  bno_femur.getCalibration(&system_femur, &gyro_femur, &accel_femur, &mag_femur);
  bno_tibia.getCalibration(&system_tibia, &gyro_tibia, &accel_tibia, &mag_tibia);

    /* Display calibration status for each sensor. */
  // Serial.print("CALIBRATION FEMUR: Sys=");
  // Serial.print(system_femur, DEC);
  // Serial.print(" Gyro=");
  // Serial.print(gyro_femur, DEC);
  // Serial.print(" Accel=");
  // Serial.print(accel_femur, DEC);
  // Serial.print(" Mag=");
  // Serial.print(mag_femur, DEC);
  // Serial.print("CALIBRATION TIBIA: Sys=");
  // Serial.print(system_tibia, DEC);
  // Serial.print(" Gyro=");
  // Serial.print(gyro_tibia, DEC);
  // Serial.print(" Accel=");
  // Serial.print(accel_tibia, DEC);
  // Serial.print(" Mag=");
  // Serial.println(mag_tibia, DEC);

  // if (gyro_femur + mag_femur < 6) Serial.print("Femur needs calibration\n");
  // if (gyro_tibia + mag_tibia < 6) Serial.print("Tibia needs calibration\n");

  // Possible vector values can be:
  // - VECTOR_ACCELEROMETER - m/s^2
  // - VECTOR_MAGNETOMETER  - uT
  // - VECTOR_GYROSCOPE     - rad/s
  // - VECTOR_EULER         - degrees
  // - VECTOR_LINEARACCEL   - m/s^2
  // - VECTOR_GRAVITY       - m/s^2

  // Quaternion data
  imu::Quaternion femur = bno_femur.getQuat();
  imu::Quaternion tibia = bno_tibia.getQuat();
  imu::Vector<3>  femur_acc = bno_femur.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Vector<3>  tibia_acc = bno_tibia.getVector(Adafruit_BNO055::VECTOR_ACCELEROMETER);
  imu::Quaternion relative_tibia = femur.conjugate() * tibia;    // Tibia in femur frame
  imu::Vector<3> relative_tibia_euler = relative_tibia.toEuler();


  // vector_print(relative_tibia_euler, true);
  Serial.print(millis()-start_time);
  Serial.print(",");
  quaternion_print(femur);
  Serial.print(",");
  quaternion_print(tibia);
  Serial.print(",");
  vector_print(femur_acc);
  Serial.print(",");
  vector_print(tibia_acc);
  Serial.print(",");
  Serial.print(gyro_femur);
  Serial.print(",");
  Serial.print(mag_femur);
  Serial.print(",");
  Serial.print(gyro_tibia);
  Serial.print(",");
  Serial.print(mag_tibia);
  Serial.print("\n");


  double knee_flexion = 180 + relative_tibia_euler[1]*180/PI;
  double knee_valgus = relative_tibia_euler[0]*180/PI;

  
  imu::Vector<3> femur_gravity = bno_femur.getVector(Adafruit_BNO055::VECTOR_GRAVITY);
  double azimuth = sqrt(femur_gravity[0]*femur_gravity[0] + femur_gravity[1]*femur_gravity[1]) + 1E-6;
  double femur_pitch = atan(femur_gravity[2] / azimuth)*180/PI;


  // Serial.print("Femur pitch      : "); Serial.println(femur_pitch);
  // Serial.print("Knee flexion     : "); Serial.println(knee_flexion);
  // Serial.print("Knee varus/valgus: "); Serial.println(knee_valgus);

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
