#include "LSM6DS3.h"
#include "Wire.h"

//Create a instance of class LSM6DS3
LSM6DS3 myIMU(I2C_MODE, 0x6A);    //I2C device address 0x6A

const int ledPin = LED_BUILTIN; // pin to use for the LED
int incomingChar;

void config_PiCrawler_IMU() {

  // ---- Accelerometer ----
  myIMU.settings.accelEnabled     = 1;
  myIMU.settings.accelRange       = 4;     // ±4 g
  myIMU.settings.accelSampleRate  = 400;   // Hz
  myIMU.settings.accelBandWidth   = 100;   // LPF ~100 Hz

  Serial.println("PiCrawler IMU configured for 400 Hz ±4 g / ±500 °/s with LPF ≈ 100 Hz");
}

void setup() {
    Serial.begin(115200); //sets baud rate
    while (!Serial);
    pinMode(ledPin, OUTPUT);

    if (myIMU.begin() != 0) { //Applies settings
        Serial.println("Device error");
        return;
    }

    Serial.println("Device OK!");
    pinMode(ledPin, OUTPUT);
    config_PiCrawler_IMU();
}

// Prints IMU data
void loop() {
  for(int i = 0; i <= 200000; i++){
        incomingChar = Serial.read();
        Serial.print(myIMU.readFloatAccelX(), 4);
        Serial.print(" , ");
        Serial.print(myIMU.readFloatAccelY(), 4);
        Serial.print(" , ");
        Serial.print(myIMU.readFloatAccelZ(), 4);
        Serial.print(" ,\n ");
  }
}