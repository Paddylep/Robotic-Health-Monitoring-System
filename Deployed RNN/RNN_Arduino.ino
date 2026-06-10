#include <LSM6DS3.h>
#include <Wire.h>
#include <math.h>
#include "regression_rnn_model.h"

// CONFIGURATION
const float accelerationThreshold = 0.5;
const int   numSamples            = 120;
const int   INPUT_SIZE            = 3;
const int   HIDDEN_SIZE           = 10;
const int   NUM_LAYERS            = 4;
const int   OUTPUT_SIZE           = 3;
#define     SAMPLE_PERIOD         2404

//CLASSIFICATION THRESHOLDS
const float MAE_THRESHOLD            = 0.1385f;
const int   LOCAL_WINDOW_SIZE        = 10;     
int Samplenum                        = 0;
int overallCorrect                   = 0;

// IMU
LSM6DS3 myIMU(I2C_MODE, 0x6A);
float offsetX = 0, offsetY = 0, offsetZ = 0;
int samplesRead = numSamples;

// LSTM STATE
float h_state[NUM_LAYERS][HIDDEN_SIZE];
float c_state[NUM_LAYERS][HIDDEN_SIZE];

// Buffers
float outputSequence[numSamples][OUTPUT_SIZE];
float inputSequence[numSamples][INPUT_SIZE];

inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// LSTM Cell
void lstm_cell(
    const float* W_ih, const float* W_hh,
    const float* b_ih, const float* b_hh,
    const float* x_in, int input_sz,
    float* h, float* c
) {
    float gates[4 * HIDDEN_SIZE];

    for (int i = 0; i < 4 * HIDDEN_SIZE; i++) {
        gates[i] = b_ih[i];
        for (int j = 0; j < input_sz; j++) {
            gates[i] += W_ih[i * input_sz + j] * x_in[j];
        }
    }
    for (int i = 0; i < 4 * HIDDEN_SIZE; i++) {
        gates[i] += b_hh[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            gates[i] += W_hh[i * HIDDEN_SIZE + j] * h[j];
        }
    }

    float* g_i = gates;
    float* g_f = gates + HIDDEN_SIZE;
    float* g_g = gates + 2 * HIDDEN_SIZE;
    float* g_o = gates + 3 * HIDDEN_SIZE;

    float new_h[HIDDEN_SIZE];
    for (int k = 0; k < HIDDEN_SIZE; k++) {
        float i_k = sigmoid(g_i[k]);
        float f_k = sigmoid(g_f[k]);
        float g_k = tanhf(g_g[k]);
        float o_k = sigmoid(g_o[k]);
        c[k] = f_k * c[k] + i_k * g_k;
        new_h[k] = o_k * tanhf(c[k]);
    }
    memcpy(h, new_h, HIDDEN_SIZE * sizeof(float));
}

// Full 4-Layer LSTM Forward
void lstm_forward(const float* input, float* output) {
    float layer_input[HIDDEN_SIZE];

    lstm_cell(rnn_weight_ih_l0, rnn_weight_hh_l0,
              rnn_bias_ih_l0,   rnn_bias_hh_l0,
              input, INPUT_SIZE, h_state[0], c_state[0]);
    memcpy(layer_input, h_state[0], HIDDEN_SIZE * sizeof(float));

    lstm_cell(rnn_weight_ih_l1, rnn_weight_hh_l1,
              rnn_bias_ih_l1,   rnn_bias_hh_l1,
              layer_input, HIDDEN_SIZE, h_state[1], c_state[1]);
    memcpy(layer_input, h_state[1], HIDDEN_SIZE * sizeof(float));

    lstm_cell(rnn_weight_ih_l2, rnn_weight_hh_l2,
              rnn_bias_ih_l2,   rnn_bias_hh_l2,
              layer_input, HIDDEN_SIZE, h_state[2], c_state[2]);
    memcpy(layer_input, h_state[2], HIDDEN_SIZE * sizeof(float));

    lstm_cell(rnn_weight_ih_l3, rnn_weight_hh_l3,
              rnn_bias_ih_l3,   rnn_bias_hh_l3,
              layer_input, HIDDEN_SIZE, h_state[3], c_state[3]);

    memcpy(output, h_state[3], HIDDEN_SIZE * sizeof(float));
}

// Fullt Connected Layer
void fc_forward(const float* input, float* output) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = fc_bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            output[i] += fc_weight[i * HIDDEN_SIZE + j] * input[j];
        }
    }
}

// Reset LSTM
void resetLSTMState() {
    memset(h_state, 0, sizeof(h_state));
    memset(c_state, 0, sizeof(c_state));
}

// Motion Analysis
bool analyseMotion() {

    float totalError = 0.0f;
    for (int t = 0; t < numSamples; t++) {
        for (int j = 0; j < min(OUTPUT_SIZE, INPUT_SIZE); j++) {
            totalError += fabs(outputSequence[t][j] - inputSequence[t][j]);
        }
    }
    float mae = totalError / (numSamples * min(OUTPUT_SIZE, INPUT_SIZE));

// Printing Results
    Serial.print("MAE (output vs input):    "); Serial.println(mae, 5);
    
// Failed Result
    bool failMAE        = (mae               > MAE_THRESHOLD);

    Serial.print("Flags — MAE:");        Serial.println(failMAE        ? "FAIL " : "ok   ");

    return !(failMAE);
}

// IMU Configuration
void config_PiCrawler_IMU() {
    myIMU.settings.accelEnabled    = 1;
    myIMU.settings.accelRange      = 4;
    myIMU.settings.accelSampleRate = 400;
    myIMU.settings.accelBandWidth  = 200;
}

// IMU Calibration
void calibrateIMU(int samples = 100) {
    float sumX = 0, sumY = 0, sumZ = 0;
    Serial.println("Calibrating IMU — keep still...");
    for (int i = 0; i < samples; i++) {
        sumX += myIMU.readFloatAccelX();
        sumY += myIMU.readFloatAccelY();
        sumZ += myIMU.readFloatAccelZ();
        delayMicroseconds(SAMPLE_PERIOD);
    }
    offsetX = sumX / samples;
    offsetY = sumY / samples;
    offsetZ = sumZ / samples;
    Serial.println("Calibration complete.");
    Serial.print("Offsets X/Y/Z: ");
    Serial.print(offsetX, 4); Serial.print(", ");
    Serial.print(offsetY, 4); Serial.print(", ");
    Serial.println(offsetZ, 4);
}

void setup() {
    Serial.begin(115200);
    while (!Serial);

    config_PiCrawler_IMU();

    if (myIMU.begin() != 0) {
        Serial.println("IMU error — halting.");
        while (1);
    }
    Serial.println("IMU OK!");

    calibrateIMU(100);
    resetLSTMState();

    Serial.println("Architecture: 4-layer LSTM | hidden=10 | input=3 | output=3");
    Serial.println("Ready — waiting for motion...");

    delay(500);
}

void loop() {
    float aX, aY, aZ;

    // Wait for significant motion
    while (samplesRead == numSamples) {
        aX = myIMU.readFloatAccelX() - offsetX;
        aY = myIMU.readFloatAccelY() - offsetY;
        aZ = myIMU.readFloatAccelZ() - offsetZ;

        if ((fabs(aX) + fabs(aY) + fabs(aZ)) >= accelerationThreshold) {
            samplesRead = 0;
            resetLSTMState();
            Serial.println("Motion detected — capturing...");
            break;
        }
    }

    // Feed timesteps one at a time
    while (samplesRead < numSamples) {
        unsigned long tStart = micros();

        aX = myIMU.readFloatAccelX() - offsetX;
        aY = myIMU.readFloatAccelY() - offsetY;
        aZ = myIMU.readFloatAccelZ() - offsetZ;

        float input[INPUT_SIZE];
        input[0] = (aX + 4.0f) / 8.0f;
        input[1] = (aY + 4.0f) / 8.0f;
        input[2] = (aZ + 4.0f) / 8.0f;

        for (int j = 0; j < INPUT_SIZE; j++) {
            inputSequence[samplesRead][j] = input[j];
        }

        float lstm_out[HIDDEN_SIZE];
        lstm_forward(input, lstm_out);

        float fc_out[OUTPUT_SIZE];
        fc_forward(lstm_out, fc_out);

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            outputSequence[samplesRead][j] = fc_out[j];
        }

        samplesRead++;
        // Get generalised error
        if (samplesRead == numSamples) {
            bool correct = analyseMotion();

            Serial.println("========================");
            if (correct) {
                overallCorrect++;
                Samplenum++;
                Serial.println("VERDICT: CORRECT MOTION");
            } else {
                overallCorrect = overallCorrect -1;
                Samplenum++;
                Serial.println("VERDICT: Incorrect motion");
            }
           
            if(Samplenum == 5){
                Serial.print("VERDICT: ");
                switch (overallCorrect) {
                case -1 ... 4:
                    Serial.println("Potential Issue");
                    overallCorrect = 0;
                break;
                case 5:
                    Serial.println("Working Correctly");
                    overallCorrect = 0;
                break;
                case -3 ... -2:
                    Serial.println("Issue");
                    overallCorrect = 0;
                break;
                case -6 ... -4:
                    Serial.println("Major Issue");
                    overallCorrect = 0;
                break;
                default:
                     overallCorrect = 0;
                break;
                }
                Samplenum = 0;
            }

            Serial.println("========================");
            
            delay(2.5);


            Serial.println("Ready — waiting for next motion...");
        }

        while ((micros() - tStart) < SAMPLE_PERIOD);
    }
}