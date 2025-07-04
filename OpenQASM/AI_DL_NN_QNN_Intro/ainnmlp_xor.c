// gcc -o ainnmlp_xor.elf64 ainnmlp_xor.c -lm
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define NUM_OPERANDS 2
#define LEARNING_RATE 0.5
#define EPOCHS 100000

double randWeight() {
    return ((double)rand() / RAND_MAX) - 0.5;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1.0 - x);
}

typedef struct {
    double inputNeurons[NUM_OPERANDS];
    double weightsInputHidden1[NUM_OPERANDS][NUM_OPERANDS];
    double hidden1[NUM_OPERANDS];
    double weightsHidden1Hidden2[NUM_OPERANDS][NUM_OPERANDS];
    double hidden2[NUM_OPERANDS];
    double weightsHidden2Output[NUM_OPERANDS];
    double biasHidden1[NUM_OPERANDS];
    double biasHidden2[NUM_OPERANDS];
    double biasOutput;
    double outputNeuron;
} NeuralNetwork;

void initializeNetwork(NeuralNetwork *nn) {
    for (int i = 0; i < NUM_OPERANDS; i++) {
        nn->inputNeurons[i] = 0.0;
        nn->biasHidden1[i] = randWeight();
        nn->biasHidden2[i] = randWeight();
        nn->weightsHidden2Output[i] = randWeight();
        for (int j = 0; j < NUM_OPERANDS; j++) {
            nn->weightsInputHidden1[j][i] = randWeight();
            nn->weightsHidden1Hidden2[j][i] = randWeight();
        }
    }
    nn->biasOutput = randWeight();
}

void train(NeuralNetwork *nn, double inputs[][2], double targets[], int size) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double totalError = 0;

        for (int i = 0; i < size; i++) {
            double *input = inputs[i];
            nn->inputNeurons[0] = input[0];
            nn->inputNeurons[1] = input[1];

            for (int j = 0; j < NUM_OPERANDS; j++) {
                nn->hidden1[j] = sigmoid(
                    nn->inputNeurons[0] * nn->weightsInputHidden1[0][j] +
                    nn->inputNeurons[1] * nn->weightsInputHidden1[1][j] +
                    nn->biasHidden1[j]);
            }

            for (int j = 0; j < NUM_OPERANDS; j++) {
                nn->hidden2[j] = sigmoid(
                    nn->hidden1[0] * nn->weightsHidden1Hidden2[0][j] +
                    nn->hidden1[1] * nn->weightsHidden1Hidden2[1][j] +
                    nn->biasHidden2[j]);
            }

            double output = sigmoid(
                nn->hidden2[0] * nn->weightsHidden2Output[0] +
                nn->hidden2[1] * nn->weightsHidden2Output[1] +
                nn->biasOutput);

            nn->outputNeuron = output;

            double error = targets[i] - output;
            totalError += error * error;
            double dOutput = error * sigmoidDerivative(output);

            double dHidden2[NUM_OPERANDS];
            for (int j = 0; j < NUM_OPERANDS; j++) {
                dHidden2[j] = dOutput * nn->weightsHidden2Output[j] * sigmoidDerivative(nn->hidden2[j]);
            }

            double dHidden1[NUM_OPERANDS];
            for (int j = 0; j < NUM_OPERANDS; j++) {
                dHidden1[j] = (dHidden2[0] * nn->weightsHidden1Hidden2[j][0] +
                               dHidden2[1] * nn->weightsHidden1Hidden2[j][1]) *
                               sigmoidDerivative(nn->hidden1[j]);
            }

            for (int j = 0; j < NUM_OPERANDS; j++) {
                nn->weightsHidden2Output[j] += LEARNING_RATE * dOutput * nn->hidden2[j];
            }
            nn->biasOutput += LEARNING_RATE * dOutput;

            for (int j = 0; j < NUM_OPERANDS; j++) {
                for (int k = 0; k < NUM_OPERANDS; k++) {
                    nn->weightsHidden1Hidden2[k][j] += LEARNING_RATE * dHidden2[j] * nn->hidden1[k];
                }
                nn->biasHidden2[j] += LEARNING_RATE * dHidden2[j];
            }

            for (int j = 0; j < NUM_OPERANDS; j++) {
                for (int k = 0; k < NUM_OPERANDS; k++) {
                    nn->weightsInputHidden1[k][j] += LEARNING_RATE * dHidden1[j] * input[k];
                }
                nn->biasHidden1[j] += LEARNING_RATE * dHidden1[j];
            }
        }

        if (epoch % 1000 == 0) {
            printf("Epoch %d - Error: %.6f\n", epoch, totalError);
        }
    }
}

double predict(NeuralNetwork *nn, double x1, double x2) {
    nn->inputNeurons[0] = x1;
    nn->inputNeurons[1] = x2;

    for (int j = 0; j < NUM_OPERANDS; j++) {
        nn->hidden1[j] = sigmoid(
            nn->inputNeurons[0] * nn->weightsInputHidden1[0][j] +
            nn->inputNeurons[1] * nn->weightsInputHidden1[1][j] +
            nn->biasHidden1[j]);
    }

    for (int j = 0; j < NUM_OPERANDS; j++) {
        nn->hidden2[j] = sigmoid(
            nn->hidden1[0] * nn->weightsHidden1Hidden2[0][j] +
            nn->hidden1[1] * nn->weightsHidden1Hidden2[1][j] +
            nn->biasHidden2[j]);
    }

    return sigmoid(
        nn->hidden2[0] * nn->weightsHidden2Output[0] +
        nn->hidden2[1] * nn->weightsHidden2Output[1] +
        nn->biasOutput);
}

int main() {
    srand(time(NULL));
    NeuralNetwork nn;
    initializeNetwork(&nn);

    double inputs[4][2] = {{0,0}, {0,1}, {1,0}, {1,1}};
    double targets[4] = {0, 1, 1, 0};

    train(&nn, inputs, targets, 4);

    printf("\nPredictions:\n");
    for (int i = 0; i < 4; i++) {
        double out = predict(&nn, inputs[i][0], inputs[i][1]);
        printf("Input: %.0f %.0f -> Output: %.4f\n", inputs[i][0], inputs[i][1], out);
    }

    return 0;
}