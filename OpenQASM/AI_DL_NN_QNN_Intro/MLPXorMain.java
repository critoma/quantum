package mlp_xor;

import java.util.Random;

// XORNeuralNetworkTwoHiddenLayers
class XORNeuralNetworkTwoHiddenLayers {
    private int numOperands = 2; // number nodes in input layer, and in each hidden1 and hidden2 layers
    private double[] inputNeurons = new double[numOperands];
    private double[][] weightsInputHidden1 = new double[numOperands][numOperands];  // input → hidden1
    private double[] hidden1 = new double[numOperands];
    private double[][] weightsHidden1Hidden2 = new double[numOperands][numOperands]; // hidden1 → hidden2
    private double[] hidden2 = new double[numOperands];
    private double[] weightsHidden2Output = new double[numOperands];       // hidden2 → output
    private double[] biasHidden1 = new double[numOperands];
    private double[] biasHidden2 = new double[numOperands];
    private double biasOutput;
    private double outputNeuron;

    private final double LEARNING_RATE = 0.5;
    private final Random rand = new Random();

    public XORNeuralNetworkTwoHiddenLayers() {
        for (int i = 0; i < 2; i++) {
            inputNeurons[i] = 0.0;
            biasHidden1[i] = rand.nextDouble() - 0.5;
            biasHidden2[i] = rand.nextDouble() - 0.5;
            weightsHidden2Output[i] = rand.nextDouble() - 0.5;
            for (int j = 0; j < 2; j++) {
                weightsInputHidden1[j][i] = rand.nextDouble() - 0.5;
                weightsHidden1Hidden2[j][i] = rand.nextDouble() - 0.5;
            }
        }
        biasOutput = rand.nextDouble() - 0.5;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public void train(double[][] inputs, double[] targets, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0;

            for (int i = 0; i < inputs.length; i++) {
                double[] input = inputs[i];
                this.inputNeurons[0] = input[0];
                this.inputNeurons[1] = input[1];

                // Forward pass
                hidden1 = new double[numOperands];
                for (int j = 0; j < 2; j++) {
                    hidden1[j] = sigmoid(inputNeurons[0] * weightsInputHidden1[0][j] +
                                         inputNeurons[1] * weightsInputHidden1[1][j] +
                                         biasHidden1[j]);
                }

                hidden2 = new double[2];
                for (int j = 0; j < 2; j++) {
                    hidden2[j] = sigmoid(hidden1[0] * weightsHidden1Hidden2[0][j] +
                                         hidden1[1] * weightsHidden1Hidden2[1][j] +
                                         biasHidden2[j]);
                }

                double output = sigmoid(hidden2[0] * weightsHidden2Output[0] +
                                        hidden2[1] * weightsHidden2Output[1] +
                                        biasOutput);
                this.outputNeuron = output;

                // Error
                double error = targets[i] - output;
                totalError += error * error;
                double dOutput = error * sigmoidDerivative(output);

                // Backpropagate to hidden2
                double[] dHidden2 = new double[2];
                for (int j = 0; j < 2; j++) {
                    dHidden2[j] = dOutput * weightsHidden2Output[j] * sigmoidDerivative(hidden2[j]);
                }

                // Backpropagate to hidden1
                double[] dHidden1 = new double[2];
                for (int j = 0; j < 2; j++) {
                    dHidden1[j] = (dHidden2[0] * weightsHidden1Hidden2[j][0] +
                                   dHidden2[1] * weightsHidden1Hidden2[j][1]) * sigmoidDerivative(hidden1[j]);
                }

                // Update weights: hidden2 → output
                for (int j = 0; j < 2; j++) {
                    weightsHidden2Output[j] += LEARNING_RATE * dOutput * hidden2[j];
                }
                biasOutput += LEARNING_RATE * dOutput;

                // Update weights: hidden1 → hidden2
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        weightsHidden1Hidden2[j][k] += LEARNING_RATE * dHidden2[k] * hidden1[j];
                    }
                    biasHidden2[j] += LEARNING_RATE * dHidden2[j];
                }

                // Update weights: input → hidden1
                for (int j = 0; j < 2; j++) {
                    for (int k = 0; k < 2; k++) {
                        weightsInputHidden1[k][j] += LEARNING_RATE * dHidden1[j] * input[k];
                    }
                    biasHidden1[j] += LEARNING_RATE * dHidden1[j];
                }
            }

            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.6f%n", epoch, totalError);
            }
        }
    }

    public double predict(double x1, double x2) {
        this.inputNeurons[0] = x1;
        this.inputNeurons[1] = x2;
        hidden1 = new double[2];
        for (int j = 0; j < 2; j++) {
            hidden1[j] = sigmoid(this.inputNeurons[0] * weightsInputHidden1[0][j] +
                                 this.inputNeurons[1] * weightsInputHidden1[1][j] +
                                 biasHidden1[j]);
        }

        hidden2 = new double[2];
        for (int j = 0; j < 2; j++) {
            hidden2[j] = sigmoid(hidden1[0] * weightsHidden1Hidden2[0][j] +
                                 hidden1[1] * weightsHidden1Hidden2[1][j] +
                                 biasHidden2[j]);
        }

        this.outputNeuron =  sigmoid(hidden2[0] * weightsHidden2Output[0] +
                       hidden2[1] * weightsHidden2Output[1] +
                       biasOutput);
                       
        return this.outputNeuron;
    }

    public void printANN(String msg) {
        System.out.println("\n ####### print ANN start ####### " + msg + "\n");
        System.out.println("\n\n Input neurons:");
        for (int i = 0; i < numOperands; i++) {
            System.out.print(" " + inputNeurons[i]);
        }
            
        System.out.println("\n\n  Weights neurons (input -> hidden1):");
        for (int i = 0; i < numOperands; i++) { // NumInputs
            System.out.println();
            for (int j = 0; j < numOperands; j++) // NumHidden1
                if (weightsInputHidden1[i][j] >= 0)
                    System.out.printf(" +%.3f", weightsInputHidden1[i][j]);
                else
                    System.out.printf(" %.3f", weightsInputHidden1[i][j]);
        }
            
        System.out.println("\n\n Hidden1 neurons:");
        for (int i = 0; i < numOperands; i++) // NumHidden1
            System.out.print(" " + hidden1[i]);
            
        System.out.println("\n\n  Weights neurons (hidden1 -> hidden2):");
        for (int i = 0; i < numOperands; i++) { // NumInputs
            System.out.println();
            for (int j = 0; j < numOperands; j++) // NumHidden1
                if (weightsHidden1Hidden2[i][j] >= 0)
                    System.out.printf(" +%.3f", weightsHidden1Hidden2[i][j]);
                else
                    System.out.printf(" %.3f", weightsHidden1Hidden2[i][j]);
        }

        System.out.println("\n\n Hidden2 neurons:");
        for (int i = 0; i < numOperands; i++) // NumHidden2
            System.out.print(" " + hidden2[i]);
        
        System.out.println("\n\n  Weights neurons (hidden2 -> output):");
            for (int i = 0; i < numOperands; i++) { // NumInputs
                System.out.println();
                for (int j = 0; j < 1; j++) // NumHidden1
                    if (weightsHidden2Output[i] >= 0)
                        System.out.printf(" +%.3f", weightsHidden2Output[i]);
                    else
                        System.out.printf(" %.3f", weightsHidden2Output[i]);
            }
            
        System.out.println("\n\n Output neurons:");
        for (int i = 0; i < 1; i++)
            System.out.print(" " + this.outputNeuron);
        
        System.out.println("\n ####### print ANN stop ####### " + msg + "\n");
    }

    
}

public class MLPXorMain {
    public static void main(String[] args) {
        XORNeuralNetworkTwoHiddenLayers nn = new XORNeuralNetworkTwoHiddenLayers();

        double[][] inputs = {
            {0, 0},
            {0, 1},
            {1, 0},
            {1, 1}
        };

        double[] targets = {0, 1, 1, 0};

        nn.train(inputs, targets, 100000);
        
        //nn.printANN(" -after training- ");
        System.out.println("Predictions:");
        for (double[] input : inputs) {
            double out = nn.predict(input[0], input[1]);
            System.out.printf("Input: %.0f %.0f → Output: %.4f%n", input[0], input[1], out);
            //nn.printANN(" -after prediction for input[] = " + input[0] + ", " + input[1]);
        }
    }
}
