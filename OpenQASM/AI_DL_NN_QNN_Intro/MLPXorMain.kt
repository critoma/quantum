package mlp_xor

import java.util.*

// XORNeuralNetworkTwoHiddenLayers
internal class XORNeuralNetworkTwoHiddenLayers {
    private val numOperands = 2 // number nodes in input layer, and in each hidden1 and hidden2 layers
    private val inputNeurons = DoubleArray(numOperands)
    private val weightsInputHidden1 = Array(numOperands) { DoubleArray(numOperands) } // input → hidden1
    private var hidden1 = DoubleArray(numOperands)
    private val weightsHidden1Hidden2 = Array(numOperands) { DoubleArray(numOperands) } // hidden1 → hidden2
    private var hidden2 = DoubleArray(numOperands)
    private val weightsHidden2Output = DoubleArray(numOperands) // hidden2 → output
    private val biasHidden1 = DoubleArray(numOperands)
    private val biasHidden2 = DoubleArray(numOperands)
    private var biasOutput: Double
    private var outputNeuron = 0.0
    private val LEARNING_RATE = 0.5
    private val rand = Random()

    init {
        for (i in 0..1) {
            inputNeurons[i] = 0.0
            biasHidden1[i] = rand.nextDouble() - 0.5
            biasHidden2[i] = rand.nextDouble() - 0.5
            weightsHidden2Output[i] = rand.nextDouble() - 0.5
            for (j in 0..1) {
                weightsInputHidden1[j][i] = rand.nextDouble() - 0.5
                weightsHidden1Hidden2[j][i] = rand.nextDouble() - 0.5
            }
        }
        biasOutput = rand.nextDouble() - 0.5
    }

    private fun sigmoid(x: Double): Double {
        return 1.0 / (1.0 + Math.exp(-x))
    }

    private fun sigmoidDerivative(x: Double): Double {
        return x * (1.0 - x)
    }

    fun train(inputs: Array<DoubleArray>, targets: DoubleArray, epochs: Int) {
        for (epoch in 0 until epochs) {
            var totalError = 0.0
            for (i in inputs.indices) {
                val input = inputs[i]
                inputNeurons[0] = input[0]
                inputNeurons[1] = input[1]

                // Forward pass
                hidden1 = DoubleArray(numOperands)
                for (j in 0..1) {
                    hidden1[j] = sigmoid(
                        inputNeurons[0] * weightsInputHidden1[0][j] + inputNeurons[1] * weightsInputHidden1[1][j] +
                                biasHidden1[j]
                    )
                }
                hidden2 = DoubleArray(2)
                for (j in 0..1) {
                    hidden2[j] = sigmoid(
                        hidden1[0] * weightsHidden1Hidden2[0][j] + hidden1[1] * weightsHidden1Hidden2[1][j] +
                                biasHidden2[j]
                    )
                }
                val output = sigmoid(
                    hidden2[0] * weightsHidden2Output[0] + hidden2[1] * weightsHidden2Output[1] +
                            biasOutput
                )
                outputNeuron = output

                // Error
                val error = targets[i] - output
                totalError += error * error
                val dOutput = error * sigmoidDerivative(output)

                // Backpropagate to hidden2
                val dHidden2 = DoubleArray(2)
                for (j in 0..1) {
                    dHidden2[j] = dOutput * weightsHidden2Output[j] * sigmoidDerivative(hidden2[j])
                }

                // Backpropagate to hidden1
                val dHidden1 = DoubleArray(2)
                for (j in 0..1) {
                    dHidden1[j] = (dHidden2[0] * weightsHidden1Hidden2[j][0] +
                            dHidden2[1] * weightsHidden1Hidden2[j][1]) * sigmoidDerivative(hidden1[j])
                }

                // Update weights: hidden2 → output
                for (j in 0..1) {
                    weightsHidden2Output[j] += LEARNING_RATE * dOutput * hidden2[j]
                }
                biasOutput += LEARNING_RATE * dOutput

                // Update weights: hidden1 → hidden2
                for (j in 0..1) {
                    for (k in 0..1) {
                        weightsHidden1Hidden2[j][k] += LEARNING_RATE * dHidden2[k] * hidden1[j]
                    }
                    biasHidden2[j] += LEARNING_RATE * dHidden2[j]
                }

                // Update weights: input → hidden1
                for (j in 0..1) {
                    for (k in 0..1) {
                        weightsInputHidden1[k][j] += LEARNING_RATE * dHidden1[j] * input[k]
                    }
                    biasHidden1[j] += LEARNING_RATE * dHidden1[j]
                }
            }
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error: %.6f%n", epoch, totalError)
            }
        }
    }

    fun predict(x1: Double, x2: Double): Double {
        inputNeurons[0] = x1
        inputNeurons[1] = x2
        hidden1 = DoubleArray(2)
        for (j in 0..1) {
            hidden1[j] = sigmoid(
                inputNeurons[0] * weightsInputHidden1[0][j] + inputNeurons[1] * weightsInputHidden1[1][j] +
                        biasHidden1[j]
            )
        }
        hidden2 = DoubleArray(2)
        for (j in 0..1) {
            hidden2[j] = sigmoid(
                hidden1[0] * weightsHidden1Hidden2[0][j] + hidden1[1] * weightsHidden1Hidden2[1][j] +
                        biasHidden2[j]
            )
        }
        outputNeuron = sigmoid(
            hidden2[0] * weightsHidden2Output[0] + hidden2[1] * weightsHidden2Output[1] +
                    biasOutput
        )
        return outputNeuron
    }

    fun printANN(msg: String) {
        println("\n ####### print ANN start ####### $msg\n")
        println("\n\n Input neurons:")
        for (i in 0 until numOperands) {
            print(" " + inputNeurons[i])
        }
        println("\n\n  Weights neurons (input -> hidden1):")
        for (i in 0 until numOperands) { // NumInputs
            println()
            for (j in 0 until numOperands)  // NumHidden1
                if (weightsInputHidden1[i][j] >= 0) System.out.printf(
                    " +%.3f",
                    weightsInputHidden1[i][j]
                ) else System.out.printf(" %.3f", weightsInputHidden1[i][j])
        }
        println("\n\n Hidden1 neurons:")
        for (i in 0 until numOperands)  // NumHidden1
            print(" " + hidden1[i])
        println("\n\n  Weights neurons (hidden1 -> hidden2):")
        for (i in 0 until numOperands) { // NumInputs
            println()
            for (j in 0 until numOperands)  // NumHidden1
                if (weightsHidden1Hidden2[i][j] >= 0) System.out.printf(
                    " +%.3f",
                    weightsHidden1Hidden2[i][j]
                ) else System.out.printf(" %.3f", weightsHidden1Hidden2[i][j])
        }
        println("\n\n Hidden2 neurons:")
        for (i in 0 until numOperands)  // NumHidden2
            print(" " + hidden2[i])
        println("\n\n  Weights neurons (hidden2 -> output):")
        for (i in 0 until numOperands) { // NumInputs
            println()
            for (j in 0..0)  // NumHidden1
                if (weightsHidden2Output[i] >= 0) System.out.printf(
                    " +%.3f",
                    weightsHidden2Output[i]
                ) else System.out.printf(" %.3f", weightsHidden2Output[i])
        }
        println("\n\n Output neurons:")
        for (i in 0..0) print(" " + outputNeuron)
        println("\n ####### print ANN stop ####### $msg\n")
    }
}

object MLPXorMain {
    @JvmStatic
    fun main(args: Array<String>) {
        val nn = XORNeuralNetworkTwoHiddenLayers()
        val inputs =
            arrayOf(doubleArrayOf(0.0, 0.0), doubleArrayOf(0.0, 1.0), doubleArrayOf(1.0, 0.0), doubleArrayOf(1.0, 1.0))
        val targets = doubleArrayOf(0.0, 1.0, 1.0, 0.0)
        var startTime = System.nanoTime()
        nn.train(inputs, targets, 100000)
        var endTime = System.nanoTime()
        var executionTime = endTime - startTime
        println("Training takes " + executionTime + "ns")

        //nn.printANN(" -after training- ");
        println("Predictions:")
        for (input in inputs) {
            startTime = System.nanoTime()
            val out = nn.predict(input[0], input[1])
            endTime = System.nanoTime()
            executionTime = endTime - startTime
            println("Prediction takes " + executionTime + "ns")
            System.out.printf("Input: %.0f %.0f → Output: %.4f%n", input[0], input[1], out)
            //nn.printANN(" -after prediction for input[] = " + input[0] + ", " + input[1]);
        }
    }
}
