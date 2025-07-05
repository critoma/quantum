import numpy as np
import time

class XORNeuralNetworkTwoHiddenLayers:
    def __init__(self):
        self.num_operands = 2
        self.input_neurons = np.zeros(self.num_operands)
        self.hidden1 = np.zeros(self.num_operands)
        self.hidden2 = np.zeros(self.num_operands)
        self.output_neuron = 0.0
        self.learning_rate = 0.5

        # Random weight initialization
        self.weights_input_hidden1 = np.random.rand(self.num_operands, self.num_operands) - 0.5
        self.weights_hidden1_hidden2 = np.random.rand(self.num_operands, self.num_operands) - 0.5
        self.weights_hidden2_output = np.random.rand(self.num_operands) - 0.5

        self.bias_hidden1 = np.random.rand(self.num_operands) - 0.5
        self.bias_hidden2 = np.random.rand(self.num_operands) - 0.5
        self.bias_output = np.random.rand() - 0.5

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            total_error = 0
            for idx, input_vec in enumerate(inputs):
                self.input_neurons = np.array(input_vec)

                # Forward pass
                self.hidden1 = self.sigmoid(np.dot(self.input_neurons, self.weights_input_hidden1) + self.bias_hidden1)
                self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights_hidden1_hidden2) + self.bias_hidden2)
                output = self.sigmoid(np.dot(self.hidden2, self.weights_hidden2_output) + self.bias_output)
                self.output_neuron = output

                # Error and backpropagation
                error = targets[idx] - output
                total_error += error ** 2
                d_output = error * self.sigmoid_derivative(output)

                d_hidden2 = d_output * self.weights_hidden2_output * self.sigmoid_derivative(self.hidden2)
                d_hidden1 = np.dot(self.weights_hidden1_hidden2, d_hidden2) * self.sigmoid_derivative(self.hidden1)

                # Update weights and biases
                self.weights_hidden2_output += self.learning_rate * d_output * self.hidden2
                self.bias_output += self.learning_rate * d_output

                self.weights_hidden1_hidden2 += self.learning_rate * np.outer(self.hidden1, d_hidden2)
                self.bias_hidden2 += self.learning_rate * d_hidden2

                self.weights_input_hidden1 += self.learning_rate * np.outer(self.input_neurons, d_hidden1)
                self.bias_hidden1 += self.learning_rate * d_hidden1

            if epoch % 1000 == 0:
                print(f"Epoch {epoch} - Error: {total_error:.6f}")

    def predict(self, x1, x2):
        self.input_neurons = np.array([x1, x2])
        self.hidden1 = self.sigmoid(np.dot(self.input_neurons, self.weights_input_hidden1) + self.bias_hidden1)
        self.hidden2 = self.sigmoid(np.dot(self.hidden1, self.weights_hidden1_hidden2) + self.bias_hidden2)
        self.output_neuron = self.sigmoid(np.dot(self.hidden2, self.weights_hidden2_output) + self.bias_output)
        return self.output_neuron

    def print_ann(self, msg):
        print(f"\n ####### print ANN start ####### {msg}\n")
        print("\nInput neurons:")
        print(self.input_neurons)

        print("\nWeights (input -> hidden1):")
        print(self.weights_input_hidden1)

        print("\nHidden1 neurons:")
        print(self.hidden1)

        print("\nWeights (hidden1 -> hidden2):")
        print(self.weights_hidden1_hidden2)

        print("\nHidden2 neurons:")
        print(self.hidden2)

        print("\nWeights (hidden2 -> output):")
        print(self.weights_hidden2_output)

        print("\nOutput neuron:")
        print(self.output_neuron)

        print(f"\n ####### print ANN stop ####### {msg}\n")


class MLPXorMain:
    @staticmethod
    def run():
        nn = XORNeuralNetworkTwoHiddenLayers()
        inputs = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        targets = np.array([0, 1, 1, 0])

        start_time = time.time_ns()
        nn.train(inputs, targets, 100_000)
        end_time = time.time_ns()

        print(f"Training takes {end_time - start_time} ns")

        print("\nPredictions:")
        for input_vec in inputs:
            start_time = time.time_ns()
            output = nn.predict(input_vec[0], input_vec[1])
            end_time = time.time_ns()
            print(f"Prediction takes {end_time - start_time} ns")
            print(f"Input: {int(input_vec[0])} {int(input_vec[1])} → Output: {output:.4f}")


if __name__ == "__main__":
    MLPXorMain.run()
