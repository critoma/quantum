class XORNeuralNetworkTwoHiddenLayers {
    constructor() {
        this.numOperands = 2;
        this.inputNeurons = [0, 0];
        this.weightsInputHidden1 = Array.from({ length: 2 }, () => [Math.random() - 0.5, Math.random() - 0.5]);
        this.weightsHidden1Hidden2 = Array.from({ length: 2 }, () => [Math.random() - 0.5, Math.random() - 0.5]);
        this.weightsHidden2Output = [Math.random() - 0.5, Math.random() - 0.5];
        this.biasHidden1 = [Math.random() - 0.5, Math.random() - 0.5];
        this.biasHidden2 = [Math.random() - 0.5, Math.random() - 0.5];
        this.biasOutput = Math.random() - 0.5;
        this.outputNeuron = 0;
        this.LEARNING_RATE = 0.5;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    sigmoidDerivative(x) {
        return x * (1 - x);
    }

    train(inputs, targets, epochs) {
        for (let epoch = 0; epoch < epochs; epoch++) {
            let totalError = 0;

            for (let i = 0; i < inputs.length; i++) {
                const input = inputs[i];
                this.inputNeurons[0] = input[0];
                this.inputNeurons[1] = input[1];

                // Forward pass
                const hidden1 = [0, 0];
                for (let j = 0; j < 2; j++) {
                    hidden1[j] = this.sigmoid(
                        this.inputNeurons[0] * this.weightsInputHidden1[0][j] +
                        this.inputNeurons[1] * this.weightsInputHidden1[1][j] +
                        this.biasHidden1[j]
                    );
                }

                const hidden2 = [0, 0];
                for (let j = 0; j < 2; j++) {
                    hidden2[j] = this.sigmoid(
                        hidden1[0] * this.weightsHidden1Hidden2[0][j] +
                        hidden1[1] * this.weightsHidden1Hidden2[1][j] +
                        this.biasHidden2[j]
                    );
                }

                const output = this.sigmoid(
                    hidden2[0] * this.weightsHidden2Output[0] +
                    hidden2[1] * this.weightsHidden2Output[1] +
                    this.biasOutput
                );
                this.outputNeuron = output;

                // Error
                const error = targets[i] - output;
                totalError += error * error;
                const dOutput = error * this.sigmoidDerivative(output);

                // Backpropagate to hidden2
                const dHidden2 = [0, 0];
                for (let j = 0; j < 2; j++) {
                    dHidden2[j] = dOutput * this.weightsHidden2Output[j] * this.sigmoidDerivative(hidden2[j]);
                }

                // Backpropagate to hidden1
                const dHidden1 = [0, 0];
                for (let j = 0; j < 2; j++) {
                    dHidden1[j] = (
                        dHidden2[0] * this.weightsHidden1Hidden2[j][0] +
                        dHidden2[1] * this.weightsHidden1Hidden2[j][1]
                    ) * this.sigmoidDerivative(hidden1[j]);
                }

                // Update weights and biases: hidden2 → output
                for (let j = 0; j < 2; j++) {
                    this.weightsHidden2Output[j] += this.LEARNING_RATE * dOutput * hidden2[j];
                }
                this.biasOutput += this.LEARNING_RATE * dOutput;

                // Update weights and biases: hidden1 → hidden2
                for (let j = 0; j < 2; j++) {
                    for (let k = 0; k < 2; k++) {
                        this.weightsHidden1Hidden2[j][k] += this.LEARNING_RATE * dHidden2[k] * hidden1[j];
                    }
                    this.biasHidden2[j] += this.LEARNING_RATE * dHidden2[j];
                }

                // Update weights and biases: input → hidden1
                for (let j = 0; j < 2; j++) {
                    for (let k = 0; k < 2; k++) {
                        this.weightsInputHidden1[k][j] += this.LEARNING_RATE * dHidden1[j] * input[k];
                    }
                    this.biasHidden1[j] += this.LEARNING_RATE * dHidden1[j];
                }
            }

            if (epoch % 1000 === 0) {
                console.log(`Epoch ${epoch} - Error: ${totalError.toFixed(6)}`);
            }
        }
    }

    predict(x1, x2) {
        this.inputNeurons[0] = x1;
        this.inputNeurons[1] = x2;

        const hidden1 = [0, 0];
        for (let j = 0; j < 2; j++) {
            hidden1[j] = this.sigmoid(
                this.inputNeurons[0] * this.weightsInputHidden1[0][j] +
                this.inputNeurons[1] * this.weightsInputHidden1[1][j] +
                this.biasHidden1[j]
            );
        }

        const hidden2 = [0, 0];
        for (let j = 0; j < 2; j++) {
            hidden2[j] = this.sigmoid(
                hidden1[0] * this.weightsHidden1Hidden2[0][j] +
                hidden1[1] * this.weightsHidden1Hidden2[1][j] +
                this.biasHidden2[j]
            );
        }

        this.outputNeuron = this.sigmoid(
            hidden2[0] * this.weightsHidden2Output[0] +
            hidden2[1] * this.weightsHidden2Output[1] +
            this.biasOutput
        );

        return this.outputNeuron;
    }
}

// ====== Main Program ======
const nn = new XORNeuralNetworkTwoHiddenLayers();

const inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
];
const targets = [0, 1, 1, 0];

const startTime = process.hrtime.bigint();
nn.train(inputs, targets, 100000);
const endTime = process.hrtime.bigint();
console.log(`Training takes ${endTime - startTime}ns`);

console.log("Predictions:");
for (const input of inputs) {
    const predictStart = process.hrtime.bigint();
    const out = nn.predict(input[0], input[1]);
    const predictEnd = process.hrtime.bigint();
    console.log(`Prediction takes ${predictEnd - predictStart}ns`);
    console.log(`Input: ${input[0]} ${input[1]} → Output: ${out.toFixed(4)}`);
}
