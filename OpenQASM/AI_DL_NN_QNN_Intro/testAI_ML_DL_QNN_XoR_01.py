from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import numpy as np
# import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator


# Define trainable parameters
# theta0 = Parameter('θ0')
# theta1 = Parameter('θ1')
# theta2 = Parameter('θ2')
# theta3 = Parameter('θ3')
theta0 = Parameter('theta0')
theta1 = Parameter('theta1')
theta2 = Parameter('theta2')
theta3 = Parameter('theta3')

input0 = Parameter('input0')
input1 = Parameter('input1')

# XOR truth table
training_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

# def create_xor_circuit(t0, t1, t2, t3, input0, input1):
def create_xor_circuit():
    qc = QuantumCircuit(3, 1)
    if input0 == 1:
        qc.x(0)
    if input1 == 1:
        qc.x(1)
    qc.rx(theta0, 0)
    qc.rx(theta1, 1)
    qc.cx(0, 1)
    qc.rz(theta2, 1)
    qc.cx(1, 2)
    qc.rx(theta3, 2)
    qc.measure(2, 0)
    # print(qc.draw("text"))  # Or "mpl" for matplotlib diagram
    return qc

def cost_function(predictions, targets):
    epsilon = 1e-10
    return -np.mean([t*np.log(p+epsilon) + (1-t)*np.log(1-p+epsilon) for p, t in zip(predictions, targets)])

def simulate(params, backend, shots=1024):
    predictions = []
    for inputs, _ in training_data:
        # qc = create_xor_circuit(*params, *inputs)
        # qc = qc.bind_parameters({theta0: params[0], theta1: params[1], theta2: params[2], theta3: params[3]})
        # from qiskit import execute
        # job = execute(qc, backend=backend, shots=shots)
        qc = create_xor_circuit()
        qc.assign_parameters({theta0: params[0], theta1: params[1], theta2: params[2], theta3: params[3]}, inplace=True)
        job = backend.run(qc, shots=shots)
        # print(qc.decompose())
        result = job.result()
        counts = result.get_counts()
        p1 = counts.get('1', 0) / shots
        predictions.append(p1)
    
    return predictions

def train_qnn(epochs=100, lr=0.3):
    # backend = Aer.get_backend('qasm_simulator')
    backend = AerSimulator()

    params = np.random.uniform(0, 2*np.pi, 4)
    loss_trace = []

    for epoch in range(epochs):
        preds = simulate(params, backend)
        targets = [t for _, t in training_data]
        loss = cost_function(preds, targets)
        loss_trace.append(loss)

        grads = np.zeros(4)
        eps = 1e-2
        for i in range(4):
            plus = params.copy()
            minus = params.copy()
            plus[i] += eps
            minus[i] -= eps
            grads[i] = (cost_function(simulate(plus, backend), targets) - cost_function(simulate(minus, backend), targets)) / (2 * eps)

        params -= lr * grads

    return params, loss_trace

# Train and plot
trained_params, loss_trace = train_qnn()
# plt.plot(loss_trace)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("QNN Training on XOR")
# plt.grid(True)
# plt.show()

print("Trained parameters:", trained_params)
