# https://docs.quantum.ibm.com/guides/install-qiskit
# https://docs.quantum.ibm.com/guides/interoperate-qiskit-qasm3
# https://docs.quantum.ibm.com/guides/setup-channel
# https://docs.quantum.ibm.com/guides/local-testing-mode
# https://docs.quantum.ibm.com/migration-guides/local-simulators
# https://docs.quantum.ibm.com/guides/visualize-results
# https://docs.quantum.ibm.com/api/qiskit/qasm3
# https://docs.quantum.ibm.com/guides/hello-world

# if-then-else in quantum circuits but for specific values:
# https://www.researchgate.net/publication/334208340 
# https://github.com/openqasm/openqasm/blob/main/examples/
# https://github.com/openqasm/openqasm/blob/main/examples/ipe.qasm 
# https://www.researchgate.net/publication/334208340_Synthesis_and_Optimization_by_Quantum_Circuit_Description_Language#pf9
# https://en.wikipedia.org/wiki/Fredkin_gate

# https://docs.quantum.ibm.com/guides/get-started-with-primitives
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/index.html
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02_neural_network_classifier_and_regressor.html
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html
# https://github.com/qiskit-community/qiskit-machine-learning/tree/main/test/neural_networks

# Physical Trasistors NN (for AI Hardware CPUs) 
# How to bild a neuron: https://www.youtube.com/watch?v=Uhuo9ketdhg
# How to build a synapse: https://www.youtube.com/watch?v=GBvF-Vv2y7c
# EasyEDA: https://www.youtube.com/watch?v=9gdtav5fx88&list=PL52rQn9fkWwNQgQW_ncxdilw7MB8veF3g | https://easyeda.com/editor

# AI NN on GPU: https://github.com/chriskinzel/OpenCL-NeuralNetwork/tree/master

# - Single-chip photonic deep neural network with forward-only training:
# https://www.nature.com/articles/s41566-024-01567-z
# - An on-chip photonic deep neural network for image classification:
# https://www.nature.com/articles/s41586-022-04714-0?fromPaywallRec=false
# https://www.nature.com/articles/s41467-024-55139-4
# QNN: https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/11_quantum_convolutional_neural_networks.html


# mkdir -p /home/cristian_toma/qasm3.0/qiskit
cd /home/cristian_toma/qasm3.0/qiskit


python3 -m venv .venv
source .venv/bin/activate
pip install qiskit
pip install qiskit-ibm-runtime
pip install qiskit-qasm3-import
pip install qiskit-aer
pip install qiskit-algorithms
pip install qiskit-machine-learning
pip install scikit-learn

pip3 install qiskit qiskit-ibm-runtime qiskit-qasm3-import qiskit-aer qiskit-algorithms qiskit-machine-learning scikit-learn


nano test_bell_01_AerSimu.py
####################
from qiskit import __version__
print(__version__)

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Operator
from qiskit_aer import AerSimulator
# from qiskit.visualization import plot_histogram

X = QuantumRegister(1, "X")
Y = QuantumRegister(1, "Y")
A = ClassicalRegister(1, "A")
B = ClassicalRegister(1, "B")

circuit = QuantumCircuit(Y, X, B, A)
circuit.barrier()
circuit.h(Y)
circuit.cx(Y, X)
circuit.barrier()
circuit.measure(Y, B)
circuit.measure(X, A)
print(circuit)
# display(circuit.draw(output="mpl"))
result = AerSimulator().run(circuit).result()
statistics = result.get_counts()
print(statistics)

print(result)
# print(result.data)

# display(plot_histogram(statistics))



nano test_bell_01_FakeManila.py
####################
from qiskit import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit import QuantumRegister, ClassicalRegister
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
 
# Bell Circuit
'''
qr = qiskit.QuantumRegister(3)
cr = qiskit.ClassicalRegister(3)
qc = qiskit.QuantumCircuit(qr, cr)
qc.reset(0); qc.reset(1); qc.reset(2);
qc.barrier()
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
# qc.measure([0, 1, 2], [0, 1, 2])
qc.measure_all()
print(qc)
'''

X = QuantumRegister(1, "X")
Y = QuantumRegister(1, "Y")
A = ClassicalRegister(1, "A")
B = ClassicalRegister(1, "B")

quantumCircuit = QuantumCircuit(Y, X, B, A)
quantumCircuit.barrier()
quantumCircuit.h(Y)
quantumCircuit.cx(Y, X)
quantumCircuit.barrier()
quantumCircuit.measure(Y, B)
quantumCircuit.measure(X, A)
print(quantumCircuit)

# Run the sampler job locally using FakeManilaV2
fake_manila = FakeManilaV2()
pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
isa_qc = pm.run(quantumCircuit)
 
# You can use a fixed seed to get fixed results.
options = {"simulator": {"seed_simulator": 42}}
sampler = Sampler(mode=fake_manila, options=options)
sampler.options.default_shots = 10

result = sampler.run([isa_qc]).result()
print(result)
print(result[0].data)
# print(result[0].data.meas.get_counts())
print(result[0].data.A.num_shots, result[0].data.B.num_shots)
print(result[0].data.A.array)
print(result[0].data.B.array)

# Extract the final statevector
# statevector = result[0].data.meas.get_statevector()
# Plot the Bloch sphere
# plot_bloch_multivector(statevector)
# print(statevector)


nano test_bell_01_QASM3_onAerAndFake_wTranspiler.py
####################
import qiskit
import qiskit.qasm3
from qiskit import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qiskit import __version__
print(__version__)
 
program = """
    OPENQASM 3.0;
    include "stdgates.inc";

    qreg q[3];
    creg c[3];

    reset q[0];
    reset q[1];
    reset q[2];
    barrier q;
    h q[0];
    cx q[0],q[1];
    // reset q[2];
    cx q[0],q[2];
    barrier q;
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """
quantumCircuit3 = qiskit.qasm3.loads(program)
# quantumCircuit3.measure_all()
# quantumCircuit.draw("mpl")
print(quantumCircuit3)

from qiskit_aer import AerSimulator
resultA = AerSimulator().run(quantumCircuit3).result()
statisticsA = resultA.get_counts()
print(statisticsA)

#print(resultA)
#print(resultA.data)

print("FakeManilaV2")

# Run the sampler job locally using FakeManilaV2
fake_manila = FakeManilaV2()
pm = generate_preset_pass_manager(backend=fake_manila, optimization_level=1)
isa_qc = pm.run(quantumCircuit3)
 
# You can use a fixed seed to get fixed results.
options = {"simulator": {"seed_simulator": 42}}
sampler = Sampler(mode=fake_manila, options=options)
sampler.options.default_shots = 10
print(isa_qc)
result = sampler.run([isa_qc]).result()
print(result)

print(result[0].data)
print(result[0].data.c.num_shots)
print(result[0].data.c.array)

#from qiskit.primitives import BitArray
import numpy as np
print( np.unpackbits( ( np.take((result[0].data.c.array[7]), 0, 0) ) ) )


python3 test_bell_01_AerSimu.py
python3 test_bell_01_FakeManila.py

# https://github.com/openqasm/openqasm/tree/main
wget https://raw.githubusercontent.com/openqasm/openqasm/refs/heads/main/examples/stdgates.inc

python3 test_bell_01_QASM3_onAerAndFake_wTranspiler.py



#################
nano test_display_02_QASM3_Circuit.py

#######
import qiskit.qasm3
 
program = """
    OPENQASM 3.0;
    include "stdgates.inc";
 
    input float[64] a;
    qubit[3] q;
    bit[2] mid;
    bit[3] out;
 
    let aliased = q[0:1];
 
    gate my_gate(a) c, t {
      gphase(a / 2);
      ry(a) c;
      cx c, t;
    }
    gate my_phase(a) c {
      ctrl @ inv @ gphase(a) c;
    }
 
    my_gate(a * 2) aliased[0], q[{1, 2}][0];
    measure q[0] -> mid[0];
    measure q[1] -> mid[1];
 
    while (mid == "00") {
      reset q[0];
      reset q[1];
      my_gate(a) q[0], q[1];
      my_phase(a - pi/2) q[1];
      mid[0] = measure q[0];
      mid[1] = measure q[1];
    }
 
    if (mid[0]) {
      let inner_alias = q[{0, 1}];
      reset inner_alias;
    }
 
    out = measure q;
"""
# quantumCircuit = qiskit.qasm3.loads(program)
# quantumCircuit.draw("mpl")

file = open("test_display_02_QASM3.qasm", "r")
contentOfProgram = file.read()
print(contentOfProgram)
file.close()
quantumCircuit = qiskit.qasm3.loads(contentOfProgram)
print(quantumCircuit)

###
python3 test_display_02_QASM3_Circuit.py


nano testAI_ML_DL_QNN_XoR_01.py
###
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

###
python3 testAI_ML_DL_QNN_XoR_wCalcParams_01.py


#######
nano testAI_ML_DL_QNN_XoR_wCalcParams_01.py
###
# from qiskit import QuantumCircuit, Aer, execute
# import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Trained parameters (use your actual values here)
# params = [1.57, 3.14, 2.71, 0.78]
params = [-0.94985782, 2.70640583, 3.19704226, 1.95428913]
theta0, theta1, theta2, theta3 = params

# Input combinations for XOR
inputs = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1)
]

# Quantum simulator
backend = AerSimulator()

# Run simulations
results = {}
for x0, x1 in inputs:
    qc = QuantumCircuit(3, 1)
    if x0:
        qc.x(0)
    if x1:
        qc.x(1)
    qc.rx(theta0, 0)
    qc.rx(theta1, 1)
    qc.cx(0, 1)
    qc.rz(theta2, 1)
    qc.cx(1, 2)
    qc.rx(theta3, 2)
    qc.measure(2, 0)

    print(qc.decompose())

    job = backend.run(qc, shots=1024)
    counts = job.result().get_counts()
    print(counts)
    p1 = counts.get("1", 0) / 1024
    results[(x0, x1)] = p1

# Print results
for k, v in results.items():
    print(f"Input {k} => Output ≈ {round(v)} (p1 = {v:.3f})")

####
python3 testAI_ML_DL_QNN_XoR_wCalcParams_01.py
# XOR results through a QNN / not SVC or VQC:
Input (0, 0) => Output ≈ 0 (p1 = 0.417)
Input (0, 1) => Output ≈ 1 (p1 = 0.567)
Input (1, 0) => Output ≈ 1 (p1 = 0.601)
Input (1, 1) => Output ≈ 0 (p1 = 0.397)





####
# https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.neural_networks.SamplerQNN.html
### 
#### Iris Dataset
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# https://onlinelibrary.wiley.com/doi/10.1111/j.1469-1809.1936.tb02137.x
# https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1469-1809.1936.tb02137.x

# https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
# https://en.wikipedia.org/wiki/Sepal#/media/File:Mature_flower_diagram.svg
# https://gist.github.com/curran/a08a1080b88344b0c8a7
# https://arxiv.org/pdf/2505.17756
# https://qiskit-community.github.io/qiskit-machine-learning/tutorials/02a_training_a_quantum_model_on_a_real_dataset.html

# XOR with QNN (we must replicate with transistors and octocouplers but adjusting "weights" from potentiometer)?:
# https://www.nature.com/articles/s41467-024-55139-4

# https://www.linkedin.com/in/cody-wabiszewski-38313731a/
# https://www.youtube.com/watch?v=9gdtav5fx88&list=PL52rQn9fkWwNQgQW_ncxdilw7MB8veF3g
# https://www.youtube.com/watch?v=Uhuo9ketdhg
# https://www.youtube.com/watch?v=GBvF-Vv2y7c

// 650 epochs
https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=4&seed=0.15031&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

// 1550 epochs
# https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=4,2&seed=0.15031&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

// 4550 epochs wrong
# https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=4,2,2&seed=0.15031&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false
// like in XOR:
# https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.01&regularizationRate=0&noise=0&networkShape=2,2&seed=0.99101&showTestData=true&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false 


