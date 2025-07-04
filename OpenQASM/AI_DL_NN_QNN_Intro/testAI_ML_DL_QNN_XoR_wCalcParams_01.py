# from qiskit import QuantumCircuit, Aer, execute
# import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Trained parameters (use your actual values here)
# params = [1.57, 3.14, 2.71, 0.78]
params = [-0.94985782, 2.70640583, 3.19704226, 1.95428913]
# params = [ 2.06352855, 5.14823772, 10.6153703, -2.03095632]
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
    print("x0, x1 = ", x0, x1)
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
