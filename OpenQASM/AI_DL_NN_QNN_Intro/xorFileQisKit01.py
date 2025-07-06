from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def get_bits_from_file(file_path):
    # Open the file in binary mode.
    with open(file_path, 'rb') as f:
        data = f.read()

    # Convert each byte to an 8-bit binary string,
    # then convert each string character ('0' or '1') to an integer.
    bits = []
    for byte in data:
        # Format the byte as an 8-bit binary string.
        binary_str = format(byte, '08b')
        bits.extend(int(bit) for bit in binary_str)

    return bits

def classical_xor_encrypt(message_bits, key_bits):
    # Ensure both lists are same length
    assert len(message_bits) == len(key_bits), "Message and key must be same length"

    encrypted_bits = []

    for i in range(len(message_bits)):
        # Create 2 qubits: 0 - message bit, 1 - key bit
        qc = QuantumCircuit(2, 1)

        if message_bits[i] == 1:
            qc.x(0)  # Set qubit 0 to |1⟩
        if key_bits[i] == 1:
            qc.x(1)  # Set qubit 1 to |1⟩

        qc.cx(1, 0)       # XOR: message = message ⊕ key (CNOT gate)
        qc.measure(0, 0)  # Measure message bit

        # backend = Aer.get_backend('qasm_simulator')
        # result = execute(qc, backend, shots=1, memory=True).result()
        result = AerSimulator().run(qc, shots=1, memory=True).result()
        measured = int(result.get_memory()[0])  # '0' or '1'

        encrypted_bits.append(measured)

    return encrypted_bits

# === Example usage ===
if __name__ == "__main__":
    # Message: 1011
    # message = [1, 0, 1, 1]
    # key     = [0, 1, 0, 1]

    file_path = 'myfile.txt'  # Replace with your file name/path.
    bits = get_bits_from_file(file_path)
    print("Bits from the file:", bits)
    message = bits

    key_file_path = 'mykeyfile.txt'  # Replace with your file name/path.
    keybits = get_bits_from_file(key_file_path)
    print("Bits from the file:", keybits)
    key = keybits

    encrypted = classical_xor_encrypt(message, key)
    print("Encrypted: ", encrypted)

    # Decrypt (same operation with same key)
    decrypted = classical_xor_encrypt(encrypted, key)
    print("Decrypted: ", decrypted)
