// Programming Quantum Computers
//   by Eric Johnston, Nic Harrigan and Mercedes Gimeno-Segovia
//   O'Reilly Media
//
// More samples like this can be found at http://oreilly-qc.github.io
//
// A complete set of all OpenQASM samples (including this one) can be found at
//  https://github.com/oreilly-qc/oreilly-qc.github.io/tree/master/samples/OpenQASM
//
// OpenQASM code was generated with the help of Qiskit (http://qiskit.org) 
// A complete set of all Qiskit samples (including this one) can be found at
//  https://github.com/oreilly-qc/oreilly-qc.github.io/tree/master/samples/Qiskit

// Run this sample in the IBM Q Experience Circuit Composer
// at https://quantum-computing.ibm.com

// Example 3-2: Entangled Qubits

OPENQASM 2.0;
include "qelib1.inc";

qreg a[1];
qreg b[1];
creg ac[1];
creg bc[1];

h a[0];
cx a[0],b[0];
measure a[0] -> ac[0];
measure b[0] -> bc[0];

/*
Entagled of 3 qubits - GHZ

OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

reset q[0];
reset q[1];
h q[0];
cx q[0],q[1];
reset q[2];
cx q[0],q[2];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
*/
