OPENQASM 2.0;
include "qelib1.inc";

// Run this sample in the IBM Q Experience Circuit Composer
// at https://quantum-computing.ibm.com/composer

// This sample generates a single random bit.

qreg q[1];
creg c[1];

h q[0];
measure q[0] -> c[0];
