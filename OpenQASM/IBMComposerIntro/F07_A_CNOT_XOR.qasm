
OPENQASM 2.0;
include "qelib1.inc";

qreg input1[2]; // Define quantum register for first input array
qreg input2[2]; // Define quantum register for second input array
// qreg output[2]; // Define quantum register for output array

creg c[2];

// Apply XOR gate between corresponding elements of input1 and input2, store result in output
//for (i = 0; i < 4; i++) {
    //cx input1[i], input2[i]; // Apply CNOT gate with input1 as control and input2 as target
    //measure input2[i] -> output[i]; // Measure target qubit and store result in output array
//}

x input1[0];
x input1[1];
x input2[0];
cx input1[0], input2[0];
cx input1[1], input2[1];
measure input2[0] -> c[0];
measure input2[1] -> c[1];