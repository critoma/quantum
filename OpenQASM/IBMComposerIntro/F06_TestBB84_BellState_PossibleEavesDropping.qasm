OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];

reset q[0];
h q[0]; // Alice send q0=0 and apply H
reset q[1]; 
cx q[0],q[1]; // Eve entangles q0=? with q1=?
h q[0]; // Bob receives q0 and apply H using Alice instructions
measure q[0] -> c[0]; // Bob measure one bit from the key
measure q[1] -> c[1]; // Eve is able to obtained Bob bit value?