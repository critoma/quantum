OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

// |110> --CCNOT--> |111>
// it is acting like: AND Gate
x q[0];
x q[1];
ccx q[0],q[1],q[2]; 
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];