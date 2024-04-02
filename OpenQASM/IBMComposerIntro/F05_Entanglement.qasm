OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];

measure q[0] -> c[0];
measure q[1] -> c[1];

//qreg a[1];
//qreg b[1];
//creg ac[1];
//creg bc[1];

//h a[0];
//cx a[0],b[0];
//measure a[0] -> ac[0];
//measure b[0] -> bc[0];

