OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[2];

// adding q[0]=1 + q[1]=1 
// => q[2] - Carry 
// => 10 in q[3] -> c[0], q[4] -> c[1] 
x q[0];
x q[1];

cx q[0], q[3];
cx q[1], q[3];
cx q[2], q[3];
barrier q;
ccx q[0], q[1], q[4];
ccx q[0], q[2], q[4];
ccx q[1], q[2], q[4];
barrier q;
measure q[3] -> c[0];
measure q[4] -> c[1];
