OPENQASM 2.0;
include "qelib1.inc";

qreg q[4];
creg c[4];

reset q[0];
reset q[1];
reset q[2];
reset q[3];
x q[0];
barrier q[0], q[1], q[2], q[3];
x q[0];
x q[1];
x q[2];
x q[3];
cx q[1], q[2];
cx q[2], q[1];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[2];
cx q[2], q[3];
cx q[0], q[3];
cx q[3], q[0];
cx q[0], q[3];
barrier q[0], q[1], q[2], q[3];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
measure q[3] -> c[3];

// @columns [0,0,0,0,1,2,3,3,3,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]