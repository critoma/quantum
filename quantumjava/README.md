# Quantum Computing samples in Java

This repository contains the source code for the samples discussed in
<a href="https://www.manning.com/books/quantum-computing-for-java-developers?a_aid=quantu
mjava&a_bid=e5166ab9">Quantum Computing for Developers</a>

<a href="https://www.manning.com/books/quantum-computing-for-java-developers?a_aid=quantumjava&a_bid=e5166ab9"><img align="right" src="./resources/qc.png" alt="Quantum Computing for Developers"/></a>

It is based on the <a href="https://github.com/redfx-quantum/strange">Strange</a> Quantum Simulator, which provides an execution environment for
quantum algorithms in Java.  
Some samples use a UI, e.g. to visualize the quantum circuit. In
that case, <a href="https://github.com/redfx-quantum/strangefx">StrangeFX</a>
is used, which is a JavaFX-based framework that allows the visualisation
of quantum circuits.

The samples in this repository correspond to the chapters in the book.
Don't worry if you don't have the book, you can still run the
samples.

## Chapter 2: HelloWorld, Quantum Computing
<a href="./ch02/hellostrange">HelloStrange</a>

## Chapter 3: Qubits and Quantum Gates, the basic units in Quantum Computing
<a href="./ch03/paulix">Pauli X</a>  
<a href="./ch03/paulixui">Pauli X with user interface</a>

## Chapter 4: Superposition
<a href="./ch04/hadamard">Hadamard</a>  

## Chapter 5: Entanglement
<a href="./ch05/classiccoin">Classic coins</a>  
<a href="./ch05/quantumcoin">Quantum coins</a>  
<a href="./ch05/cnot">CNot gate</a>  
<a href="./ch05/bellstate">Bell state</a>  

## Chapter 6: Quantum networking, the basics
<a href="./ch06/classic">Classic network</a>  
<a href="./ch06/classiccopy">No-cloning theorem</a>  
<a href="./ch06/teleport">Quantum Teleportation</a>  
<a href="./ch06/repeater">Quantum Repeater</a>  

## Chapter 7: Our HelloWorld explained
<a href="./ch07/randombit">RandomBit</a>  
<a href="./ch07/randombitdebug">RandomBit with debug</a>  
<a href="./ch07/add1">Quantum Adder</a>  
<a href="./ch07/add2">Quantum Adder with carry bit</a>  

## Chapter 8: Secure Communication using quantum computing
<a href="./ch08/naive">A first (naive) approach</a>  
<a href="./ch08/haha">Applying 2 Hadamard gates</a>  
<a href="./ch08/superposition">Using superposition</a>  
<a href="./ch08/guess">Guess the possibilities</a>  
<a href="./ch08/bb84">QKD in Java</a>  

## Chapter 9: Deutsch-Jozsa algorithm
<a href="./ch09/function">Constant and balanced functions</a>  
<a href="./ch09/reversibleX">Reversible gates</a>  
<a href="./ch09/oracle">Quantum oracle</a>  
<a href="./ch09/applyoracle">Applying a Quantum oracle</a>  
<a href="./ch09/deutsch">Deutsch algorithm</a>  
<a href="./ch09/deutschjozsa">Deutsch Jozsa algorithm</a>  

## Chapter 10: Grover's Search Algorithm
<a href="./ch10/classicsearch">A classic search function</a>  
<a href="./ch10/quantumsearch">Quantum search</a>  
<a href="./ch10/grover">The algorithm behind Grover's search</a>  
<a href="./ch10/stepbystepgrover">The algorithm, step by step</a>  
<a href="./ch10/groveroracle">The quantum Oracle in Grover's search</a>  

## Chapter 11: Shor's Algorithm
<a href="./ch11/classicfactor">A classic factoring approach</a>  
<a href="./ch11/semiclassicfactor">A classic implementation of a quantum factoringapproach</a>  
<a href="./ch11/quantumfactor">A quantum factoring approach</a>  

