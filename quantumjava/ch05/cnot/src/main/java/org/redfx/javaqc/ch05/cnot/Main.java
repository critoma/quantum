package org.redfx.javaqc.ch05.cnot;

import org.redfx.strange.*;
import org.redfx.strange.gate.*;
import org.redfx.strange.local.*;
import org.redfx.strangefx.render.*;

public class Main {

    public static void main(String[] args) {
        run00();
        run01();
        run10();
        run11();
    }

    private static void run00() {
        QuantumExecutionEnvironment simulator = new SimpleQuantumExecutionEnvironment();
        Program program = new Program(2);
        Step step1 = new Step();
        step1.addGate(new Cnot(0,1));
        program.addStep(step1);
        Result result = simulator.runProgram(program);
        Qubit[] qubits = result.getQubits();
        Qubit q0 = qubits[0];
        Qubit q1 = qubits[1];
        int v0 = q0.measure();
        int v1 = q1.measure();
        System.out.println("IN = |00>\tOUT= |"+v1+""+v0+">");
        Renderer.renderProgram(program);
        Renderer.showProbabilities(program, 1000);
    }  

    private static void run01() {
        QuantumExecutionEnvironment simulator = new SimpleQuantumExecutionEnvironment();
        Program program = new Program(2);
        Step step1 = new Step();
        step1.addGate(new X(0));
        program.addStep(step1);
        Step step2 = new Step();
        step2.addGate(new Cnot(0,1));
        program.addStep(step2);
        Result result = simulator.runProgram(program);
        Qubit[] qubits = result.getQubits();
        Qubit q0 = qubits[0];
        Qubit q1 = qubits[1];
        int v0 = q0.measure();
        int v1 = q1.measure();
        System.out.println("IN = |01>\tOUT= |"+v1+""+v0+">");
        Renderer.renderProgram(program);
        Renderer.showProbabilities(program, 1000);
    }  

    private static void run10() {
        QuantumExecutionEnvironment simulator = new SimpleQuantumExecutionEnvironment();
        Program program = new Program(2);
        Step step1 = new Step();
        step1.addGate(new X(1));
        program.addStep(step1);
        Step step2 = new Step();
        step2.addGate(new Cnot(0,1));
        program.addStep(step2);
        Result result = simulator.runProgram(program);
        Qubit[] qubits = result.getQubits();
        Qubit q0 = qubits[0];
        Qubit q1 = qubits[1];
        int v0 = q0.measure();
        int v1 = q1.measure();
        System.out.println("IN = |10>\tOUT= |"+v1+""+v0+">");
        Renderer.renderProgram(program);
        Renderer.showProbabilities(program, 1000);
    }  

    private static void run11() {
        QuantumExecutionEnvironment simulator = new SimpleQuantumExecutionEnvironment();
        Program program = new Program(2);
        Step step1 = new Step();
        step1.addGate(new X(0));
        step1.addGate(new X(1));
        program.addStep(step1);
        Step step2 = new Step();
        step2.addGate(new Cnot(0,1));
        program.addStep(step2);
        Result result = simulator.runProgram(program);
        Qubit[] qubits = result.getQubits();
        Qubit q0 = qubits[0];
        Qubit q1 = qubits[1];
        int v0 = q0.measure();
        int v1 = q1.measure();
        System.out.println("IN = |11>\tOUT= |"+v1+""+v0+">");
        Renderer.renderProgram(program);
        Renderer.showProbabilities(program, 1000);
    }  

}
