/*-
 * #%L
 * Strange
 * %%
 * Copyright (C) 2020 Johan Vos
 * %%
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the Johan Vos nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 * #L%
 */
package org.redfx.strange.test;

import java.util.List;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.redfx.strange.Block;
import org.redfx.strange.ControlledBlockGate;
import org.redfx.strange.Program;
import org.redfx.strange.Qubit;
import org.redfx.strange.Result;
import org.redfx.strange.Step;
import org.redfx.strange.gate.Add;
import org.redfx.strange.gate.AddInteger;
import org.redfx.strange.gate.AddIntegerModulus;
import org.redfx.strange.gate.AddModulus;
import org.redfx.strange.gate.Cnot;
import org.redfx.strange.gate.MulModulus;
import org.redfx.strange.gate.Swap;
import org.redfx.strange.gate.X;
import org.redfx.strange.local.Computations;

/**
 *
 * @author johan
 */
public class SingleTest extends BaseGateTests {
           
     @Test
    public void multiplyModGate2x5mod3() { // 2 x 5 mod 3 = 1
        Program p = new Program(6);
        int mul = 5;
        int N = 3;
        Step prep = new Step();
        prep.addGates(new X(1)); // 2 in high register
        Step s = new Step(new MulModulus(0,1,mul, N));
        p.addStep(prep);
        p.addStep(s);
        Result result = runProgram(p);
        Qubit[] q = result.getQubits();
        for (int i = 0; i < 6; i++) {
            System.err.println("q["+i+"] = "+q[i].measure());
        }
        assertEquals(6, q.length);
        assertEquals(1, q[0].measure()); // q2,q1,q0,q3 should be clean
        assertEquals(0, q[1].measure());  
        assertEquals(0, q[2].measure());
        assertEquals(0, q[3].measure());
        assertEquals(0, q[4].measure()); // result in q4,q5,q6,q7
        assertEquals(0, q[5].measure());
     //   assertEquals(0, q[6].measure());  
    }
    
    // @Test
    public void multiplyModGate5x3mod6() { // 5 x 3 mod 6 = 3
        Program p = new Program(9);
        int mul = 5;
        int N = 6;
        Step prep = new Step();
        prep.addGates(new X(0), new X(1)); // 3 in high register
    //    Step s = new Step(new MulModulus(0,3,mul, N));
        p.addStep(prep);
        List<Step> steps = createBlock(0,3,mul,N);
        for (Step step : steps) {
            p.addStep(step);
        }
        Result result = runProgram(p);
        Qubit[] q = result.getQubits();
        for (int i = 0; i < 9; i++) {
            System.err.println("q["+i+"] = "+q[i].measure());
        }
        assertEquals(9, q.length);
        assertEquals(1, q[0].measure()); // q2,q1,q0,q3 should be clean
        assertEquals(1, q[1].measure());  
        assertEquals(0, q[2].measure());
        assertEquals(0, q[3].measure());
        assertEquals(0, q[4].measure()); // result in q4,q5,q6,q7
        assertEquals(0, q[5].measure());
        assertEquals(0, q[6].measure());  
        assertEquals(0, q[7].measure());  
        assertEquals(0, q[8].measure());  
    }
    
     public List<Step> createBlock(int y0, int y1, int mul, int mod) {
        int n = y1 - y0;
        System.err.println("Need to create block with mul = "+mul+" and mod = "+mod);
        int hash = 1000000 * y0 + 10000*y1+ 100*mul + mod;

        int x0 = y0;
        int x1 = y1-y0;
        int size = 1 + x1-x0;
        Block answer = new Block("MulModulus", 2 * size+1);
        for (int i = 0; i < n; i++) {
            int m = 1;
            for (int j = 0; j < 1 << i; j++) {
                m = m * mul % mod;
            }
            System.err.println("Create AddIntMod, x0 = "+x0+", x1 = "+x1+", m = " +m+", mod = "+mod);
            AddIntegerModulus add = new AddIntegerModulus(x0, x1, m, mod);
            System.err.println("Create CBG with idx = "+(n+1) +" and ctrl "+i);
            ControlledBlockGate cbg = new ControlledBlockGate(add, n+1, i);
            answer.addStep(new Step(cbg));
        }

        for (int i = x0; i < x1; i++) {
            answer.addStep(new Step (new Swap(i, i + size)));
        }

        int invsteps = Computations.getInverseModulus(mul,mod);
        for (int i = 0; i < invsteps; i++) {
                        int m = 1;
            for (int j = 0; j < 1 << i; j++) {
                m = m * mul % mod;
            }
            AddIntegerModulus add = new AddIntegerModulus(x0, x1, m, mod).inverse();
            ControlledBlockGate cbg = new ControlledBlockGate(add, n+1, i);
            answer.addStep(new Step(cbg));
        }
        System.err.println("Number of steps in mulblock: " + answer.getSteps().size());
        return answer.getSteps();
    }
  
    
   // @Test
    public void findPeriod7_15() { // 5 x 3 mod 6 = 3
        Program p = new Program(9);
        int mul = 7;
        int n = 3;
        Step prep = new Step();
        prep.addGates(new X(4), new X(5)); // 3 in high register
        Step s = new Step(new MulModulus(0,3,mul, n));
        p.addStep(prep);
        p.addStep(s);
        Result result = runProgram(p);
        Qubit[] q = result.getQubits();
        assertEquals(9, q.length);
        assertEquals(0, q[0].measure()); // q2,q1,q0,q3 should be clean
        assertEquals(0, q[1].measure());  
        assertEquals(0, q[2].measure());
        assertEquals(0, q[3].measure());
        assertEquals(1, q[4].measure()); // result in q4,q5,q6,q7
        assertEquals(1, q[5].measure());
        assertEquals(0, q[6].measure());  
        assertEquals(0, q[7].measure());  
        assertEquals(0, q[8].measure());  
    }
    
    
  public static void main(String[] args) {
      SingleTest st = new SingleTest();
  }
}
