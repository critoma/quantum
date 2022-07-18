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

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import org.redfx.strange.Program;
import org.redfx.strange.Qubit;
import org.redfx.strange.Result;
import org.redfx.strange.Step;
import org.redfx.strange.gate.Identity;
import org.redfx.strange.gate.X;

public class InitialStateTests extends BaseGateTests {

    @Test
    public void dontInitialize() {
        Program p = new Program(1, new Step(new Identity(0)));
        Result res = runProgram(p);
        Qubit[] qubits = res.getQubits();
        assertEquals(0, qubits[0].measure());
    }

    @Test
    public void noStep() {
        Program p = new Program(0);
        Result res = runProgram(p);
        assertEquals(0, res.getQubits().length);
    }

    @Test
    public void initialize0() {
        Program p = new Program(1, new Step(new Identity(0)));
        p.initializeQubit(0, 1);
        Result res = runProgram(p);
        Qubit[] qubits = res.getQubits();
        assertEquals(0, qubits[0].measure());
    }

    @Test
    public void initialize1() {
        Program p = new Program(1, new Step(new Identity(0)));
        p.initializeQubit(0, 0);
        Result res = runProgram(p);
        Qubit[] qubits = res.getQubits();
        assertEquals(1, qubits[0].measure());
    }

    @Test
    public void initialize1Not() {
        Program p = new Program(1, new Step(new X(0)));
        p.initializeQubit(0, 0);
        Result res = runProgram(p);
        Qubit[] qubits = res.getQubits();
        assertEquals(0, qubits[0].measure());
    }

    // 2 qubits, initally 1. Flip the first one, keep the second
    @Test
    public void TwoQubitinitialize1Not() {
        Program p = new Program(2, new Step(new X(0)));
        p.initializeQubit(0, 0);
        p.initializeQubit(1, 0);
        Result res = runProgram(p);
        Qubit[] qubits = res.getQubits();
        assertEquals(0, qubits[0].measure());
        assertEquals(1, qubits[1].measure());
    }


    // 1 qubit, 50% change 0 or 1, should at least be once 0 and once 1 in 100 tries
    @Test
    public void initializeFifty() {
        Program p = new Program(1, new Step(new Identity(0)));
        double sq = Math.sqrt(.5);
        p.initializeQubit(0, sq);
        int cnt = 0;
        boolean got0 = false;
        boolean got1 = false;
        int i = 0;
        while (i < 100 && !(got0 && got1)) {
            Result res = runProgram(p);
            Qubit[] qubits = res.getQubits();
            int q0 = qubits[0].measure();
            if (q0 == 0) {
                got0 = true;
            }
            if (q0 == 1) {
                got1 = true;
            }
            i++;
        }
        assertTrue(got0);
        assertTrue(got1);
    }

    // 1 qubit, 10% change 0, should be more 1 than 0 in 100 tries
    @Test
    public void initializeLowHigh() {
        Program p = new Program(1, new Step(new Identity(0)));
        double sq = Math.sqrt(.1);
        p.initializeQubit(0, sq);

        for (int c = 0; c < 10; c++){
            int cnt0 = 0;
            int cnt1 = 1;
            int i = 0;
            while (i < 100) {
                Result res = runProgram(p);
                Qubit[] qubits = res.getQubits();
                int q0 = qubits[0].measure();
                if (q0 == 0) {
                    cnt0++;
                }
                if (q0 == 1) {
                    cnt1++;
                }
                i++;
            }
            assertTrue(cnt1 > cnt0);
        }
    }
}
