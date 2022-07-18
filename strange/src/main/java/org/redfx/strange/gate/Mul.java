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
package org.redfx.strange.gate;

import org.redfx.strange.Block;
import org.redfx.strange.BlockGate;
import org.redfx.strange.Step;
import org.redfx.strange.local.Computations;

/**
 * <p>Mul class.</p>
 *
 * @author johan
 * @version $Id: $Id
 */
public class Mul extends BlockGate<Mul> {

    final Block block;
    
    /**
     * Multiply the qubit in the x register with an integer mul
     *
     * @param x0 start idx x register
     * @param x1 end idx x register
     * @param mul
     * x_0 ----- y_0 + x_0
     * x_1 ----- y+1 + x_1
     * y_0 ----- 0
     * y_1 ----- 0
     */
    public Mul(int x0, int x1, int mul) {
        super();
        setIndex(x0);
        this.idx = x0;
        this.block = createBlock(x0, x1,mul);
        setBlock(block);
       
    }
    
    /**
     * <p>createBlock.</p>
     *
     * @param y0 a int
     * @param y1 a int
     * @param mul a int
     * @return a {@link org.redfx.strange.Block} object
     */
    public Block createBlock(int y0, int y1, int mul) {
        int x0 = 0;
        int x1 = y1-y0;
        int size = 1 + x1-x0;
        int dim = 1 << size;
        Block answer = new Block("Mul", 2 * size);

        for (int i = 0; i < mul; i++) {
            Add add = new Add(x0, x1, x1+1, x1 + size);
            answer.addStep(new Step(add));
        }

        for (int i = x0; i < x1+1; i++) {
            answer.addStep(new Step (new Swap(i, i + size)));
        }

        int invsteps = Computations.getInverseModulus(mul,dim);
        for (int i = 0; i < invsteps; i++) {
            Add add = new Add(x0, x1, x1+1, x1 + size).inverse();
            answer.addStep(new Step(add));
        }
        return answer;
    }
  
}
