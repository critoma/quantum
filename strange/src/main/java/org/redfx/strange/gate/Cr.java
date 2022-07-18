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

import org.redfx.strange.Complex;

/**
 * <p>Cr class.</p>
 *
 * @author johan
 * @version $Id: $Id
 */
public class Cr extends TwoQubitGate {
    
    private Complex[][] matrix =  new Complex[][]{
        {Complex.ONE,Complex.ZERO,Complex.ZERO,Complex.ZERO},
        {Complex.ZERO,Complex.ONE,Complex.ZERO,Complex.ZERO},
        {Complex.ZERO,Complex.ZERO,Complex.ONE,Complex.ZERO},
        {Complex.ZERO,Complex.ZERO,Complex.ZERO,Complex.ONE.mul(-1)}
    };
    private int pow = -1;
    
    /**
     * <p>Constructor for Cr.</p>
     */
    public Cr() {    
    }
    
    /**
     * Control-R gate
     *
     * @param a target qubit
     * @param b control qubit
     * @param exp exp
     */
    public Cr (int a, int b, double exp) {
        super(a,b);
        double ar = Math.cos(exp);
        double ai = Math.sin(exp);
        if (Math.abs(Math.PI -exp)  < 1e-6) {
            ar = -1;
            ai = 0;
        } else if (Math.abs(Math.PI/2 -exp)  < 1e-6) {
            ar = 0;
            ai = 1;
        }
        matrix =  new Complex[][]{
                {Complex.ONE,Complex.ZERO,Complex.ZERO,Complex.ZERO},
        {Complex.ZERO,Complex.ONE,Complex.ZERO,Complex.ZERO},
        {Complex.ZERO,Complex.ZERO,Complex.ONE,Complex.ZERO},
        {Complex.ZERO,Complex.ZERO,Complex.ZERO,new Complex(ar, ai)}        
        };
    }
    /**
     * <p>Constructor for Cr.</p>
     *
     * @param a a int
     * @param b a int
     * @param base a int
     * @param pow a int
     */
    public Cr(int a, int b, int base, int pow) {
        this(a,b, Math.PI*2/Math.pow(base, pow));
        this.pow = pow;
    }
    
    /** {@inheritDoc} */
    @Override
    public Complex[][] getMatrix() {
        return matrix;
    }
    
    /** {@inheritDoc} */
    @Override 
    public void setInverse(boolean inv) {
        if (inv) {
            Complex[][] m = getMatrix();
            this.matrix = Complex.conjugateTranspose(m);
        }
    }

    /** {@inheritDoc} */
    @Override public String getCaption() {
        return "Cr" + ((pow> -1)? Integer.toString(pow): "th");
    }
}
