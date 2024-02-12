package org.redfx.javaqc.ch10.quantumsearch;

import org.redfx.strange.algorithm.Classic;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.Function;

// only for didactic display:
import java.util.Arrays;
import org.redfx.strange.Complex;

import org.redfx.strange.*;
import org.redfx.strange.gate.*;

public class Main {

    public static void main (String[] args) {
        System.err.println("Hello, quantum search");
        Main main = new Main();
        main.quantumSearch();
    }

    void quantumSearch() {
        Function<Person, Integer> f29Mexico
                = (Person p) -> ((p.getAge() == 29) && (p.getCountry().equals("Mexico"))) ? 1 : 0;
        List<Person> persons = prepareDatabase();
        for (int i = 0; i < 10; i++) {
            // List<Person> persons = prepareDatabase();
            Collections.shuffle(persons);
            // Quantum Search 
	    // https://github.com/critoma/quantum/blob/main/strange/src/main/java/org/redfx/strange/algorithm/Classic.java 
            // https://github.com/redfx-quantum/strange/blob/master/src/main/java/org/redfx/strange/algorithm/Classic.java): 
            // Person target = Classic.search(persons, f29Mexico);
            // Non-quantum search: // Person target = findPersonByFunction(persons, f29Mexico);
            // System.out.println("Result of function Search = " + target.getName());
            System.out.println("### Quantum:");
            Person target = Classic.search(persons, f29Mexico);
            System.out.println("Result of 'Classic' aka Quantum Function Search = " + target.getName());

            System.out.println("--- NON-Quantum:");
            Person targetNQ = findPersonByFunction(persons, f29Mexico);
            System.out.println("Result of function non-quantum Search = " + targetNQ.getName());
        }
        // didactic oracles display - optional:
        displayOracles(persons, f29Mexico);
    }

    Person findPersonByFunction(List<Person> persons, Function<Person, Integer> function) {
        boolean found = false;
        int idx = 0;
        while (!found && (idx<persons.size())) {
            Person target = persons.get(idx++);
            if (function.apply(target) == 1) {
                found = true;
            }
        }
        System.out.println("Got result after "+idx+" tries");
        return persons.get(idx-1); 
    }
    
    List<Person> prepareDatabase() {
        List<Person> persons = new LinkedList<>();
        persons.add(new Person("Alice", 42, "USA"));
        persons.add(new Person("Bob", 36, "UK"));
        persons.add(new Person("Eve", 85, "Australia"));
        persons.add(new Person("Nikos", 18, "Greece"));
        persons.add(new Person("Albert", 29, "Mexico"));
        persons.add(new Person("Andreea", 29, "Romania"));
        persons.add(new Person("Marie", 15, "Ukraine"));
        persons.add(new Person("Janice", 50, "Israel"));
        persons.add(new Person("Jake", 41, "USA"));    
        persons.add(new Person("Andrew", 58, "UK"));
        persons.add(new Person("Nicole", 19, "Australia"));
        persons.add(new Person("Elena", 21, "Greece"));
        persons.add(new Person("Gabriela", 22, "Spain"));
        persons.add(new Person("Alexandru", 28, "Canada"));
        persons.add(new Person("Maria", 17, "Italy"));
        persons.add(new Person("Naomi", 32, "France"));
        return persons;
    }


    // #############################################
    // these are OPTIONAL - JUST for didactic debug:
    public static<T> void displayOracles(List<T> list, Function<T, Integer> function) {
        int size = list.size();
        int n = (int) Math.ceil((Math.log(size)/Math.log(2)));
        int N = 1 << n;
        double cnt = Math.PI*Math.sqrt(N)/4;

        Oracle oracle = createGroverOracleDisplay(n, list, function);
        oracle.setCaption("O");
        Complex[][] oracleMatrix = oracle.getMatrix();

        Complex[][] dif = createDiffMatrixDisplay(n);
        Oracle difOracle = new Oracle(dif);
        difOracle.setCaption("D");

        System.out.println("Oracle matrix:");
        for (Complex[] row : oracleMatrix)
            System.out.println(Arrays.toString(row));
        System.out.println("Dif Oracle matrix:");
        for (Complex[] row : dif)
            System.out.println(Arrays.toString(row));
    }
    
    private static<T> Oracle createGroverOracleDisplay(int n, List<T> list, Function<T, Integer> function) {
        int N = 1 << n;
        int listSize = list.size();
        Complex[][] matrix = new Complex[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                matrix[i][j] = (i!=j) ? Complex.ZERO : 
                        ((i >= listSize) || function.apply(list.get(i)) == 0 )? 
                        Complex.ONE : Complex.ONE.mul(-1);
            }
        }
        return new Oracle(matrix);
    }

    private static Complex[][] createDiffMatrixDisplay(int n) {
        int N = 1<<n;
        Gate g = new Hadamard(0);
        Complex[][] matrix = g.getMatrix();
        Complex[][] h2 = matrix;
        for (int i = 1; i < n; i++) {
            h2 = Complex.tensor(h2, matrix);
        }
        Complex[][] I2 = new Complex[N][N];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <N;j++) {
                if (i!=j) {
                    I2[i][j] = Complex.ZERO;
                } else {
                    I2[i][j] = Complex.ONE;
                }
            }
        }
        I2[0][0] = Complex.ONE.mul(-1);
        int nd = n<<1;

        Complex[][] inter1 = Complex.mmul(h2,I2);
        Complex[][] dif = Complex.mmul(inter1, h2);
        return dif;
    }
} // end class
