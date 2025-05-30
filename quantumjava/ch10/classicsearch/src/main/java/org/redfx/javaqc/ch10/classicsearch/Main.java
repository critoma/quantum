package org.redfx.javaqc.ch10.classicsearch;

import java.util.Collections;
import java.util.LinkedList;
import java.util.List;
import java.util.function.ToIntFunction;

public class Main {

    public static void main (String[] args) {
        System.err.println("Hello, classical search");
        Main main = new Main();
        main.complexSearch();
        main.functionSearch();
    }

    void complexSearch() {
        for (int i = 0; i < 10; i++) {
            List<Person> persons = prepareDatabase();
            Collections.shuffle(persons);
            Person target = findPersonByAgeAndCountry(persons, 29, "Mexico");
            System.out.println("Result of complex search = " + target.getName());
        }
    }
    
    Person findPersonByAgeAndCountry(List<Person> persons, int age, String country) {
        boolean found = false;
        int idx = 0;
        while (!found && (idx<persons.size())) {
            Person target = persons.get(idx++);
            if ((target.getAge() == age) && (target.getCountry().equals(country))) {
                found = true;
            }
        }
        System.out.println("Got result after "+idx+" tries");
        return persons.get(idx-1);
/*
        return persons.stream()
                .filter(p -> {return (p.getAge() == age && p.getCountry().equals(country));})
                .findFirst().get();
*/
}
    
    void functionSearch() {
        ToIntFunction<Person> f29Mexico
                = (Person p) -> ((p.getAge() == 29) && (p.getCountry().equals("Mexico"))) ? 1 : 0;
        for (int i = 0; i < 10; i++) {
            List<Person> persons = prepareDatabase();
            Collections.shuffle(persons);
            Person target = findPersonByFunction(persons, f29Mexico);
            System.out.println("Result of function Search = " + target.getName());
        }
    }

    Person findPersonByFunction(List<Person> persons, ToIntFunction<Person> function) {
        boolean found = false;
        int idx = 0;
        while (!found && (idx<persons.size())) {
            Person target = persons.get(idx++);
            if (function.applyAsInt(target) == 1) {
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
        persons.add(new Person("Alexandra", 29, "Romania"));
        persons.add(new Person("Marie", 15, "Ukraine"));
        persons.add(new Person("Janice", 52, "Israel"));
        return persons;
    }

}

