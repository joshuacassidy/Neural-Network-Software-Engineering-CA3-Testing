package com.josh.SoftwareEngineeringCA3;

/**
 * Created by Josh on 08/04/2018.
 */
public class Main {

    public static void main(String[] args) {
            Agent agent = new Agent();
            agent.learn();
            System.out.println(agent.makePrediction(new double[]{0, 0}));
            System.out.println(agent.makePrediction(new double[]{0, 1}));
            System.out.println(agent.makePrediction(new double[]{1, 0}));
            System.out.println(agent.makePrediction(new double[]{1, 1}));
    }

}

