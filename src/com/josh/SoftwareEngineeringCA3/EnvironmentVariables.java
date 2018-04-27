package com.josh.SoftwareEngineeringCA3;

public class EnvironmentVariables {


    private static EnvironmentVariables instance = new EnvironmentVariables();

    private EnvironmentVariables() {
    }

    private final double LEARNING_RATE = 0.3;
    private final double MOMENTUM = 0.6;
    private int NETWORK_TRAINING_ITERATIONS = 100000;
    private final int LAYER_TRAINING_ITERATIONS = 10;

    private double[][] trainingData = new double[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}};

    private double[] trainingResults = new double[] {0, 1, 1, 1};

    public double getMOMENTUM() {
        return MOMENTUM;
    }

    public double getLEARNING_RATE() {
        return LEARNING_RATE;
    }

    public int getNETWORK_TRAINING_ITERATIONS() {
        return NETWORK_TRAINING_ITERATIONS;
    }

    public int getLAYER_TRAINING_ITERATIONS() {
        return LAYER_TRAINING_ITERATIONS;
    }

    public double sigmoid(double x) {
        return 1 / (1+Math.exp(-x));
    }

    public double derivativeOfSigmoidFunction(double x) {
        return x * (1-x);
    }

    public double[][] getTrainingData() {
        return trainingData;
    }

    public double[] getTrainingResults() {
        return trainingResults;
    }

    public static EnvironmentVariables getInstance() {
        return instance;
    }

}

