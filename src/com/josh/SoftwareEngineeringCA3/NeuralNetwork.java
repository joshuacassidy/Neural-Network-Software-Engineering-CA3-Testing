package com.josh.SoftwareEngineeringCA3;

public class NeuralNetwork {

    private Layer[] layers;
    private EnvironmentVariables env;

    public NeuralNetwork(int noOfInputNeurons, int noOfHiddenNeurons, int noOfOutputNeurons) {
        layers = new Layer[] {
                new Layer(noOfInputNeurons, noOfHiddenNeurons), // edges going from input to the hidden layer
                new Layer(noOfHiddenNeurons, noOfOutputNeurons) // edges going from hidden layer to output layer
        };
        env = EnvironmentVariables.getInstance();
    }

    public double[] feedForward(double[] input) {
        return updateInputActivations(0, input);
    }

    public double[] updateInputActivations(int index, double[] input) {
        return index < layers.length ? updateInputActivations(index+1, layers[index].feedForward(input)) : input;
    }

    public double[] train(double[] input, double targetOutput) {

        double[] calculatedOutput = feedForward(input);
        double[] error;
        error = calculateError(calculatedOutput, targetOutput,  new double[calculatedOutput.length], targetOutput - calculatedOutput[0], 0);

        for (int layerIterations = 0; layerIterations < env.getLAYER_TRAINING_ITERATIONS(); layerIterations++) {
            for (int i = (layers.length-1); i > 0; i--) {
                layers[i].train(error);
            }
        }

        return error;
    }

    public double[] calculateError(double[] calculatedOutput, double targetOutput, double[] error, double errorUpdate, int index) {
        return index < error.length ? calculateError(calculatedOutput, targetOutput, error, error[index] = targetOutput - calculatedOutput[index], index+1) : error;
    }

    public Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public EnvironmentVariables getEnv() {
        return env;
    }

}