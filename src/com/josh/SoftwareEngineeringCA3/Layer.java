package com.josh.SoftwareEngineeringCA3;

import java.util.Random;

public class Layer {

    private double[] output;
    private double[] input;
    private double[] edgeWeights;
    private double[] changeInEdgeWeights;
    private Random random;
    private EnvironmentVariables env;

    public Layer(int noOfInputNeurons, int noOfOutputNeurons) {
        input = new double[noOfInputNeurons + 1];
        output = new double[noOfOutputNeurons];
        changeInEdgeWeights = new double[(noOfInputNeurons+1) * noOfOutputNeurons];
        random = new Random();
        env = EnvironmentVariables.getInstance();
        edgeWeights = initWeights(new double[(noOfInputNeurons+1) * noOfOutputNeurons], 0, 0);

    }

    public double[] initWeights(double[] edgeWeights, double newEdge, int index) {
        return index < edgeWeights.length ? initWeights(edgeWeights, edgeWeights[index] = (random.nextDouble() - 0.5) * 4, index+1) : edgeWeights;
    }

    protected double[] feedForward(double[] inputNeurons) {
        input = copyArray(inputNeurons, input);
        input[input.length-1] = 1;
        for (int i = 0, layerIndex = 0; i < output.length; i++, layerIndex += input.length) {
            output[i] += sumFunction(output[i], 0, layerIndex);
            output[i] = env.sigmoid(output[i]);
        }

        return copyArray(output, new double[output.length]);

    }


    protected double sumFunction(double sum, int index, int layerIndex) {
        sum += edgeWeights[layerIndex+index] * input[index];
        if (index < input.length-1) {
            return sumFunction(sum, index+1, layerIndex);
        } else {
            return sum;
        }
    }

    protected double[] copyArray(double[] originalArray, double[] duplicateArray) {
        return copyArray(originalArray, duplicateArray, 0, 0);
    }

    protected double[] copyArray(double[] originalArray, double duplicateArray[], double updateValue, int index) {
        return index < originalArray.length ? copyArray(originalArray, duplicateArray, duplicateArray[index] = originalArray[index], index+1) : duplicateArray;
    }

    protected double[] train (double[] error) {

        double[] nextError = new double[input.length];
        for (int i = 0, offset = 0; i < output.length; i++, offset += input.length) {
            double delta = error[i] * env.derivativeOfSigmoidFunction(output[i]);
            for (int j = 0; j < input.length; j++) {
                int weightIndex = offset + j;
                nextError[j] = nextError[j] + edgeWeights[weightIndex] * delta;
                edgeWeights[weightIndex] += changeInEdgeWeights[weightIndex] * env.getMOMENTUM() + (input[j] * delta * env.getLEARNING_RATE());
                changeInEdgeWeights[weightIndex] = (input[j] * delta * env.getLEARNING_RATE());
            }
        }
        return nextError;
    }

    protected double calculateGradient(int index, double[] error) {
        return error[index] * env.derivativeOfSigmoidFunction(output[index]);
    }

    protected double[] getEdgeWeights() {
        return edgeWeights;
    }

    public double[] getOutput() {
        return output;
    }

    public void setOutput(double[] output) {
        this.output = output;
    }

    public double[] getInput() {
        return input;
    }

    public void setInput(double[] input) {
        this.input = input;
    }

    public void setEdgeWeights(double[] edgeWeights) {
        this.edgeWeights = edgeWeights;
    }

    public double[] getChangeInEdgeWeights() {
        return changeInEdgeWeights;
    }

    public void setChangeInEdgeWeights(double[] changeInEdgeWeights) {
        this.changeInEdgeWeights = changeInEdgeWeights;
    }

    public EnvironmentVariables getEnv() {
        return env;
    }
}