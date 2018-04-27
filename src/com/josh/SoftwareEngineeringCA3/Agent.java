package com.josh.SoftwareEngineeringCA3;

/**
 * Created by Josh on 26/04/2018.
 */
public class Agent {

    private NeuralNetwork neuralNetwork;
    private EnvironmentVariables env;

    public Agent() {
        neuralNetwork = new NeuralNetwork(2, 4, 1);
        env = EnvironmentVariables.getInstance();
    }

    public void learn() {
        for (int i = 0; i < env.getNETWORK_TRAINING_ITERATIONS(); i++) {
            for (int j = 0; j < env.getTrainingResults().length; j++) {
                neuralNetwork.train(env.getTrainingData()[j], env.getTrainingResults()[j]);
            }
        }
    }

    public int makePrediction(double[] inputData) {
        if ((inputData.length > 2 || inputData.length < 2) || (inputData[0] != 1 && inputData[0] != 0) || (inputData[1] != 1 && inputData[1] != 0)) {
            throw new InvalidInputForNeuralNetworkException("Invalid Input has been entered for the neural network, the neural network only supports logical operations on two binary numbers (1,0)");
        } else {
            System.out.print("The result of: " + (int) inputData[0] + " OR " + (int) inputData[1] + " = ");
            return (int) verifyPrediction(inputData, predict(inputData));
        }
    }

    public double[] predict(double[] inputData) {
        return neuralNetwork.feedForward(inputData);
    }

    public double verifyPrediction(double[] inputData, double[] result) {
        int index = searchForInputData(inputData, 0);
        result[0] = (int) (Math.round(result[0]));
        if (env.getTrainingResults()[index] == result[0]) {
            return result[0];
        } else {
            throw new InadequatelyTrainedNeuralNetworkException("There was an issue when training the neural network this could be due to the network not being trained for long enough, oversizing or under sizing");
        }
    }

    public int searchForInputData(double[] inputData, int index) {
        return (inputData[0] == env.getTrainingData()[index][0]) && (inputData[1] == env.getTrainingData()[index][1]) || index > inputData.length ? index : searchForInputData(inputData, index+1);
    }

    public void setNeuralNetwork(NeuralNetwork neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public NeuralNetwork getNeuralNetwork() {
        return neuralNetwork;
    }

    public EnvironmentVariables getEnv() {
        return env;
    }

}
