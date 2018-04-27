package com.josh.SoftwareEngineeringCA3;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by Josh on 26/04/2018.
 */
public class AgentTest {
    private class MockAgent extends Agent {

        public MockAgent() {
            super();
        }

        @Override
        public void learn() {
            return;
        }
    }

    private Agent agent;
    private MockAgent mockAgent;
    private EnvironmentVariables env;
    private double[] inputLayerInput;
    private double[] inputLayerOutput;
    private double[] inputLayerEdgeWeights;
    private double[] inputLayerChangeInEdgeWeights;
    private double[] outputLayerInput;
    private double[] outputLayerOutput;
    private double[] outputLayerEdgeWeights;
    private double[] outputLayerChangeInEdgeWeights;
    private double[][][] trainingDataAndLearnerResults;
    private double[][] invalidInputData;
    private double[][][] searchData;
    private double[][][] verifyPredictionData;
    private double[][][] verifyPredictionFailingData;

    @Before
    public void setUp() throws Exception {
        agent = new Agent();
        mockAgent = new MockAgent();
        env = env.getInstance();
        initializeNeuralNetwork();
    }

    public void initializeNeuralNetwork() {
        inputLayerInput = new double[2];
        inputLayerOutput = new double[3];
        inputLayerEdgeWeights = new double[] {0.87, 0.58, 0.43, 1.33, -0.58, -1.84, 0.66, 1.92, -0.28, 1.43, 1.04, -0.39};
        inputLayerChangeInEdgeWeights = new double[inputLayerOutput.length * 4];
        agent.getNeuralNetwork().getLayers()[0].setInput(inputLayerInput);
        agent.getNeuralNetwork().getLayers()[0].setOutput(inputLayerOutput);
        agent.getNeuralNetwork().getLayers()[0].setEdgeWeights(inputLayerEdgeWeights);
        agent.getNeuralNetwork().getLayers()[0].setChangeInEdgeWeights(inputLayerChangeInEdgeWeights);
        outputLayerInput = new double[4];
        outputLayerOutput = new double[1];
        outputLayerEdgeWeights = new double[] {-0.22, 1.56, -0.42, 0.75, 1.87};
        outputLayerChangeInEdgeWeights = new double[outputLayerOutput.length * 4];
        agent.getNeuralNetwork().getLayers()[1].setInput(outputLayerInput);
        agent.getNeuralNetwork().getLayers()[1].setOutput(outputLayerOutput);
        agent.getNeuralNetwork().getLayers()[1].setEdgeWeights(outputLayerEdgeWeights);
        agent.getNeuralNetwork().getLayers()[1].setChangeInEdgeWeights(outputLayerChangeInEdgeWeights);
        trainingDataAndLearnerResults = new double[][][] {{{0,0}, {0.856}}, {{0,1}, {0.975}}, {{1,0}, {0.981}}, {{1,1}, {0.981}}};
        invalidInputData = new double[][] {{-1, 0}, {-1, 1}, {2, 0}, {2, 1}, {0.1,1}, {0,1.1}, {0,1,1}, {}, {1}};
        searchData = new double[][][] {{{0,0}, {0}}, {{0,1}, {1}}, {{1,0}, {2}}, {{1,1}, {3}}, {{2,1}, {3}}, {{-2,1}, {3}}};
        verifyPredictionData = new double[][][] {{{0,0}, {0}}, {{0, 1}, {1}}, {{1, 0}, {1}}, {{1, 1}, {1}}};
        verifyPredictionFailingData = new double[][][] {{{0,0}, {1}}, {{0, 1}, {0}}, {{1, 0}, {0}}, {{1, 1}, {0}}, {{0,0}, {0.856}}, {{0,1}, {0.975}}, {{1,0}, {0.981}}, {{1,1}, {0.981}}};
    }

    @Test
    public void learn() throws Exception {
        agent.learn();
    }

    @Test
    public void mockAgentLearn() throws Exception {
        mockAgent.learn();
    }

    @Test
    public void predict() throws Exception {
        for (int i = 0; i < trainingDataAndLearnerResults.length; i++) {
            assertArrayEquals(trainingDataAndLearnerResults[i][1], agent.predict(trainingDataAndLearnerResults[i][0]), 0.001);
        }
    }

    @Test
    public void makePrediction() throws Exception {
        for (int i = 0; i < trainingDataAndLearnerResults.length; i++) {
            assertEquals((int) Math.round(trainingDataAndLearnerResults[i][1][0]), (int) Math.round(agent.predict(trainingDataAndLearnerResults[i][0])[0]), 0.001);
        }
    }

    @Test (expected = InvalidInputForNeuralNetworkException.class)
    public void makePredictionWithInvalidInputs() throws Exception {
        for (int i = 0; i < invalidInputData.length; i++) {
            agent.makePrediction(invalidInputData[i]);
        }
    }

    @Test
    public void verifyPrediction() throws Exception {
        for (int i = 0; i < verifyPredictionData.length; i++) {
            assertEquals(verifyPredictionData[i][1][0], agent.verifyPrediction(verifyPredictionData[i][0], verifyPredictionData[i][1]), 0);
        }
    }

    @Test (expected = InadequatelyTrainedNeuralNetworkException.class)
    public void failingToVerifyPrediction() throws Exception {
        for (int i = 0; i < verifyPredictionFailingData.length; i++) {
            assertEquals(verifyPredictionFailingData[i][1][0], agent.verifyPrediction(verifyPredictionFailingData[i][0], verifyPredictionFailingData[i][1]), 0);
        }
    }

    @Test
    public void searchForInputData() throws Exception {
        for (int i = 0; i < searchData.length; i++) {
            assertEquals(searchData[i][1][0], agent.searchForInputData(searchData[i][0], 0), 0);
        }
    }

    @Test (expected = InvalidInputForNeuralNetworkException.class)
    public void InvalidInputForNeuralNetworkException() throws Exception {
        for (int i = 0; i < invalidInputData.length; i++) {
            agent.makePrediction(invalidInputData[i]);
        }
    }

    @Test
    public void getEnv() throws Exception {
        assertEquals(agent.getEnv(), env.getInstance());
    }

    @Test
    public void getAndSetNeuralNetwork() throws Exception {
        NeuralNetwork neuralNetwork = new NeuralNetwork(2,4,1);
        agent.setNeuralNetwork(neuralNetwork);
        assertEquals(neuralNetwork, agent.getNeuralNetwork());
    }

}