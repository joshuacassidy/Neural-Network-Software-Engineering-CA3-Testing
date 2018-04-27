package com.josh.SoftwareEngineeringCA3;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

public class NeuralNetworkTest {

    private NeuralNetwork neuralNetwork;
    private double[] inputLayerInput;
    private double[] inputLayerOutput;
    private double[] inputLayerEdgeWeights;
    private double[] inputLayerChangeInEdgeWeights;
    private double[] outputLayerInput;
    private double[] outputLayerOutput;
    private double[] outputLayerEdgeWeights;
    private double[] outputLayerChangeInEdgeWeights;

    @Before
    public void setUp() throws Exception {
        neuralNetwork = new NeuralNetwork(2, 4, 1);
        initializeTestInputLayer();
        initializeTestOutputLayer();
    }

    public void initializeTestInputLayer() {
        inputLayerInput = new double[2];
        inputLayerOutput = new double[3];
        inputLayerEdgeWeights = new double[] {0.87, 0.58, 0.43, 1.33, -0.58, -1.84, 0.66, 1.92, -0.28, 1.43, 1.04, -0.39};
        inputLayerChangeInEdgeWeights = new double[inputLayerOutput.length * 4];
        neuralNetwork.getLayers()[0].setInput(inputLayerInput);
        neuralNetwork.getLayers()[0].setOutput(inputLayerOutput);
        neuralNetwork.getLayers()[0].setEdgeWeights(inputLayerEdgeWeights);
        neuralNetwork.getLayers()[0].setChangeInEdgeWeights(inputLayerChangeInEdgeWeights);
    }

    public void initializeTestOutputLayer() {
        outputLayerInput = new double[4];
        outputLayerOutput = new double[1];
        outputLayerEdgeWeights = new double[] {-0.22, 1.56, -0.42, 0.75, 1.87};
        outputLayerChangeInEdgeWeights = new double[outputLayerOutput.length * 4];
        neuralNetwork.getLayers()[1].setInput(outputLayerInput);
        neuralNetwork.getLayers()[1].setOutput(outputLayerOutput);
        neuralNetwork.getLayers()[1].setEdgeWeights(outputLayerEdgeWeights);
        neuralNetwork.getLayers()[1].setChangeInEdgeWeights(outputLayerChangeInEdgeWeights);
    }

    @Test
    public void calculateError() throws Exception {
        // Structure of calculateErrorValues is below
        // calculatedOutput, targetOutput, error, result
        double[][][] calculateErrorValues = {
                {{0.5}, {1}, {0.5}, {0.5}},
                {{0}, {1}, {1}, {1.0}},
                {{1}, {1}, {1}, {0.0}},
                {{0.2}, {0}, {0.2}, {-0.2}},
                {{0.4}, {1}, {0.4}, {0.6}},
                {{0.8}, {1}, {0.8}, {0.2}},
                {{0.4}, {1}, {0.8}, {0.6}}
        };
        for (int i = 0; i < calculateErrorValues.length; i++) {
            assertArrayEquals(calculateErrorValues[i][3], neuralNetwork.calculateError(calculateErrorValues[i][0], calculateErrorValues[i][1][0], calculateErrorValues[i][2], 0, 0), 0.01);
        }
    }

    @Test
    public void setAndGetLayers() throws Exception {
        Layer inputLayer = new Layer(2, 4);
        Layer outputLayer = new Layer(4, 1);
        neuralNetwork.setLayers(new Layer[] {inputLayer, outputLayer});
        assertEquals(neuralNetwork.getLayers()[0], inputLayer);
        assertEquals(neuralNetwork.getLayers()[1], outputLayer);
    }

    @Test
    public void getEnv() throws Exception {
        assertEquals(EnvironmentVariables.getInstance(), neuralNetwork.getEnv());
    }

    @Test
    public void run() throws Exception {
        assertArrayEquals(neuralNetwork.feedForward(new double[] {0,0}), new double[] {0.856}, 0.001);
    }

    @Test
    public void updateInputActivations() throws Exception {
        assertArrayEquals(neuralNetwork.updateInputActivations(0, new double[] {0,0}), new double[] {0.856}, 0.001);
    }

    @Test
    public void train() throws Exception {
        assertArrayEquals(neuralNetwork.train(new double[] {0,0}, 0), new double[] {-0.856}, 0.001);
        assertArrayEquals(neuralNetwork.train(new double[] {0,1}, 1), new double[] {0.0727}, 0.001);
        assertArrayEquals(neuralNetwork.train(new double[] {1,0}, 1), new double[] {0.0620}, 0.001);
        assertArrayEquals(neuralNetwork.train(new double[] {1,1}, 1), new double[] {0.0578}, 0.001);
    }

}
