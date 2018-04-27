package com.josh.SoftwareEngineeringCA3;

import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

import static org.junit.Assert.*;

/**
 * Created by Josh on 20/04/2018.
 */
public class LayerTest {

    private class MockLayer extends Layer {

        private double[] edgeWeights;

        public MockLayer(int noOfInputNeurons, int noOfOutputNeurons) {
            super(noOfInputNeurons, noOfOutputNeurons);
            edgeWeights = initWeights((new double[(noOfInputNeurons+1) * noOfOutputNeurons]), 0, 0);
        }

        @Override
        public double[] initWeights(double[] edgeWeights, double newEdge, int index) {
            return edgeWeights;
        }

    }

    private Layer inputLayer;
    private Layer outputLayer;


    private MockLayer mockLayer;

    @Before
    public void setUp() throws Exception {
        inputLayer = new Layer(2, 4);
        outputLayer = new Layer(4, 1);
    }

    @Test
    public void initWeights() throws Exception {
        for (int i = 0; i < 100; i++) {
            mockLayer = new MockLayer(i,i);
            assertArrayEquals(mockLayer.getEdgeWeights(), new double[(i+1) * i], 0);
        }
    }

    @Test
    public void feedForward() throws Exception {
        inputLayer.setInput(new double[] {0.5, 0.5, 0.5, 0.5, 0.5});
        inputLayer.setOutput(new double[] {0.9});
        inputLayer.setEdgeWeights(new double[] {1, 1, 1, 1, 1});
        assertArrayEquals(new double[] {0.995}, inputLayer.feedForward(new double[] {1}), 0.001);
    }


    @Test
    public void sumFunction() throws Exception {
        HashMap<Double[][], Double> sumFunctionHashMap = new HashMap<>();
        sumFunctionHashMap.put(new Double[][] {
                new Double[] {1.0,1.0,2.0},
                new Double[] {5.0,8.0,13.0,21.0, 5.0,8.0,13.0,21.0, 5.0,8.0,13.0,21.0, 5.0,8.0,13.0,21.0}
        }, 39.0);
        sumFunctionHashMap.put(new Double[][] {
                new Double[] {1.0,1.0,2.0},
                new Double[] {34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0}
        }, 267.0);
        sumFunctionHashMap.put(new Double[][] {
                new Double[] {1.0,1.0,2.0},
                new Double[] {233.0,377.0,610.0,987.0, 233.0,377.0,610.0,987.0, 233.0,377.0,610.0,987.0, 233.0,377.0,610.0,987.0}
        }, 1830.0);
        sumFunctionHashMap.put(new Double[][] {
                new Double[] {34.0,55.0,89.0},
                new Double[] {34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0, 34.0,55.0,89.0,144.0}
        }, 12102.0);
        for (Map.Entry<Double[][], Double> inputAndEdgeWeights: sumFunctionHashMap.entrySet()) {
            double[] input = new double[inputAndEdgeWeights.getKey()[0].length];
            double[] edgeWeights = new double[inputAndEdgeWeights.getKey()[1].length];
            for (int i = 0; i < inputAndEdgeWeights.getKey()[0].length; i++) {
                input[i] = inputAndEdgeWeights.getKey()[0][i];
            }
            for (int i = 0; i < inputAndEdgeWeights.getKey()[1].length; i++) {
                edgeWeights[i] = inputAndEdgeWeights.getKey()[1][i];
            }
            inputLayer.setInput(input);
            inputLayer.setEdgeWeights(edgeWeights);
            for (int i = 0, layerIndex= 0; i < 4; i++, layerIndex+=4) {
                assertEquals(inputAndEdgeWeights.getValue(), inputLayer.sumFunction(0,0,layerIndex), 0);
            }
        }

    }

    @Test
    public void copyArray() throws Exception {
        for (int i = 0; i < 1000; i++) {
            double[] arr1 = {i, i+1, i+2};
            double[] arr2 = {i, i+1, i+2, i+3, i+4};
            assertArrayEquals(arr1, inputLayer.copyArray(arr1, new double[arr1.length]), 0);
            assertArrayEquals(arr2, outputLayer.copyArray(arr2, new double[arr2.length]), 0);
        }
    }

    @Test
    public void copyArrayHelper() throws Exception {
        for (int j = 0; j <= 3; j++) {
            for (int i = 0; i < 1000; i++) {
                double[] arr1 = {i, i+1, i+2};
                double[] arr2 = {i, i+1, i+2};
                for (int k = 0; k < j; k++) {
                    arr2[k] = 0;
                }
                assertArrayEquals(arr2, inputLayer.copyArray(arr1, new double[arr1.length], 0, j), 0);
                assertArrayEquals(arr2, outputLayer.copyArray(arr2, new double[arr2.length]), 0);
            }
        }
    }

    @Test
    public void train() throws Exception {
        inputLayer.setInput(new double[] {0.5, 0.5, 0.5, 0.5, 0.5});
        inputLayer.setOutput(new double[] {0.9});
        inputLayer.setEdgeWeights(new double[] {1, 1, 1, 1, 1});
        inputLayer.setChangeInEdgeWeights(new double[] {0.20, 0.20, 0.20, 0.20, 0.20});
        assertArrayEquals(new double[] {0.0899,0.0899,0.0899,0.0899,0.0899}, inputLayer.train(new double[] {1}), 0.001);
    }

    @Test
    public void calculateGradient() throws Exception {
        inputLayer.setOutput(new double[] {4,4,4});
        assertEquals(-48, inputLayer.calculateGradient(0, new double[] {4,4,4}), 0);
    }

    @Test
    public void getEnv() throws Exception {
        assertEquals(EnvironmentVariables.getInstance(), inputLayer.getEnv());
    }

    @Test
    public void getAndSetOutput() throws Exception {
        double[] arr = {1, 2, 3};
        inputLayer.setOutput(arr);
        assertArrayEquals(inputLayer.getOutput(), arr, 0);
    }

    @Test
    public void getAndSetInput() throws Exception {
        double[] arr = {1, 2, 3};
        inputLayer.setInput(arr);
        assertArrayEquals(inputLayer.getInput(), arr, 0);
    }

    @Test
    public void getAndSetEdgeWeights() throws Exception {
        double[] arr = {1, 2, 3};
        inputLayer.setEdgeWeights(arr);
        assertArrayEquals(inputLayer.getEdgeWeights(), arr, 0);
    }

    @Test
    public void getAndSetChangeInEdgeWeights() throws Exception {
        double[] arr = {1, 2, 3};
        inputLayer.setChangeInEdgeWeights(arr);
        assertArrayEquals(inputLayer.getChangeInEdgeWeights(), arr, 0);
    }

}