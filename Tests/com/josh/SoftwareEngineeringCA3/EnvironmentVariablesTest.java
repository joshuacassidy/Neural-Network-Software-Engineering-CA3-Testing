package com.josh.SoftwareEngineeringCA3;

import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by Josh on 26/04/2018.
 */
public class EnvironmentVariablesTest {

    private EnvironmentVariables env;
    private double[] sigmoidValues;
    private double[] derivativeOfSigmoidFunctionValues;

    @Before
    public void setUp() throws Exception {
        env = env.getInstance();
        sigmoidValues = new double[] {0.2689, 0.3208, 0.3775, 0.4378, 0.5000, 0.5622, 0.6225, 0.6792, 0.7311};
        derivativeOfSigmoidFunctionValues = new double[] {-2, -1.3125, -0.75, -0.3125, 0, 0.1875, 0.25, 0.1875, 0};
    }

    @Test
    public void getMOMENTUM() throws Exception {
        assertEquals(0.6, env.getMOMENTUM(), 0);
    }

    @Test
    public void getLEARNING_RATE() throws Exception {
        assertEquals(0.3, env.getLEARNING_RATE(), 0);
    }

    @Test
    public void getNETWORK_TRAINING_ITERATIONS() throws Exception {
        assertEquals(100000, env.getNETWORK_TRAINING_ITERATIONS());
    }

    @Test
    public void getLAYER_TRAINING_ITERATIONS() throws Exception {
        assertEquals(10, env.getLAYER_TRAINING_ITERATIONS());
    }



    @Test
    public void sigmoid() throws Exception {
        for (double i = -1, index = 0; i <= 1; i+=0.25, index++) {
            assertEquals(env.sigmoid(i), sigmoidValues[(int) index], 0.0001);
        }
    }

    @Test
    public void derivativeOfSigmoidFunction() throws Exception {
        for (double i = -1, index = 0; i <= 1; i+=0.25, index++) {
            assertEquals(env.derivativeOfSigmoidFunction(i), derivativeOfSigmoidFunctionValues[(int) index], 0.0001);
        }
    }

    @Test
    public void getInstance() throws Exception {
        assertEquals(env.getInstance(), env);
    }

    @Test
    public void getTrainingData() throws Exception {
        assertArrayEquals(new double[][] {{0,0}, {0,1}, {1,0}, {1,1}}, env.getTrainingData());
    }

    @Test
    public void getTrainingResults() throws Exception {
        assertArrayEquals(new double[] {0, 1, 1, 1}, env.getTrainingResults(), 0);
    }

}