package com.josh.SoftwareEngineeringCA3;

import org.junit.Before;
import org.junit.Test;

/**
 * Created by Josh on 20/04/2018.
 */
public class MainTest {
    private Main main;
    private String[] args;

    @Before
    public void setUp() throws Exception {
        args = null;
        main = new Main();
    }

    @Test
    public void main() throws Exception {
        Main.main(args);
    }

    @Test
    public void testMainWithNewInstance() throws Exception {
        main.main(args);
    }
}
