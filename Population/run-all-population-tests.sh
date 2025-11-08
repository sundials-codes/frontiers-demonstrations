#!/bin/bash
# This script generates all results for the population model.

echo -e "Generating reference solutions for each diffusion coefficient\n"
python3 runtest_population_refSol.py

echo -e "Generating all stats and saving in an excel file\n"
python3 runtest_population_imex.py

echo -e "Generating the numerical solution graph at the final time step \n"
python3 plot_population_finalStepSoln.py

echo -e "Run tests completed!\n"
