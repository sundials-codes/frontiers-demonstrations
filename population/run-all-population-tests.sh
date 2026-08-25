#!/bin/bash
# This script generates all results for the population model.

echo -e "Generating reference solutions for each diffusion coefficient\n"
python3 runtests_population_refSol.py

echo -e "Generating all stats and saving in an excel file\n"
python3 runtests_population.py

echo -e "Run tests completed!\n"
