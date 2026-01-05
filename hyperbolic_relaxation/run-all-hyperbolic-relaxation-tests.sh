#!/bin/bash
# This script generates all results for the hyperbolic relaxation problem.

echo -e "Generating reference solutions for each diffusion coefficient\n"
python3 runtests_hyperbolic_relaxation_refSol.py

echo -e "Generating all stats and saving in an excel file\n"
python3 runtests_hyperbolic_relaxation.py

echo -e "Run tests completed!\n"
