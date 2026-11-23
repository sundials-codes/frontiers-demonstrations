#!/bin/bash
# This script generates all results for examples.

echo -e "Generating stats for the linear advection reaction example. \n"

echo -e "Generating reference solution for each sigma value. \n"
python3 runtests_linear_adv_rec_refSol.py

echo -e "Generating all stats and saving in an excel file as well as plots.\n"
python3 runtests_linear_adv_rec.py

echo -e "Run tests for the linear advection-reaction example completed!\n"

# ----------------------------------------------------------------------------
echo -e "Generating stats for the population density example. \n"

echo -e "Generating reference solutions for each diffusion coefficient. \n"
python3 runtests_population_refSol.py

echo -e "Generating all stats and saving in an excel file. \n"
echo -e "  Step 1: MODE = uncheckedSMOOTH (bisection runs and final-time plots). \n"
python3 runtests_population.py uncheckedSMOOTH

echo -e "  Step 2: MODE = ssp (uses the TAU values set in runtests_population.py). \n"
python3 runtests_population.py ssp

echo -e "Run tests for the population density example completed!\n"

# ----------------------------------------------------------------------------
echo -e "Generating stats for the hyperbolic relaxation example. \n"

echo -e "Generating reference solution for the hyperbolic equation with stiff relaxation. \n"
python3 runtests_hyperbolic_relaxation_refSol.py

echo -e "Generating all stats and saving in an excel file as well as plots. \n"
python3 runtests_hyperbolic_relaxation.py

echo -e "Run tests for the hyperbolic relaxation example completed!\n"