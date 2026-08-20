#!/bin/bash
# This script generates all results for the linear advection-reaction example.

echo -e "Generating reference solution for each sigma value \n"
python3 runtests_linear_adv_rec_refSol.py

echo -e "Generating all stats and saving in an excel file as well as plots\n"
python3 runtests_linear_adv_rec.py

echo -e "Run tests completed!\n"