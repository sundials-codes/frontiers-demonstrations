#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: This script generates the reference solutions needed to compute the error norm for the populatin density model.
#-------------------------------------------------------------------------------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import time
import shlex
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec
from math import log10, floor

# utility routine to run a test, storing the run options and solver statistics
def refSoln(solver, modetype, runV, kval, kname, showcommand=True):
    """
    This function generates the reference solution needed to compute the
    error for the population density model.

    Input: solver:            imex scheme
           modetype (string): adaptive or fixed
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient

    Output: returns the reference solution as a textfile
    """

    if (modetype == "adaptive"):
        runcommand = " %s --rtol %e  --k %.2f" % (solver['exe'], runV, kval)
    elif (modetype == "fixed"):
        runcommand = " %s --fixed_h %.6f --k %.2f" % (solver['exe'], runV, kval)

    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print(f"Running {modetype} reference solution for {kval}: " + runcommand + " SUCCESS")
            new_fileName = f"{modetype}_referenceSoln_{kname}.txt"

            ## rename plot file
            if os.path.exists("population.txt"):
                os.rename("population.txt", new_fileName)
                print(f"reference solution saved as: {new_fileName}")
            else:
                print("Warning: population.txt not found.")

    return new_fileName 
## end of function


# method to generate reference solution
SSP_ARK_423 = "./population_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3  --EXintegrator ARKODE_SSP_ERK_4_2_3"     

adaptive_params = [1e-10] ## relative tolerance for reference solution
fixed_params    = [1e-4] ## fixed time step size for reference solution

## Diffusion coefficients
diff_coef = {'k2':0.02, 'k4':0.04}

## Integrator types
solvertype = [{'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423}]

# run function to generate reference solution
for k_name, k_val in diff_coef.items():
    for run_val in adaptive_params:
        for solver in solvertype:
            adapt_refSoln = refSoln(solver, "adaptive", run_val, k_val, k_name, showcommand=True)

    for run_val in fixed_params:
        for solver in solvertype:
            fixed_refSoln = refSoln(solver, "fixed", run_val, k_val, k_name, showcommand=True)

