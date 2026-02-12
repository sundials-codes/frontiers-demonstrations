#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County.
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
# def refSoln(solver, modetype, runV, kstiff, ksN, knonstiff, showcommand=True):
# def refSoln(solver, runV, kstiff, ksN, knonstiff, showcommand=True):
def refSoln(solver, runV, k1Val, k1Valname, showcommand=True):
    """
    This function generates the reference solution needed to compute the
    error for the population density model.

    Input: solver:            imex scheme to run
        #    modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           runN:              given name of rtol or fixed_h
           kstiff:            stiffness parameter
           knonstiff:         nonstiffness parameter

    Output: returns the reference solution as a textfile
    """

    k2Val = 2.0 * k1Val

    runcommand = "%s  --rtol %e  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val)

    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print(f"Running reference solution : " + runcommand + " SUCCESS")
            new_fileName = f"refSoln_linear_adv_rec_{k1Valname}.txt"

            ## rename plot file
            if os.path.exists("linear_adv_rec.txt"):
                os.rename("linear_adv_rec.txt", new_fileName)
                print(f"reference solution saved as: {new_fileName}")
            else:
                print("Warning: linear_adv_rec.txt not found.")

    return new_fileName 
## end of function


# method to generate reference solution
SSP423 = "./linear_adv_rec   --IMintegrator ARKODE_ARK548L2SA_DIRK_8_4_5      --EXintegrator ARKODE_ARK548L2SA_ERK_8_4_5"     

adaptive_params = [1e-14] #relative tolerance for reference solution
k1values = {'k1Val1': 1.0, 'k1Val1e6': 1e6}#, 'k1Val1e8': 1e8}

## Integrator types
solvertype = [{'name': 'SSP-ARK-4-2-3', 'exe': SSP423}]

# run function to generate reference solution
for runvalue in adaptive_params:
    # for k1_val, k2_val in zip(k1values, k2values):
    for k1_valName, k1_val in k1values.items():
        for solver_adapt in solvertype:
            adaptive_stat= refSoln(solver_adapt, runvalue, k1_val, k1_valName, showcommand=True)
