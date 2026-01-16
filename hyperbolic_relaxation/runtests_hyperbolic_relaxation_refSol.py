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
def refSoln(solver, runV, kstiff, ksN, knonstiff, showcommand=True):
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

    # if (modetype == "adaptive"):
    runcommand = " %s  --rtol %.6e  --eps_stiff %.2e  --eps_nonstiff %.2e" % (solver['exe'], runV, kstiff, knonstiff)
    # elif (modetype == "fixed"):
    #     runcommand = " %s  --fixed_h %.6e  --eps_stiff %.2e  --eps_nonstiff %.2e" % (solver['exe'], runV, kstiff, knonstiff)

    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            # print(f"Running {modetype} reference solution : " + runcommand + " SUCCESS")
            # new_fileName = f"{modetype}_referenceSoln_{ksN}.out"
            print(f"Running reference solution : " + runcommand + " SUCCESS")
            new_fileName = f"referenceSoln_{ksN}.out"

            ## rename plot file
            if os.path.exists("hyperbolic_relaxation.out"):
                os.rename("hyperbolic_relaxation.out", new_fileName)
                print(f"reference solution saved as: {new_fileName}")
            else:
                print("Warning: hyperbolic_relaxation.out not found.")

    return new_fileName 
## end of function


# method to generate reference solution
SSP423 = "./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3  --EXintegrator ARKODE_SSP_ERK_4_2_3  --output 2"     

adaptive_params = [1e-14] #relative tolerance for reference solution
# fixed_params    = [1e-10]  #fixed time step size for reference solution
nonstiff_params = [1e2]
stiff_params    = {'ks1e6': 1e6, 'ks1e7': 1e7, 'ks1e8': 1e8}

## Integrator types
solvertype = [{'name': 'SSP-ARK-4-2-3', 'exe': SSP423}]

# run function to generate reference solution
for knsval in nonstiff_params:
    for ksname, ksval in stiff_params.items():
        for run_val in adaptive_params:
            for solver in solvertype:
                adapt_refSoln = refSoln(solver, run_val, ksval, ksname, knsval, showcommand=True)

# for knsval in nonstiff_params:
#     for ksname, ksval in stiff_params.items():
#         for run_val in fixed_params:
#             for solver in solvertype:
#                 fixed_refSoln = refSoln(solver, "fixed", run_val, ksval, ksname, knsval, showcommand=True)

