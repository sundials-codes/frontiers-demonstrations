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
def refSoln(solver, runV, showcommand=True):
    """
    This function generates the reference solution needed to compute the
    error for the population density model.

    Input: solver:            imex scheme to run
        #    modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)

    Output: returns the reference solution as a textfile
    """

    runcommand = " %s  --rtol %e " % (solver['exe'], runV)

    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout_lines = result.stdout.decode().split('\n')
    stderr_lines = result.stderr.decode().split('\n')

    if (result.returncode != 0):
        print(result.stderr.decode())
        sys.exit("Reference run failed")
    else:
        # If SUNDIALS failed  
        sundials_failed = False
        for line in stderr_lines:
            if (("test failed repeatedly" in line) or ("mxstep steps taken before reaching tout" in line)):
                sundials_failed = True
        if sundials_failed == True:
            msg = "Running: " + runcommand + " FAILED"
            sys.exit(msg)
            
        # If SUNDIALS did not fail
        if not sundials_failed:
            print("Running reference solution!")
            final_time = None
            for line in stdout_lines:
                if line.strip().startswith("Current time"):
                    final_time = float(line.split('=')[1].strip())
                    break 

            if final_time is None:
                sys.exit("ERROR: 'Current time' not found in reference output")
            if abs(final_time - 0.3) > 1e-10:
                sys.exit(f"ERROR: reference reached only t = {final_time}, not 0.3")
            
            print(f"Running reference solution : " + runcommand + " SUCCESS")
            new_fileName = f"hyperbolic_relaxation_reference_solution.out"

            ## rename plot file
            if os.path.exists("hyperbolic_relaxation_refSol.out"):
                os.rename("hyperbolic_relaxation_refSol.out", new_fileName)
                print(f"reference solution saved as: {new_fileName}")
            else:
                sys.exit("Warning: hyperbolic_relaxation_refSol.out not found.")

            return new_fileName 

## end of function


# method to generate reference solution
DIRK845 = "./hyperbolic_relaxation_refSol  --dirk_table ARKODE_ARK548L2SA_DIRK_8_4_5  --erk_table ARKODE_ARK548L2SA_ERK_8_4_5  --output 2"     

adaptive_params = [1e-13] #relative tolerance for reference solution

## Integrator types
solvertype = [{'name': 'DIRK-8-4-5', 'exe': DIRK845}]

# run function to generate reference solution
for run_val in adaptive_params:
    for solver in solvertype:
        adapt_refSoln = refSoln(solver, run_val, showcommand=True)

