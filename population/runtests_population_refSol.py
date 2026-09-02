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
def refSoln(solver, runV, kval, kname, showcommand=True):
    """
    This function generates the reference solution needed to compute the
    error for the population density model.

    Input: solver:            imex scheme
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient

    Output: returns the reference solution as a textfile
    """

    runcommand = " %s  --rtol %e   --k %e" % (solver['exe'], runV, kval)

    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout_lines = result.stdout.decode().split('\n')
    stderr_lines = result.stderr.decode().split('\n')

    if (result.returncode != 0):
        print(result.stderr.decode())
        sys.exit("Reference run failed")
        # print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        # print(result.stderr)
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
            if abs(final_time - 10.0) > 1e-10:
                sys.exit(f"ERROR: reference reached only t = {final_time}, not 10.0")

            print(f"Running reference solution for {kval}: " + runcommand + " SUCCESS")
            new_fileName = f"referenceSoln_population_{kname}.txt"

            ## rename plot file
            if os.path.exists("population_refSol.txt"):
                os.rename("population_refSol.txt", new_fileName)
                print(f"reference solution saved as: {new_fileName}")
            else:
                sys.exit("Warning: population_refSol.txt not found.")

            return new_fileName 
                
        # if (showcommand):
        #     print(f"Running reference solution for {kval}: " + runcommand + " SUCCESS")
        #     new_fileName = f"referenceSoln_population_{kname}.txt"

        #     ## rename plot file
        #     if os.path.exists("population_refSol.txt"):
        #         os.rename("population_refSol.txt", new_fileName)
        #         print(f"reference solution saved as: {new_fileName}")
        #     else:
        #         print("Warning: population_refSol.txt not found.")

    # return new_fileName 
## end of function


# method to generate reference solution
ARK845 = "./population_refSoln  --dirk_table ARKODE_ARK548L2SA_DIRK_8_4_5   --erk_table ARKODE_ARK548L2SA_ERK_8_4_5  --atol 1e-14   "     

adaptive_params = [1e-14] ## relative tolerance for reference solution

## Diffusion coefficients
diff_coef = {'diffk0':0.0, 'diffk02':0.02, 'diffk04':0.04}

## Integrator types
solvertype = [{'name': 'ARK-8-4-5', 'exe': ARK845}]

# run function to generate reference solution
for k_name, k_val in diff_coef.items():
    for run_val in adaptive_params:
        for solver in solvertype:
            adapt_refSoln = refSoln(solver, run_val, k_val, k_name, showcommand=True)

