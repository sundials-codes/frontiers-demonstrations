#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadMe: This script generates the log files needed to plot the all the steps sizes used in the adaptive run.
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

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, runN, runV, kName, kVal, showcommand=True):
    stats = {'ReturnCode': 0}

    runcommand = f"SUNLOGGER_INFO_FILENAME=sun_%s_%s_%s.log %s  --rtol %e  --k %.2f" % (solver['name'], kName, runN, solver['exe'], runV, kVal)

    result = subprocess.run(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stats['ReturnCode'] = result.returncode
        
    if (result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Run command " + runcommand + " SUCCESS")     
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP_ARK_212       = "./population_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./population_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./population_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./population_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"            

## common testing parameters
adaptive_params = {'r1':1.e-1, 'r2':1.e-2, 'r3':1.e-3, 'r4':1.e-4, 'r5':1.e-5} ## Relative tolerances

## Diffusion coefficients
diff_coef = {'k2':0.02, 'k4':0.04}

## Integrator types
solvertype = [{'name': '212',       'exe': SSP_ARK_212},
              {'name': '312',       'exe': SSP_ARK_312},
              {'name': 'L312',      'exe': SSP_LSPUM_ARK_312},
              {'name': '423',       'exe': SSP_ARK_423}]

# run program
for k_name, k_val in diff_coef.items():
    for runV_name, runV_val in adaptive_params.items():
            for solver in solvertype:
              stat = runtest(solver, runV_name, runV_val, k_name, k_val, showcommand=True)

