#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: This scripts runs the different imex schemes with different diffusion coefficients and parameters, 
#         using either adaptive or fixed time stepping
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
def runtest(solver, modetype, runV, runN, showcommand=True, sspcommand=True):
    """
    This function runs the hyperbolic equation with relaxation using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           runN:              given name of rtol or fixed_h

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 'Steps': 0, 
             'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 'runtime':0.0}

    if (modetype == "adaptive"):
        runcommand = "SUNLOGGER_INFO_FILENAME=sun-%s-%s.log %s  --rtol %e" % (solver['name'], runN, solver['exe'], runV)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f --output 2" % (solver['exe'], runV)
    
    start_time = time.time()
    result = subprocess.run(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']    = modetype
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = length_time

    stdout_lines = str(result.stdout).split('\\n')
    stderr_lines = str(result.stderr).split('\\n')

     # If SUNDIALS failed  
    sundials_failed = False
    for line in stderr_lines:
        if (("test failed repeatedly" in line) or ("mxstep steps taken before reaching tout" in line)):
            sundials_failed = True
    if sundials_failed == True:
        print("Running: " + runcommand + " FAILED")
        stats['ReturnCode']       = 1
        stats['Steps']            = 0
        stats['StepAttempts']     = 0
        stats['ErrTestFails']     = 0
        stats['Explicit_RHS']     = 0 
        stats['Implicit_RHS']     = 0   

    # If SUNDIALS did not fail
    if not sundials_failed:
        print("Running: " + runcommand + " SUCCESS")
        for line in stdout_lines:
            txt = line.split()
            if ("Steps" in txt):
                stats['Steps'] = int(txt[2])
            elif (("Step" in txt) and ("attempts" in txt)):
                stats['StepAttempts'] = int(txt[3])
            elif (("Error" in txt) and ("Fails" in txt)):
                stats['ErrTestFails'] = float(txt[4])
            elif (("Explicit" in txt) and ("RHS" in txt)):
                stats['Explicit_RHS'] = int(txt[5])       #right hand side evaluations for explicit method
            elif (("Implicit" in txt) and ("RHS" in txt)):
                stats['Implicit_RHS'] = int(txt[5])       #right hand side evaluations for implicit method

        datafile = "plot_hyperbolic_relaxation.py"
        # return with an error if the file does not exist
        if not os.path.isfile(datafile):
            msg = "Error: file " + datafile + " does not exist"
            sys.exit(msg)

        if (modetype=="adaptive"):
            new_file  = "sun-%s-%s.log" % (solver['name'], runN)
            save_file = "sun-%s-%s" % (solver['name'], runN)
            with open(datafile, "r") as file:
                lines = file.readlines()

            # keep the orignal files
            original_lines = lines.copy()
            
            # modify name of .log file and its figure
            modified_lines = []
            for line in lines:
                if "file_to_copy = './sun1.log'" in line:
                    modified_lines.append(f"file_to_copy = './{new_file}'\n")
                elif ("log_example.py" in line) and  ("--save sun_save" in line):
                    modified_lines.append(line.replace("sun_save", save_file))
                else:
                    modified_lines.append(line)
                
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

            ## running python file to plot pressure and density
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 

            #restore to original line
            with open(datafile, "w") as f:
                f.writelines(original_lines)
        else:
            ## running python file to plot pressure and density
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)   
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2        --output 2" 
SSP312  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2        --output 2"           
SSPL312 = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2  --output 2"  
SSP423  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3        --output 2"    

## common testing parameters
adaptive_params = {'r1':1e-5, 'r2':1e-4, 'r3':1e-3, 'r4':1e-4, 'r5':1e-5} ## relative tolerances
# fixed_params    = [] # fixed time step sizes
# for i in range(8,-1,-1):
#     fixed_params.append(0.002/(2.0**i))

## Integrator types
solvertype = [{'name': 'SSP212',  'exe': SSP212},
              {'name': 'SSP312',  'exe': SSP312},
              {'name': 'SSPL312', 'exe': SSPL312},
              {'name': 'SSP423',  'exe': SSP423}]

# run tests and collect results as a pandas data frame
fname = 'hyperbolic_relaxation_stats' 
RunStats = []

for runname, runvalue in adaptive_params.items():
    for solver_adapt in solvertype:
        adaptive_stat = runtest(solver_adapt, "adaptive", runvalue, runname, showcommand=True, sspcommand=True)
        RunStats.append(adaptive_stat)

# for runname, runvalue in fixed_params:
#     for solver_fixed in solvertype:
#         fixed_stat = runtest(solver_fixed, "fixed", runvalue, showcommand=True, sspcommand=True)
#         RunStats.append(fixed_stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)