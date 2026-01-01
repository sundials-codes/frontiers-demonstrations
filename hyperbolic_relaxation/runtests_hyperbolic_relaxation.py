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
import shutil
import shlex
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec
from math import log10, floor

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, modetype, runV, runN, kstiff, knonstiff, showcommand=True, sspcommand=True):
    """
    This function runs the hyperbolic equation with relaxation using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           runN:              given name of rtol or fixed_h
           kstiff:            stiffness parameter
           knonstiff:         nonstiffness parameter

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 'runtime':0.0, 'stiff_param': 0.0, 
             'nonstiff_param': 0.0, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0}

    if (modetype == "adaptive"):
        runcommand = "SUNLOGGER_INFO_FILENAME=sun-%s-%s.log %s  --rtol %.2e  --eps_stiff %.2e  --eps_nonstiff %.2e" % (solver['name'], runN, solver['exe'], runV, kstiff, knonstiff)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f  --eps_stiff %.2e  --eps_nonstiff %.2e" % (solver['exe'], runV, kstiff, knonstiff)
    
    start_time = time.time()
    result = subprocess.run(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']        = modetype
    stats['ReturnCode']     = result.returncode
    stats['runtime']        = length_time
    stats['stiff_param']    = kstiff
    stats['nonstiff_param'] = knonstiff 

    stdout_lines = str(result.stdout).split('\\n')
    stderr_lines = str(result.stderr).split('\\n')

     # If SUNDIALS failed  
    sundials_failed = False
    for line in stderr_lines:
        if (("test failed repeatedly" in line) or ("mxstep steps taken before reaching tout" in line)):
            sundials_failed = True
    if sundials_failed == True:
        print("Running: " + runcommand + " FAILED")
        stats['ReturnCode']      = 1
        stats['Steps']           = 0
        stats['StepAttempts']    = 0
        stats['ErrTestFails']    = 0
        stats['Explicit_RHS']    = 0 
        stats['Implicit_RHS']    = 0   

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
             # running python file to plot pressure and density
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 
            new_fileName = f"hyperbolic_graph_{solver['name']}_{runN}.png"
            # rename plot file
            if os.path.exists("hyperbolic_relaxation_frames.png"):
                os.rename("hyperbolic_relaxation_frames.png", new_fileName)
                print(f"Plot saved as: {new_fileName}")
            else:
                print("Warning: hyperbolic_relaxation_frames.png not found.")
            #end

            ssp_stdout_lines = str(ssp_result.stdout).split('\\n')
            for line in ssp_stdout_lines:
                txt = line.split()
                if (("grid" in txt) and ("point" in txt) and ("shock" in txt)):
                    tstar = float(txt[13])
                #end
            #end

            ## ==============================================================================
            ## use the t_star to determine time history on the left and right side of the shock
            ## ==============================================================================
            # copy sun.log file into the /sundials/tools folder
            file_to_copy = "sun-%s-%s.log" % (solver['name'], runN) #'./sun.log'
            save_file = "sun-%s-%s" % (solver['name'], runN)
            destination_directory = './../deps/sundials/tools'
            shutil.copy(file_to_copy, destination_directory)

            # change the working directory to sundials/tools
            curent_directory = os.getcwd()
            # print("Current directory:", curent_directory)
            tools_directory  = os.chdir("../deps/sundials/tools")
            tools_directory  = os.getcwd()
            # print("tools directory:", tools_directory)

            # add tstar to time histroy plot
            logcommand = f"./log_example.py {file_to_copy} --tstar %f  --save {save_file}" %(tstar)
            log_result = subprocess.run(shlex.split(logcommand), stdout=subprocess.PIPE)

            # after the tools directory come back to the bin directory
            bin_directory = os.chdir("../../../bin")
            bin_directory  = os.getcwd()
            # print("bin directory:", bin_directory)
        elif (modetype == "fixed"):
            ## running python file to plot pressure and density
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)   
            new_fileName = f"hyperbolic_graph_{solver['name']}_{runN}.png"
            # rename plot file
            if os.path.exists("hyperbolic_relaxation_frames.png"):
                os.rename("hyperbolic_relaxation_frames.png", new_fileName)
                print(f"Plot saved as: {new_fileName}")
            else:
                print("Warning: hyperbolic_relaxation_frames.png not found.")
            #end
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2        --output 2" 
SSP312  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2        --output 2"           
SSPL312 = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2  --output 2"  
SSP423  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3        --output 2"    

## common testing parameters
adaptive_params = {'r1':1e-5, 'r2':1e-4, 'r3':1e-3, 'r4':1e-2, 'r5':1e-1, 'r6':1.0, 'r7':2.0} #relative tolerances
fixed_params    = {} #fixed time step sizes
for i in range(7):
    fixed_params[f"h{i}"] = 0.01/(2.0**i)

## parameters
nonstiff_params = [1e2]
stiff_params    = [1e8]

## Integrator types
solvertype = [{'name': 'SSP212',  'exe': SSP212},
              {'name': 'SSP312',  'exe': SSP312},
              {'name': 'SSPL312', 'exe': SSPL312},
              {'name': 'SSP423',  'exe': SSP423}]

# run tests and collect results as a pandas data frame
fname = 'hyperbolic_relaxation_stats' 
RunStats = []

for knonstiff in nonstiff_params:
    for kstiff in stiff_params:
        for runname, runvalue in adaptive_params.items():
            for solver_adapt in solvertype:
                adaptive_stat = runtest(solver_adapt, "adaptive", runvalue, runname, kstiff, knonstiff, showcommand=True, sspcommand=True)
                RunStats.append(adaptive_stat)

for knonstiff in nonstiff_params:
    for kstiff in stiff_params:
        for runname, runvalue in fixed_params.items():
            for solver_fixed in solvertype:
                fixed_stat = runtest(solver_fixed, "fixed", runvalue, runname, kstiff, knonstiff, showcommand=True, sspcommand=True)
                RunStats.append(fixed_stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)