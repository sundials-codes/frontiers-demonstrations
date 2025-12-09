#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: This scripts runs the different imex schemes with different diffusion coefficients and parameters, 
#         using either adaptive or fixed time stepping, for a linear advection-reaction test problem.
#         The goal is to test the accuracy of the IMEX SSP schemes.
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
def runtest(solver, modetype, runV, k1Val, k2Val, showcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           k1Val, k2Val:      stiffness parameters

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 
             'k1': k1Val, 'k2': k2Val, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0,
             'Explicit_RHS': 0, 'Implicit_RHS': 0,  'L1_norm': 0.0, 'runtime':0.0}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %e  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val)
    
    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']    = modetype
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = length_time

    stdout_lines = str(result.stdout).split('\\n')
    stderr_lines = str(result.stderr).split('\\n')
    # print("stdout\n", stdout_lines)
    # print("stderr\n", stderr_lines)

    # If SUNDIALS failed  
    sundials_failed = False
    for line in stderr_lines:
        if ("the error test failed repeatedly" in line):
            sundials_failed = True
    
    if sundials_failed == True:
        print("SUNDIALS failed for %s  --rtol %e  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val))
        stats['ReturnCode']   = 1
        stats['L1_norm']      = 0
        stats['Steps']        = 0
        stats['StepAttempts'] = 0
        stats['ErrTestFails'] = 0
        stats['Explicit_RHS'] = 0     #right hand side evaluations for explicit method
        stats['Implicit_RHS'] = 0       #right hand side evaluations for implicit method

    # If SUNDIALS did not fail
    if not sundials_failed:
        print("Running: " + runcommand + " SUCCESS")
        for line in stdout_lines:
            txt = line.split()
            if ("L1-norm" in txt):
                stats['L1_norm'] = float(txt[2])
            elif ("Steps" in txt):
                stats['Steps'] = int(txt[2])
            elif (("Step" in txt) and ("attempts" in txt)):
                stats['StepAttempts'] = int(txt[3])
            elif (("Error" in txt) and ("Fails" in txt)):
                stats['ErrTestFails'] = float(txt[4])
            elif (("Explicit" in txt) and ("RHS" in txt)):
                stats['Explicit_RHS'] = int(txt[5])       #right hand side evaluations for explicit method
            elif (("Implicit" in txt) and ("RHS" in txt)):
                stats['Implicit_RHS'] = int(txt[5])       #right hand side evaluations for implicit method   
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP_ARK_212       = "./linear_adv_rec --IMintegrator ARKODE_SSP_SDIRK_2_1_2       --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./linear_adv_rec --IMintegrator ARKODE_SSP_DIRK_3_1_2        --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./linear_adv_rec --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2 --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./linear_adv_rec --IMintegrator ARKODE_SSP_ESDIRK_4_2_3      --EXintegrator ARKODE_SSP_ERK_4_2_3"     

adaptive_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]          # relative tolerances
# fixed_params  = [6.25*1e-4, 1.25*1e-3, 2.50*1e-3, 5.00*1e-3, 1.00*1e-2] # fixed time step sizes
fixed_params    = [] # fixed time step sizes
for i in range(10,-1,-1):
    fixed_params.append(0.01/(2.0**i))
#end
k1values = [1.0, 1e6]
k2values = [1.0, 2e6]


# ----------------------------------------------------------------------------------------------------
# This section generates the data for each method with different fixed step sizes and rtols
# ----------------------------------------------------------------------------------------------------

# Integrator types
solvertype = [{'name': 'SSP-ARK-2-1-2',       'exe': SSP_ARK_212},
              {'name': 'SSP-ARK-3-1-2',       'exe': SSP_ARK_312},
              {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312},
              {'name': 'SSP-ARK-4-2-3',       'exe': SSP_ARK_423}]

# run tests and collect results as a pandas data frame
fname = 'linear_adv_rec_stats' 
RunStats = []

for runvalue in adaptive_params:
    for k1_val, k2_val in zip(k1values, k2values):
        for solver_adapt in solvertype:
            adaptive_stat= runtest(solver_adapt, "adaptive", runvalue, k1_val, k2_val, showcommand=True)
            RunStats.append(adaptive_stat)

for runvalue in fixed_params:
    for k1_val, k2_val in zip(k1values, k2values):
        for solver_fixed in solvertype:
            fixed_stat = runtest(solver_fixed, "fixed", runvalue, k1_val, k2_val, showcommand=True)
            RunStats.append(fixed_stat)

RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)

##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel('linear_adv_rec_stats' + '.xlsx') # excel file

adapt_efficiency_time_stiffRec     = True
adapt_efficiency_time_nonstiffRec  = True

# # --------------------------------------------------- Run Adaptive Time Steps --------------------------------------------------------------------------------  
data_adaptive_stiffRec = df[(df["Runtype"] == "adaptive") & (df["k1"] == 1e6) & (df["k2"] == 2e6)][["Runtype", "IMEX_method", "runVal", "k1", "k2", "Explicit_RHS", "L1_norm", "runtime", "Steps"]]

if (adapt_efficiency_time_stiffRec):
    plt.figure()
    for SSPmethodAdt in data_adaptive_stiffRec['IMEX_method'].unique():
        SSPmethodAdt_data = data_adaptive_stiffRec[data_adaptive_stiffRec['IMEX_method'] == SSPmethodAdt]
        method_line = plt.plot(SSPmethodAdt_data['runtime'], SSPmethodAdt_data['L1_norm'], marker='.', linestyle='-', label=SSPmethodAdt)
        method_line_color = method_line[0].get_color()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('runtime')
    plt.ylabel('$L_{1}$ error')
    plt.legend()
    plt.savefig("adaptive_efficiency_time_linear_adv_stiffRec")
    # plt.show()


data_adaptive_nonstiffRec = df[(df["Runtype"] == "adaptive") & (df["k1"] == 1) & (df["k2"] == 1)][["Runtype", "IMEX_method", "runVal", "k1", "k2", "Explicit_RHS", "L1_norm", "runtime", "Steps"]]

if (adapt_efficiency_time_nonstiffRec):
    plt.figure()
    for SSPmethodAdt in data_adaptive_nonstiffRec['IMEX_method'].unique():
        SSPmethodAdt_data = data_adaptive_nonstiffRec[data_adaptive_nonstiffRec['IMEX_method'] == SSPmethodAdt]
        method_line = plt.plot(SSPmethodAdt_data['runtime'], SSPmethodAdt_data['L1_norm'], marker='.', linestyle='-', label=SSPmethodAdt)
        method_line_color = method_line[0].get_color()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('runtime')
    plt.ylabel('$L_{1}$ error')
    plt.legend()
    plt.savefig("adaptive_efficiency_time_linear_adv_nonstiffRec")
    # plt.show()




