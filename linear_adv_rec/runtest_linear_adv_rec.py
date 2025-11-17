#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
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
def runtest(solver, modetype, runV, showcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 'Steps': 0, 
             'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 'runtime':0.0, 'L1_norm': 0.0}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %e" % (solver['exe'], runV)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f" % (solver['exe'], runV)
    
    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']    = modetype
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = length_time

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Running: " + runcommand + " SUCCESS")
        lines = str(result.stdout).split('\\n')
        for line in lines:
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
SSP_ARK_212       = "./linear_adv_rec  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./linear_adv_rec  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./linear_adv_rec  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./linear_adv_rec  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"     

adaptive_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]          # relative tolerances
fixed_params    = [1.25*1e-3, 2.50*1e-3, 5.00*1e-3, 1.00*1e-2] # fixed time step sizes


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
    for solver_adapt in solvertype:
        adaptive_stat= runtest(solver_adapt, "adaptive", runvalue, showcommand=True)
        RunStats.append(adaptive_stat)

for runvalue in fixed_params:
    for solver_fixed in solvertype:
        fixed_stat = runtest(solver_fixed, "fixed", runvalue, showcommand=True)
        RunStats.append(fixed_stat)

RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)

##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel('linear_adv_rec_stats' + '.xlsx') # excel file

adapt_accuracy         = True
adapt_efficiency_time  = True
adapt_efficiency_steps = True
fixed_convergence      = True
fixed_efficiency_work  = True
fixed_efficiency_time  = True

  
# # --------------------------------------------------- Run Adaptive Time Steps --------------------------------------------------------------------------------  
data_adaptive = df[(df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "runVal", "Explicit_RHS", "L1_norm", "runtime", "Steps"]]
# if (adapt_accuracy):
#     plt.figure()
#     for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
#         SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
#         # Plot the whole method line with '.' markers
#         method_line = plt.plot(SSPmethodAdt_data['runVal'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
#         method_line_color = method_line[0].get_color()
#         # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#         sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
#         plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('rtol')
#     plt.ylabel('$L_{\\infty}$ error')
#     plt.legend()
#     plt.savefig("Adaptive accuracy plot with d = %.2f.pdf"%dck)
#     plt.show()

if (adapt_efficiency_time):
    plt.figure()
    for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
        SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
        method_line = plt.plot(SSPmethodAdt_data['runtime'], SSPmethodAdt_data['L1_norm'], marker='.', linestyle='-', label=SSPmethodAdt)
        method_line_color = method_line[0].get_color()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('runtime')
    plt.ylabel('$L_{1}$ error')
    plt.legend()
    plt.savefig("Adaptive efficiency time plot")
    plt.show()

# if (adapt_efficiency_steps):
#     plt.figure()
#     for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
#         SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
#         # Plot the whole method line with '.' markers
#         method_line = plt.plot(SSPmethodAdt_data['Steps'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
#         method_line_color = method_line[0].get_color()
#         # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#         sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
#         plt.plot(sspness['Steps'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('number of steps')
#     plt.ylabel('$L_{\\infty}$ error')
#     plt.legend()
#     plt.savefig("Adaptive efficiency steps plot with d = %.2f.pdf"%dck)
#     plt.show()

# # # ------------------------------------------------ Run Fixed Time Steps ---------------------------------------------------------------------------            
#     data_fixed = df[(df["diff_coef"] == dck) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
#                                                                             "Total Func Eval", "error", "maxIntStep", "Negative_model", "sspCondition", "runtime", "Steps"]]
#     if (fixed_convergence):
#         plt.figure()
#         for SSPmethodFix in data_fixed['IMEX_method'].unique():
#             SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
#             # Plot the whole method line with '.' markers
#             method_line = plt.plot(SSPmethodFix_data['runVal'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
#             method_line_color = method_line[0].get_color()
#             # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#             sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
#             plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('h')
#         plt.ylabel('$L_{\\infty}$ error')
#         plt.legend()
#         plt.savefig("Fixed convergence plot with d = %.2f.pdf"%dck)
#         plt.show()

# if (fixed_efficiency_time):
#     plt.figure()
#     for SSPmethodFix in data_fixed['IMEX_method'].unique():
#         SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
#         # Plot the whole method line with '.' markers
#         method_line = plt.plot(SSPmethodFix_data['runtime'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
#         method_line_color = method_line[0].get_color()
#         # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#         sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
#         plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('runtime')
#     plt.ylabel('$L_{\\infty}$ error')
#     plt.legend()
#     plt.savefig("Fixed effciency time plot for d = %.2f.pdf"%dck)
#     plt.show()

# if (fixed_efficiency_work):
#     plt.figure()
#     for SSPmethodFix in data_fixed['IMEX_method'].unique():
#         SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
#         # Plot the whole method line with '.' markers
#         method_line = plt.plot(SSPmethodFix_data['Total Func Eval'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
#         method_line_color = method_line[0].get_color()
#         # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#         sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
#         plt.plot(sspness['Total Func Eval'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#     plt.xscale('log')
#     plt.yscale('log')
#     plt.xlabel('Total Num of Func Evals')
#     plt.ylabel('$L_{\\infty}$ error')
#     plt.legend()
#     plt.savefig("Fixed effciency work for d = %.2f.pdf"%dck)
#     plt.show()




