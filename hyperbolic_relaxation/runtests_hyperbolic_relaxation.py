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
def runtest(solver, modetype, runV, showcommand=True, sspcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV,
            'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 
            'l2_error': 0.0, 'runtime':0.0}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %e --output 2" % (solver['exe'], runV)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f --output 2" % (solver['exe'], runV)
    
    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
            # elif (("L2" in txt) and ("error" in txt) and ("norm" in txt)):
            #     stats['l2_error'] = float(txt[4])   #l2 error for the differenc ebetween the 

            ## running python file to plot pressure and density
        sspcommand = " python ./plot_hyperbolic_relaxation.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 

        if (sspcommand):
            print("Run solution graph: " + sspcommand + " SUCCESS")
            new_fileName = f"soln_graph_{solver['name']}_{runV}.png"

            ## rename plot file
            if os.path.exists("hyperbolic_relaxation_frames.png"):
                os.rename("hyperbolic_relaxation_frames.png", new_fileName)
                print(f"Plot saved as: {new_fileName}")
            else:
                print("Warning: hyperbolic_relaxation_frames.png not found.")   
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP_ARK_212       = "./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"     

adaptive_params = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1] ## relative tolerances
fixed_params    = [] # fixed time step sizes
for i in range(8,-1,-1):
    fixed_params.append(0.002/(2.0**i))


## ----------------------------------------------------------------------------------------------------
# This section generates the data for each method, diffusion coefficient with different fixed step
# sizes and rtols
## ----------------------------------------------------------------------------------------------------
sorted_adaptive_params = sorted(adaptive_params) ## relative tolerances
sorted_fixed_params    = sorted(fixed_params) ## fixed time step sizes

## Integrator types
solvertype = [{'name': 'SSP-ARK-2-1-2',       'exe': SSP_ARK_212},
              {'name': 'SSP-ARK-3-1-2',       'exe': SSP_ARK_312},
              {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312},
              {'name': 'SSP-ARK-4-2-3',       'exe': SSP_ARK_423}]

# run tests and collect results as a pandas data frame
fname = 'hyperbolic_relaxation_stats' 
RunStats = []

for runvalue in sorted_adaptive_params:
    for solver_adapt in solvertype:
        adaptive_stat = runtest(solver_adapt, "adaptive", runvalue, showcommand=True, sspcommand=True)
        RunStats.append(adaptive_stat)

for runvalue in sorted_fixed_params:
    for solver_fixed in solvertype:
        fixed_stat = runtest(solver_fixed, "fixed", runvalue, showcommand=True, sspcommand=True)
        RunStats.append(fixed_stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)



# # -----------------------------------------------------------------------------------------
# # This section generates accuracy, convergence and efficiency plots
# # -----------------------------------------------------------------------------------------

# df = pd.read_excel('population_density_imex_stats' + '.xlsx') # excel file

# diff_coeff = {'k0':0.0,'k2':0.02, 'k4':0.04} #diffusion coefficients

# adapt_accuracy         = True
# adapt_efficiency_time  = True
# fixed_convergence      = True
# fixed_efficiency_work  = True

# for kname, kval in diff_coeff.items():    
# # ------------ adaptive run ---------------  
#     data_adaptive = df[(df["diff_coef"] == kval) & (df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
#                                                                                   "Total Func Eval", "maxIntStep", "error", "Negative_model", "sspCondition", "runtime", "Steps"]]
#     if (adapt_accuracy):
#         plt.figure()
#         for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
#             SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
#             # Plot the whole method line with '.' markers
#             method_line = plt.plot(SSPmethodAdt_data['runVal'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
#             method_line_color = method_line[0].get_color()
#             # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#             sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
#             plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('rtol')
#         plt.ylabel('$L_{\\infty}$ error')
#         plt.legend()
#         plt.savefig(f"popu_adaptive_accuracy_{kname}.pdf")
#         # plt.show()

#     if (adapt_efficiency_time):
#         plt.figure()
#         for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
#             SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
#             # Plot the whole method line with '.' markers
#             method_line = plt.plot(SSPmethodAdt_data['runtime'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
#             method_line_color = method_line[0].get_color()
#             # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#             sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
#             plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('runtime')
#         plt.ylabel('$L_{\\infty}$ error')
#         plt.legend()
#         plt.savefig(f"popu_adaptive_efficiency_time_{kname}.pdf")
#         # plt.show()

# # --------------- fixed run ----------------            
#     data_fixed = df[(df["diff_coef"] == kval) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
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
#         plt.savefig(f"popu_fixed_convergence_{kname}.pdf")
#         # plt.show()

#     if (fixed_efficiency_work):
#         plt.figure()
#         for SSPmethodFix in data_fixed['IMEX_method'].unique():
#             SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
#             # Plot the whole method line with '.' markers
#             method_line = plt.plot(SSPmethodFix_data['Total Func Eval'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
#             method_line_color = method_line[0].get_color()
#             # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
#             sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
#             plt.plot(sspness['Total Func Eval'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
#         plt.xscale('log')
#         plt.yscale('log')
#         plt.xlabel('Total Num of Func Evals')
#         plt.ylabel('$L_{\\infty}$ error')
#         plt.legend()
#         plt.savefig(f"popu_fixed_efficiency_work_{kname}.pdf")
#         # plt.show()



