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
def runtest(solver, modetype, runV, k1Val, showcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver           : imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV             : rtol (adaptive) or fixed_h (fixed)
           k1Val            : stiffness parameters

    Output: returns the statistics
    """

    k2Val = 2.0 * k1Val

    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 
             'k1': k1Val, 'k2': k2Val, 'Steps': 0,'StepAttempts': 0, 'ErrTestFails': 0, 'Implicit_solves': 0,
             'Explicit_RHS': 0, 'Implicit_RHS': 0,'maxIntStep': 0.0, 'L1_norm': 0.0, 'runtime':0.0}

    if (modetype == "adaptive"):
        runcommand = "%s  --rtol %e  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val)
    elif (modetype == "fixed"):
        runcommand = "%s  --fixed_h %.6f  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val)
    
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
        if ("the error test failed repeatedly" in line):
            sundials_failed = True
    
    if sundials_failed == True:
        print("SUNDIALS failed for %s  --rtol %e  --k1 %e  --k2 %e" % (solver['exe'], runV, k1Val, k2Val))
        stats['ReturnCode']      = 1
        stats['L1_norm']         = 0
        stats['Steps']           = 0
        stats['StepAttempts']    = 0
        stats['ErrTestFails']    = 0
        stats['Explicit_RHS']    = 0     #right hand side evaluations for explicit method
        stats['Implicit_RHS']    = 0     #right hand side evaluations for implicit method
        stats['runtime']         = 0     # runtime should be 0 is test failed
        stats['Implicit_solves'] = 0
        stats['maxIntStep']      = 0

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
            elif (("Error" in txt) and ("fails" in txt)):
                stats['ErrTestFails'] = float(txt[4])
            elif (("Explicit" in txt) and ("RHS" in txt)):
                stats['Explicit_RHS'] = int(txt[5])       #right hand side evaluations for explicit method
            elif (("Implicit" in txt) and ("RHS" in txt)):
                stats['Implicit_RHS'] = int(txt[5])       #right hand side evaluations for implicit method   
            elif (("Largest" in txt) and ("avg" in txt) and ("step" in txt) and ("size" in txt)):
                stats['maxIntStep'] = float(txt[7])         #last internal step size used in adaptive run

        # number of implicit solves for each method
        if (solver['name']== 'SSP212'):
            stats['Implicit_solves'] = 2 * stats['StepAttempts']
        elif (solver['name']== 'SSP312'):
            stats['Implicit_solves'] = 3 * stats['StepAttempts']
        elif (solver['name']== 'SSPL312'):
            stats['Implicit_solves'] = 3 * stats['StepAttempts']
        elif (solver['name']== 'SSP423'):
            stats['Implicit_solves'] = 3 * stats['StepAttempts']
        elif (solver['name']== 'SSP923'):
            stats['Implicit_solves'] = 4 * stats['StepAttempts']
        # end
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "./linear_adv_rec --IMintegrator ARKODE_SSP_SDIRK_2_1_2       --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP312  = "./linear_adv_rec --IMintegrator ARKODE_SSP_DIRK_3_1_2        --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSPL312 = "./linear_adv_rec --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2 --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP423  = "./linear_adv_rec --IMintegrator ARKODE_SSP_ESDIRK_4_2_3      --EXintegrator ARKODE_SSP_ERK_4_2_3"  
SSP923  = "./linear_adv_rec --IMintegrator ARKODE_SSP_ESDIRK_9_2_3      --EXintegrator ARKODE_SSP_ERK_9_2_3"    

adaptive_params = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]          # relative tolerances
fixed_params    = [] # fixed time step sizes
for i in range(10,-4,-1):
    fixed_params.append(0.01/(2.0**i))
#end
k1values = [1.0, 1e6]


# ----------------------------------------------------------------------------------------------------
# This section generates the data for each method with different fixed step sizes and rtols
# ----------------------------------------------------------------------------------------------------

# Integrator types
solvertype = [{'name': 'SSP212',  'exe': SSP212},
              {'name': 'SSP312',  'exe': SSP312},
              {'name': 'SSPL312', 'exe': SSPL312},
              {'name': 'SSP423',  'exe': SSP423},
              {'name': 'SSP923',  'exe': SSP923}]
              

# run tests and collect results as a pandas data frame
fname = 'linear_adv_rec_stats' 
RunStats = []

for runvalue in adaptive_params:
    # for k1_val, k2_val in zip(k1values, k2values):
    for k1_val in k1values:
        for solver_adapt in solvertype:
            adaptive_stat= runtest(solver_adapt, "adaptive", runvalue, k1_val, showcommand=True)
            RunStats.append(adaptive_stat)

for runvalue in fixed_params:
    # for k1_val, k2_val in zip(k1values, k2values):
    for k1_val in k1values:
        for solver_fixed in solvertype:
            fixed_stat = runtest(solver_fixed, "fixed", runvalue, k1_val, showcommand=True)
            RunStats.append(fixed_stat)

RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)


# ===============================================================================================================================
#  Generate plots to test the efficiency and accuracy of the IMEX SSP methods
# ===============================================================================================================================
df = pd.read_excel('linear_adv_rec_stats' + '.xlsx') # excel file
methods = df['IMEX_method'].unique()

colors   = ['red', 'black', 'blue', 'green', 'orange'] 
markers  = ['o', '*', 's', '^', '+']
modetype = ['fixed', 'adaptive']

# --------------------------- accepted steps vs error ----------------------------------
#create a figure of subplots (columns are stiffness parameters and rows are methods)
fig, axes = plt.subplots(2, len(k1values), figsize=(15, 12))
for col_ind, k1Val in enumerate(k1values):
    k2Val = 2.0 * k1Val
    data_fixed    = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "fixed")]
    data_adaptive = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "adaptive")]

    for row_ind, runtype in enumerate(modetype):
        if row_ind == 0:
        # fixed run
            for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                x = SSPmethodFix_data['StepAttempts'].where(SSPmethodFix_data['ReturnCode'] != 1)
                y = SSPmethodFix_data['L1_norm'].where(SSPmethodFix_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-', label=SSPmethodFix)

        elif row_ind == 1:
        # adaptive run
            for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                x = SSPmethodAdapt_data['StepAttempts'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                y = SSPmethodAdapt_data['L1_norm'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x,y, color = colors[i], marker = markers[i], markersize=10, linestyle='-.', label=SSPmethodAdapt)

        # each row should correspond to a runtype
        if col_ind==0:
            axes[row_ind,col_ind].set_ylabel(f"{runtype}", fontsize=15)
        #end

        # each column should correspond to a stiffness parameter
        if row_ind == 0:
            axes[row_ind,col_ind].set_title(f"k1 = {k1Val: .1e}, k2 = {k2Val: .1e}", fontsize=18)
        #end
        
    #end
        axes[row_ind,col_ind].set_xscale('log')
        axes[row_ind,col_ind].set_yscale('log')
        axes[row_ind,col_ind].set_ylim([1e-18, 1e2])
        axes[row_ind,col_ind].legend(loc="best")
#end
fig.supxlabel('step attempts', fontsize=18)
fig.supylabel('$L_{1}$ error', fontsize=18)
fig.suptitle("step attempts vs error", fontsize=20)
fig.tight_layout()
plt.savefig("step_attempts_error_linear_adv_rec.png")


# --------------------------- implicit solves vs error ----------------------------------
#create a figure of subplots (columns are stiffness parameters and rows are methods)
fig, axes = plt.subplots(2, len(k1values), figsize=(15, 12))
for col_ind, k1Val in enumerate(k1values):
    k2Val = 2.0 * k1Val
    data_fixed    = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "fixed")]
    data_adaptive = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "adaptive")]

    for row_ind, runtype in enumerate(modetype):
        # fixed run
        if row_ind == 0:
            for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                x = SSPmethodFix_data['Implicit_solves'].where(SSPmethodFix_data['ReturnCode'] != 1)
                y = SSPmethodFix_data['L1_norm'].where(SSPmethodFix_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-', label=SSPmethodFix)

        #adaptive run
        elif row_ind == 1:
            for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                x = SSPmethodAdapt_data['Implicit_solves'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                y = SSPmethodAdapt_data['L1_norm'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-.', label=SSPmethodAdapt)

        # each row should correspond to a runtype
        if col_ind==0:
            axes[row_ind,col_ind].set_ylabel(f"{runtype}", fontsize=15)
        #end

        # each column should correspond to a stiffness parameter
        if row_ind == 0:
            axes[row_ind,col_ind].set_title(f"k1 = {k1Val: .1e}, k2 = {k2Val: .1e}", fontsize=18)
        #end
    #end
        axes[row_ind,col_ind].set_xscale('log')
        axes[row_ind,col_ind].set_yscale('log')
        axes[row_ind,col_ind].set_ylim([1e-18, 1e2])
        axes[row_ind,col_ind].legend(loc="best")
#end
fig.supxlabel('implicit solves', fontsize=18)
fig.supylabel('$L_{1}$ error', fontsize=18)
fig.suptitle("implicit solves vs error", fontsize=20)
fig.tight_layout()
plt.savefig("implicit_solves_error_linear_adv_rec.png")


# --------------------------- runtime vs error ----------------------------------
#create a figure of subplots (columns are stiffness parameters and rows are methods)
fig, axes = plt.subplots(2, len(k1values), figsize=(15, 12))
for col_ind, k1Val in enumerate(k1values):
    k2Val = 2.0 * k1Val
    data_fixed    = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "fixed")]
    data_adaptive = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "adaptive")]

    for row_ind, runtype in enumerate(modetype):
        # fixed run
        if row_ind == 0:
            for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                x = SSPmethodFix_data['runtime'].where(SSPmethodFix_data['ReturnCode'] != 1)
                y = SSPmethodFix_data['L1_norm'].where(SSPmethodFix_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-', label=SSPmethodFix)

        #adaptive run
        elif row_ind == 1:
            for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                x = SSPmethodAdapt_data['runtime'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                y = SSPmethodAdapt_data['L1_norm'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
                axes[row_ind,col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-.', label=SSPmethodAdapt)

        # each row should correspond to a runtype
        if col_ind==0:
            axes[row_ind,col_ind].set_ylabel(f"{runtype}", fontsize=15)
        #end

        # each column should correspond to a stiffness parameter
        if row_ind == 0:
            axes[row_ind,col_ind].set_title(f"k1 = {k1Val: .1e}, k2 = {k2Val: .1e}", fontsize=18)
        #end
    #end
        axes[row_ind,col_ind].set_xscale('log')
        axes[row_ind,col_ind].set_yscale('log')
        axes[row_ind,col_ind].set_ylim([1e-18, 1e2])
        axes[row_ind,col_ind].legend(loc="best")
#end
fig.supxlabel('runtime', fontsize=18)
fig.supylabel('$L_{1}$ error', fontsize=18)
fig.suptitle("runtime vs error", fontsize=20)
fig.tight_layout()
plt.savefig("runtime_error_linear_adv_rec.png")




# --------------------------- rtol vs error ----------------------------------
#create a figure of subplots (columns are stiffness parameters and rows are methods)
fig, axes = plt.subplots(1, len(k1values), figsize=(15, 12))
for col_ind, k1Val in enumerate(k1values):
    k2Val = 2.0 * k1Val
    data_adaptive    = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "adaptive")]

    for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
        SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
        x = SSPmethodAdapt_data['runVal'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
        y = SSPmethodAdapt_data['L1_norm'].where(SSPmethodAdapt_data['ReturnCode'] != 1)
        axes[col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-', label=SSPmethodAdapt)

    # each row should correspond to a runtype
    if col_ind==0:
        axes[col_ind].set_ylabel("fixed", fontsize=15)
    #end

    axes[0].set_title(f"k1 = {k1values[0]}", fontsize=18)
    axes[1].set_title(f"k1 = {k1values[1]}", fontsize=18)
    
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].set_ylim([1e-18, 1e2])
    axes[col_ind].legend(loc="best")
#end
fig.supxlabel('rtol', fontsize=18)
fig.supylabel('$L_{1}$ error', fontsize=18)
fig.suptitle("rtol vs error", fontsize=20)
fig.tight_layout()
plt.savefig("rtol_error_linear_adv_rec.png")


# --------------------------- fixedh vs error ----------------------------------
#create a figure of subplots (columns are stiffness parameters and rows are methods)
fig, axes = plt.subplots(1, len(k1values), figsize=(15, 12))
for col_ind, k1Val in enumerate(k1values):
    k2Val = 2.0 * k1Val
    data_fixed    = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["Runtype"] == "fixed")]

    for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
        SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
        x = SSPmethodFix_data['runtime'].where(SSPmethodFix_data['ReturnCode'] != 1)
        y = SSPmethodFix_data['L1_norm'].where(SSPmethodFix_data['ReturnCode'] != 1)
        axes[col_ind].plot(x, y, color = colors[i], marker = markers[i], markersize=10, linestyle='-', label=SSPmethodFix)

    # each row should correspond to a runtype
    if col_ind==0:
        axes[col_ind].set_ylabel("fixed", fontsize=15)
    #end

    axes[0].set_title(f"k1 = {k1values[0]}", fontsize=18)
    axes[1].set_title(f"k1 = {k1values[1]}", fontsize=18)
    
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].set_ylim([1e-18, 1e2])
    axes[col_ind].legend(loc="best")
#end
fig.supxlabel('h', fontsize=18)
fig.supylabel('$L_{1}$ error', fontsize=18)
fig.suptitle("h vs error", fontsize=20)
fig.tight_layout()
plt.savefig("fixedh_error_linear_adv_rec.png")