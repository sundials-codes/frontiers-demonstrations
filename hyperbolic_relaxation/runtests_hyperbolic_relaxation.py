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
def runtest(solver, modetype, runV, showcommand=True, sspcommand=True):
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
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'runVal': runV, 'runtime':0.0, 
             'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 
             'Implicit_solves': 0, 'err_rho': 0.0, 'energy_err': 0.0, 'avg_dt':0.0}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %e  " % (solver['exe'], runV)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %e  " % (solver['exe'], runV)
    
    start_time = time.time()
    result = subprocess.run(runcommand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']        = modetype
    stats['ReturnCode']     = result.returncode
    stats['runtime']        = length_time

    stdout_lines = result.stdout.decode().split('\n')
    stderr_lines = result.stderr.decode().split('\n')

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
        stats['Implicit_solves'] = 0  
        stats['err_rho']         = 0 
        stats['energy_err']      = 0
        stats['runtime']         = 0     # runtime should be 0 is test failed
        stats['avg_dt']          = 0

    # If SUNDIALS did not fail
    if not sundials_failed:
        print("Running: " + runcommand + " SUCCESS")
        for line in stdout_lines:
            txt = line.split()
            if ("Steps" in txt):
                stats['Steps'] = int(txt[2])
            elif (("Step" in txt) and ("attempts" in txt)):
                stats['StepAttempts'] = int(txt[3])
            elif (("Error" in txt) and ("fails" in txt)):
                stats['ErrTestFails'] = float(txt[4])
            elif (("Explicit" in txt) and ("RHS" in txt)):
                stats['Explicit_RHS'] = int(txt[5])       #right hand side evaluations for explicit method
            elif (("Implicit" in txt) and ("RHS" in txt)):
                stats['Implicit_RHS'] = int(txt[5])       #right hand side evaluations for implicit method

        stats['avg_dt'] = (0.3 - 0.0) / stats['StepAttempts'] 

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

        datafile = "plot_hyperbolic_relaxation.py"
        # return with an error if the file does not exist
        if not os.path.isfile(datafile):
            msg = "Error: file " + datafile + " does not exist"
            sys.exit(msg)
                
            # running python file to plot pressure and density
        sspcommand = " python ./plot_hyperbolic_relaxation.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
        if ssp_result.returncode != 0:
            sys.exit("plot script FAILED")

        ssp_stdout_lines = str(ssp_result.stdout).split('\\n')
        for line in ssp_stdout_lines:
            txt = line.split()
            if (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt)):
                stats['err_rho'] = float(txt[6])
            elif (("Maximum" in txt) and ("energy" in txt) and ("error" in txt)):
                stats['energy_err'] = float(txt[4])

    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "  ./hyperbolic_relaxation  --dirk_table ARKODE_SSP_SDIRK_2_1_2        --erk_table ARKODE_SSP_ERK_2_1_2        --output 2" 
SSP312  = "  ./hyperbolic_relaxation  --dirk_table ARKODE_SSP_DIRK_3_1_2         --erk_table ARKODE_SSP_ERK_3_1_2        --output 2"           
SSPL312 = "  ./hyperbolic_relaxation  --dirk_table ARKODE_SSP_LSPUM_SDIRK_3_1_2  --erk_table ARKODE_SSP_LSPUM_ERK_3_1_2  --output 2"  
SSP423  = "  ./hyperbolic_relaxation  --dirk_table ARKODE_SSP_ESDIRK_4_2_3       --erk_table ARKODE_SSP_ERK_4_2_3        --output 2"   
SSP923  = "  ./hyperbolic_relaxation  --dirk_table ARKODE_SSP_ESDIRK_9_2_3       --erk_table ARKODE_SSP_ERK_9_2_3        --output 2"   

## common testing parameters
adaptive_params = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] 
fixed_params    = [] 
for i in range(18, -2, -1): 
    fixed_params.append(0.01/(2.0**i))

## Integrator types
solvertype = [{'name': 'SSP212',  'exe': SSP212},
              {'name': 'SSP312',  'exe': SSP312},
              {'name': 'SSPL312', 'exe': SSPL312},
              {'name': 'SSP423',  'exe': SSP423},
              {'name': 'SSP923',  'exe': SSP923}]
              
# run tests and collect results as a pandas data frame
fname = 'hyperbolic_relaxation_stats' 
RunStats = []

for runvalue in adaptive_params:
    for solver_adapt in solvertype:
        adaptive_stat = runtest(solver_adapt, "adaptive", runvalue, showcommand=True, sspcommand=True)
        RunStats.append(adaptive_stat)

for runvalue in fixed_params:
    for solver_fixed in solvertype:
        fixed_stat = runtest(solver_fixed, "fixed", runvalue, showcommand=True, sspcommand=True)
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
df = pd.read_excel('hyperbolic_relaxation_stats' + '.xlsx') # excel file
methods = df['IMEX_method'].unique()
colors   = ['red', 'black', 'blue', 'green', 'orange'] 

x_metrics = [('StepAttempts', 'step-attempts','step_attempts'), 
             ('Implicit_solves', 'implicit-solves','implicit_solves'), 
             ('runtime', 'runtime','runtime')]

y_metrics = [('err_rho', 'err_rho','err_rho')]

for x_metric, x_label, x_filename in x_metrics:
    for y_metric, y_label, y_filename in y_metrics:
        fig, ax = plt.subplots( figsize=(7, 6))

        data_fixed = df[df["Runtype"] == "fixed"]
        data_adaptive = df[df["Runtype"] == "adaptive"]

        # fixed run
        for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
            x = valid_data[x_metric]
            y = valid_data[y_metric]
            ax.plot(x, y, color = colors[i], marker = 'o', markersize=5, linestyle='-', linewidth=2,label=f"{SSPmethodFix}-h")
    
        #adaptive run
        for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
            SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
            valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
            x = valid_data[x_metric]
            y = valid_data[y_metric]
            ax.plot(x, y, color = colors[i], marker = '*', markersize=5, linestyle='-.', linewidth=2,label=f"{SSPmethodAdapt}-rtol")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=13)

        #remove duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize=10)

        fig.supxlabel(f'{x_label}', fontsize=11)
        fig.supylabel(f'{y_label}', fontsize=11)
        plt.savefig(f"{x_filename}_{y_filename}_hyperbolic.png", bbox_inches="tight")
        plt.close(fig)

