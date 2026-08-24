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
def runtest(solver, modetype, runV, k1Val, pulseVal, pulseName, showcommand=True, sspcommand=True):
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
             'k1': k1Val, 'k2': k2Val, 'sigma': pulseVal, 'Steps': 0,'StepAttempts': 0, 
             'ErrTestFails': 0, 'Implicit_solves': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 
             'erroru': 0.0, 'errorv': 0.0, 'erroruv': 0.0, 'l1erru': 0.0, 'l1errv': 0.0, 
             'l1erruv': 0.0, 'avg_dt':0.0, 'runtime':0.0}

    if (modetype == "adaptive"):
        runcommand = "%s  --rtol %e  --k1 %e  --k2 %e  --sigma %e" % (solver['exe'], runV, k1Val, k2Val, pulseVal)
    elif (modetype == "fixed"):
        runcommand = "%s  --fixed_h %.6f  --k1 %e  --k2 %e  --sigma %e" % (solver['exe'], runV, k1Val, k2Val, pulseVal)

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
        print("SUNDIALS failed for %s  --val %e  --k1 %e  --k2 %e  --sigma %e" % (solver['exe'], runV, k1Val, k2Val, pulseVal))
        stats['ReturnCode']      = 1
        stats['erroru']          = 0
        stats['errorv']          = 0
        stats['erroruv']         = 0
        stats['l1erru']          = 0
        stats['l1errv']          = 0
        stats['l1erruv']         = 0
        stats['Steps']           = 0
        stats['StepAttempts']    = 0
        stats['ErrTestFails']    = 0
        stats['Explicit_RHS']    = 0     #right hand side evaluations for explicit method
        stats['Implicit_RHS']    = 0     #right hand side evaluations for implicit method
        stats['runtime']         = 0     # runtime should be 0 if test failed
        stats['Implicit_solves'] = 0
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
        stats['avg_dt'] = (1.0 - 0.0) / stats['StepAttempts']  

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

        datafile = "plot_linear_adv_rec.py"
        # return with an error if the file does not exist
        if not os.path.isfile(datafile):
            msg = "Error: file " + datafile + " does not exist"
            sys.exit(msg)

        # Pulse = 0.1   
        if (pulseName == "pulseP1"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if line.startswith("pulseP1 ="):
                    val = "True" 
                    modified_lines.append(f"pulseP1 = {val}\n")
                elif line.startswith("pulseP05 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP05 = {val}\n")
                elif line.startswith("pulseP02 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP02 = {val}\n")
                elif line.startswith("pulseP01 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP01 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        
        # Pulse = 0.05   
        elif (pulseName == "pulseP05"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if line.startswith("pulseP1 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP1 = {val}\n")
                elif line.startswith("pulseP05 ="):
                    val = "True" 
                    modified_lines.append(f"pulseP05 = {val}\n")
                elif line.startswith("pulseP02 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP02 = {val}\n")
                elif line.startswith("pulseP01 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP01 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

        # Pulse = 0.02   
        elif (pulseName == "pulseP02"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if line.startswith("pulseP1 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP1 = {val}\n")
                elif line.startswith("pulseP05 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP05 = {val}\n")
                elif line.startswith("pulseP02 ="):
                    val = "True" 
                    modified_lines.append(f"pulseP02 = {val}\n")
                elif line.startswith("pulseP01 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP01 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        #Pulse = 0.01                
        elif (pulseName == "pulseP01"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if line.startswith("pulseP1 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP1 = {val}\n")
                elif line.startswith("pulseP05 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP05 = {val}\n")
                elif line.startswith("pulseP02 ="):
                    val = "False" 
                    modified_lines.append(f"pulseP02 = {val}\n")
                elif line.startswith("pulseP01 ="):
                    val = "True" 
                    modified_lines.append(f"pulseP01 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        # # ==== end using correct reference solution for each pulse value ====

        ## running python file to plot pressure and density
        sspcommand = " python ./plot_linear_adv_rec.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)   
        ssp_stdout_lines = ssp_result.stdout.decode('utf-8').splitlines()

        ## uncomment the following lines if you want to plot the time evolution for each method, each sigma, each rtol or dt
        ## uncomment the corresponding lines in plot_linear_adv_rec.py to save the frames as png files
        # new_name_frames= f"linear_adv_rec_frames_{modetype}_{solver['name']}_{runV}_{pulseName}.png"
        # if os.path.exists("linear_adv_rec_frames.png"):
        #     os.rename("linear_adv_rec_frames.png", new_name_frames)
        #     print(f"linear_adv_rec frames plot file saved as: {new_name_frames}")
        # else:
        #     print("Warning: linear_adv_rec_frames.png not found.")

        for line in ssp_stdout_lines:
            txt = line.split()
            if (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("u" in txt)):
                stats['erroru'] = float(line.split('=')[-1].strip())
            elif (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("v" in txt)):
                stats['errorv'] = float(line.split('=')[-1].strip())
            elif (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("uv" in txt)):
                stats['erroruv'] = float(line.split('=')[-1].strip())

            if (("L1" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("u" in txt)):
                stats['l1erru'] = float(line.split('=')[-1].strip())
            elif (("L1" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("v" in txt)):
                stats['l1errv'] = float(line.split('=')[-1].strip())
            elif (("L1" in txt) and ("reference" in txt) and ("solution" in txt) and ("for" in txt) and ("uv" in txt)):
                stats['l1erruv'] = float(line.split('=')[-1].strip())

            # ignore errors greater than 10   
            if (stats['erroru'] or stats['l1erru']) > 10.0:
                stats['ReturnCode'] = 1
            elif (stats['errorv'] or stats['l1errv']) > 10.0:
                stats['ReturnCode'] = 1
            elif (stats['erroruv'] or stats['l1erruv'])> 10.0:
                stats['ReturnCode'] = 1
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "./linear_adv_rec --dirk_table ARKODE_SSP_SDIRK_2_1_2       --erk_table ARKODE_SSP_ERK_2_1_2" 
SSP312  = "./linear_adv_rec --dirk_table ARKODE_SSP_DIRK_3_1_2        --erk_table ARKODE_SSP_ERK_3_1_2"           
SSPL312 = "./linear_adv_rec --dirk_table ARKODE_SSP_LSPUM_SDIRK_3_1_2 --erk_table ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP423  = "./linear_adv_rec --dirk_table ARKODE_SSP_ESDIRK_4_2_3      --erk_table ARKODE_SSP_ERK_4_2_3"  
SSP923  = "./linear_adv_rec --dirk_table ARKODE_SSP_ESDIRK_9_2_3      --erk_table ARKODE_SSP_ERK_9_2_3"    

adaptive_params = [1e-8, 1e-7, 1e-6,1e-5, 1e-4, 1e-3, 1e-2]          # relative tolerances
fixed_params = [] # fixed time step sizes
for i in range(10,-2,-1): 
    fixed_params.append(0.005/(2.0**i))

k1values = [1e4]
pulse_steepness = {"pulseP1": 0.1, "pulseP05": 0.05, "pulseP02": 0.02, "pulseP01": 0.01} # steepness of the pulse


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
    for k1_val in k1values:
        for pulse_name, pulse_val in pulse_steepness.items():
            for solver_adapt in solvertype:
                adaptive_stat= runtest(solver_adapt, "adaptive", runvalue, k1_val, pulse_val, pulse_name, showcommand=True, sspcommand=True)
                RunStats.append(adaptive_stat)

for runvalue in fixed_params:
    for k1_val in k1values:
        for pulse_name, pulse_val in pulse_steepness.items():
            for solver_fixed in solvertype:
                fixed_stat = runtest(solver_fixed, "fixed", runvalue, k1_val, pulse_val, pulse_name, showcommand=True, sspcommand=True)
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
# markers  = ['o', '*', 's', '^', '+']
# modetype = ['fixed', 'adaptive']

x_metrics = [('StepAttempts', 'step-attempts','step_attempts'), 
             ('Implicit_solves', 'implicit-solves','implicit_solves'), 
             ('runtime', 'runtime','runtime')]

y_metrics = [('erroru', 'erroru','erroru'), 
             ('errorv', 'errorv','errorv'), 
             ('erroruv', 'erroruv','erroruv')]

for pulse_name, pulse_val in pulse_steepness.items():
    for x_metric, x_label, x_filename in x_metrics:
        for y_metric, y_label, y_filename in y_metrics:
            fig, ax = plt.subplots(figsize=(15, 15))
            for col_ind, k1Val in enumerate(k1values):
                k2Val = 2.0 * k1Val

                #filter data by fixed and adaptive tests
                col_data = df[(df["k1"] == k1Val) & (df["k2"] == k2Val) & (df["sigma"] == pulse_val)]
                data_fixed = col_data[col_data["Runtype"] == "fixed"]
                data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

                # fixed run
                for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                    SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                    valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
                    x = valid_data[x_metric]
                    y = valid_data[y_metric]
                    ax.plot(x, y, color = colors[i], marker = 'o', markersize=5, linestyle='-', label=f"{SSPmethodFix}-h")

                #adaptive run
                for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                    SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                    valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
                    x = valid_data[x_metric]
                    y = valid_data[y_metric]
                    ax.plot(x, y, color = colors[i], marker = '*', markersize=5, linestyle='-.', label=f"{SSPmethodAdapt}-rtol")

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(axis='both', labelsize=18)
            #end

            #remove duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize=20)

            fig.supxlabel(f'{x_label}', fontsize=20)
            fig.supylabel(f'{y_label}', fontsize=20)
            plt.savefig(f"{x_filename}_{y_filename}_LAR_{pulse_name}.png", bbox_inches="tight")

