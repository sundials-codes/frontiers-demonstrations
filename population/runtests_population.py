#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, University of Maryland Baltimore County.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# Description: This scripts runs the different imex schemes with different diffusion coefficients and parameters,
#              using either adaptive or fixed time stepping for the population density model.
#              The stats for the fixed runs and the adaptive runs, for the efficiency plots are stored in two separate excel files.
#              The excel file population_all_runs.xlsx contains the stats for all runs.
#              The excel file population_brackets.xlsx contains the brackets for each method and diffusion coefficient to use for the bisection method.
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
from math import log10, floor, isnan

# smoothness tolerances (tau_low, tau_high): final profile is smooth if D_ref*(1 - tau_low) <= D <= D_ref*(1 + tau_high)
smooth_tol = {0.0: (0.0015, 0.0015), 0.02: (0.17, 0.02), 0.04: (0.17, 0.02)}

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, modetype, runV, kVal, kName, showcommand=True, sspcommand=True):
    """
    This function runs the population model using either fixed or adaptive time
    stepping with different parameters and returns the stats

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient
           kName:             diffusion coefficient name

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 
             'runVal': runV, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 
             'Implicit_RHS': 0, 'Implicit_solves':0,  'Negative_model': 0,
             'lmax_1dev': 0.0, 'lmax_1dev_ref': 0.0, 'rel_diff_1dev': 0.0,
             'error': 0.0, 'runtime':0.0, 'avg_dt':0.0, 'sspCondition': " "}

    if (modetype == "adaptive"):
        runcommand = " %s   --rtol %e   --k %e" % (solver['exe'], runV, kVal)
    elif (modetype == "fixed"):
        runcommand = " %s   --fixed_h %e   --k %e" % (solver['exe'], runV, kVal)
    
    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']    = modetype
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = length_time

    stdout_lines = result.stdout.decode('utf-8').splitlines()
    stderr_lines = result.stderr.decode('utf-8').splitlines()

     # If SUNDIALS failed  
    sundials_failed = False
    for line in stderr_lines:
        if ("test failed repeatedly" in line):
            sundials_failed = True

    if sundials_failed == True:
        if (modetype == "adaptive"):
            print("SUNDIALS failed for  %s  --rtol %e --k %e" % (solver['exe'], runV, kVal))
        elif (modetype == "fixed"):
            print("SUNDIALS failed for %s  --fixed_h %e --k %e" % (solver['exe'], runV, kVal))
        stats['ReturnCode']       = 1
        stats['error']            = 0
        stats['Steps']            = 0
        stats['StepAttempts']     = 0
        stats['ErrTestFails']     = 0
        stats['Explicit_RHS']     = 0 
        stats['Implicit_RHS']     = 0   
        stats['runtime']          = 0     # runtime should be 0 is test failed
        stats['Implicit_solves']  = 0 
        stats['avg_dt']           = 0

    ssp_cond = 1
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

        #record the number of times the model has negative values at any time step
        sum_negLines = 0
        for line in stdout_lines:
            txt = line.split()
            if (("Model" in txt) and ("has" in txt) and ("a" in txt) and ("negative" in txt) and ("time" in txt) and ("step" in txt) and ("t" in txt)):
                sum_negLines += 1
        stats['Negative_model'] = sum_negLines 
        stats['avg_dt'] = (10.0 - 0.0) / stats['StepAttempts']         

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

        datafile = "plot_population.py"
        # return with an error if the file does not exist
        if not os.path.isfile(datafile):
            msg = "Error: file " + datafile + " does not exist"
            sys.exit(msg)
        

        # ==== in the plot script, only one diffusion coefficient can be true at a time =====
        # so that you can use the correct reference solution for each diffusion coefficient
        # ===================================================================================
        K = 0.0
        if (kName == "diffk0"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "diffk0 =" in line:
                    val = "True" 
                    modified_lines.append(f"diffk0 = {val}\n")
                elif "diffk02 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk02 = {val}\n")
                elif "diffk04 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk04 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

        # K = 0.02
        elif (kName == "diffk02"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "diffk0 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk0 = {val}\n")
                elif "diffk02 =" in line:
                    val = "True" 
                    modified_lines.append(f"diffk02 = {val}\n")
                elif "diffk04 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk04 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        
        # K = 0.04
        elif (kName == "diffk04"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "diffk0 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk0 = {val}\n")
                elif "diffk02 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk02 = {val}\n")
                elif "diffk04 =" in line:
                    val = "True" 
                    modified_lines.append(f"diffk04 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        # ==== end using correct reference solution for each diffusion coefficient ====

        # # ======================================================================
        # # check the SSP condition (smooth final profile using the smooth_tol
        # # and non-negative values at all time steps for each diffusion coefficient)
        # # ======================================================================
        # # running python file to plot pressure and density 
        sspcommand = " python ./plot_population.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 
        ssp_stdout_lines = ssp_result.stdout.decode('utf-8').splitlines()

        # uncomment the following lines if you want to plot the time evolution for each method, each sigma, each rtol or dt
        # uncomment the corresponding lines in plot_population.py to save the frames as png files
        # rename final solution plot file to include the diffusion coefficient and run type
        # new_name_final_soln = f"finalSoln_population_{kName}_{modetype}_{solver['name']}_{runV}.png"
        # if os.path.exists("populationModel_finalsoln.png"):
        #     os.rename("populationModel_finalsoln.png", new_name_final_soln)
        #     print(f"final solution plot file saved as: {new_name_final_soln}")
        # else:
        #     print("Warning: populationModel_finalsoln.png not found.")

        # new_name_frames= f"population_frames_{kName}_{modetype}_{solver['name']}_{runV}.png"
        # if os.path.exists("populationModel_frames.png"):
        #     os.rename("populationModel_frames.png", new_name_frames)
        #     print(f"population frames plot file saved as: {new_name_frames}")
        # else:
        #     print("Warning: populationModel_frames.png not found.")

        for line in ssp_stdout_lines:
            txt = line.split()
            if (("Lmax" in txt) and ("of" in txt) and ("first" in txt) and ("derivative" in txt) and ("final" in txt)):
                stats['lmax_1dev'] = float(line.split('=')[-1].strip())
            elif (("Lmax" in txt) and ("error" in txt) and ("using" in txt) and ("reference" in txt) and ("solution" in txt)):
                stats['error'] = float(line.split('=')[-1].strip())
            elif (("Reference" in txt) and ("first-derivative" in txt) and ("max" in txt)):
                stats['lmax_1dev_ref'] = float(line.split('=')[-1].strip())

        # ignore errors greater than 10
        if stats['error'] > 10.0:
            stats['ReturnCode'] = 1

        # assessing SSPness based on positivity at all time steps and smooth profile at final time step
        # (smooth: first derivative within smooth_tol of the reference solution's first derivative)
        if kVal in smooth_tol:
            tau_low, tau_high = smooth_tol[kVal]
            D     = stats['lmax_1dev'] #derivative value
            D_ref = stats['lmax_1dev_ref'] #derivative value of the reference solution
            stats['rel_diff_1dev'] = (D - D_ref) / D_ref     # + means D bigger than D_ref, - means D smaller than D_ref
            if (stats['Negative_model'] == 0) and (D_ref * (1 - tau_low) <= D <= D_ref * (1 + tau_high)):
                stats['sspCondition'] = str('ssp')
                ssp_cond = 0
            else:
                stats['sspCondition'] = str('not ssp')
                ssp_cond = 1

            if ssp_cond == 1:
                stats['ReturnCode'] = 1
            #end
        else:
            stats['sspCondition'] = str('ssp')
            ssp_cond = 0
    
    return stats, ssp_cond
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212       = "./population   --dirk_table ARKODE_SSP_SDIRK_2_1_2        --erk_table ARKODE_SSP_ERK_2_1_2  " 
SSP312       = "./population   --dirk_table ARKODE_SSP_DIRK_3_1_2         --erk_table ARKODE_SSP_ERK_3_1_2  "           
SSPL312      = "./population   --dirk_table ARKODE_SSP_LSPUM_SDIRK_3_1_2  --erk_table ARKODE_SSP_LSPUM_ERK_3_1_2  "  
SSP423       = "./population   --dirk_table ARKODE_SSP_ESDIRK_4_2_3       --erk_table ARKODE_SSP_ERK_4_2_3  "  
SSP923       = "./population   --dirk_table ARKODE_SSP_ESDIRK_9_2_3       --erk_table ARKODE_SSP_ERK_9_2_3  "    

# common lists: used both for the Stage 1 search and for the plots (sorted from smallest to largest)
adaptive_params = sorted([1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4,
                          1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0])                  # relative tolerances
fixed_params    = sorted([0.25 * (2 ** i) for i in range(-14, 5)])                   # fixed step sizes (0.25*2^-14 to 0.25*2^4)

## Diffusion coefficients
diff_coef = {'diffk0':0.0, 'diffk02':0.02, 'diffk04':0.04}

solvertype = [{'name': 'SSP212',       'exe': SSP212},
              {'name': 'SSP312',       'exe': SSP312},
              {'name': 'SSPL312',      'exe': SSPL312},
              {'name': 'SSP423',       'exe': SSP423},
              {'name': 'SSP923',       'exe': SSP923}]

## -------------------------------------------------------------------------------------------------------------
# This section uses the Bisection Method to compute the step size or rtol at which the method
# switches from ssp to nonssp. Here, ssp refers to positiviy at all time steps and a smooth final profile.
# The step size or rtol values are rounded to 4 significant figures. 
# The values computed in this section can be appended to the run values (rtol or step sizes)
# to generate the plots.
## -------------------------------------------------------------------------------------------------------------
def round_to_sf(x, sf):
    """
    Converts a number to sf significant figures

    Input:
        x:  number to round
        sf: number of significant figures to run the number to.

    Output: returns the rounded number
    """
    return round(x,-int(floor(log10(abs(x)))) + (sf - 1))


all_runs = []   # stats of every run in Stage 1 and Stage 2

def bisection_midval(solver, runtype):
    """
    Bisection between solver['sspVal'] (ssp-preserving) and solver['nonsspVal'] (not ssp-preserving),
    with each test value rounded to four significant figures.

    Input:
        solver (dict): one bracket from Stage 1 with 'name', 'exe', 'kvalue', 'kname', 'sspVal', 'nonsspVal'
        runtype (str): adaptive or fixed

    Output: returns the largest ssp-preserving value found (to four significant figures)
    """
    sspVal    = solver['sspVal']        # known to pass (from Stage 1)
    nonsspVal = solver['nonsspVal']     # known to fail (from Stage 1)
    preMidVal = None
    iters     = 0

    while True:
        midVal = round_to_sf((sspVal + nonsspVal) / 2.0, 4)

        # stop when rounding no longer gives a new value
        if midVal == preMidVal or midVal == sspVal or midVal == nonsspVal:
            break
        preMidVal = midVal

        stats, condMid = runtest(solver, runtype, midVal, solver['kvalue'], solver['kname'],
                                 showcommand=True, sspcommand=True)
        all_runs.append(stats)
        if condMid == 0:
            sspVal = midVal             # midpoint passes, replace the lower end with the midpoint
        else:
            nonsspVal = midVal          # midpoint fails, replace the upper end with the midpoint
        iters += 1

    print(f"{runtype} {solver['name']} {solver['kname']}: largest ssp = {sspVal}, "
          f"smallest non-ssp = {nonsspVal}, iterations = {iters}")
    return {'name': solver['name'], 'kname': solver['kname'], 'runtype': runtype,
            'largest_ssp': sspVal, 'smallest_nonssp': nonsspVal, 'status': "bisected"}


## -------------------------------------------------------------------------------------------------------------
# For each method and diffusion coefficient, run the search list from the smallest value to the
# largest and stop at the first value that is not ssp-preserving. The bracket for the bisection is
# [last ssp value, first non-ssp value]. All values below the bracket have been checked and are ssp-preserving.
## -------------------------------------------------------------------------------------------------------------
search_fixed    = fixed_params      
search_adaptive = adaptive_params   

def find_brackets(solvers, runtype, searchList, diffCoefs):
    """
    Runs each method from the smallest to the largest value in searchList and stops at the first
    non-ssp value.

    Input:
        solvers (list):    list of solvers with 'name' and 'exe'
        runtype (str):     adaptive or fixed
        searchList (list): fixed step sizes or rtols
        diffCoefs (dict):  diffusion coefficients {kname: kval}

    Output: returns a list of brackets (one per method and diffusion coefficient)
    """
    brackets = []
    for k_name, k_val in diffCoefs.items():
        for solver in solvers:
            sspVal, nonsspVal, status = None, None, ""
            for runV in sorted(searchList):                       # smallest to largest
                stats, cond = runtest(solver, runtype, runV, k_val, k_name, showcommand=True, sspcommand=True)
                all_runs.append(stats)
                if cond == 0:
                    sspVal = runV                                  # still ssp, keep going from the smallest value to the largest
                else:
                    nonsspVal = runV                               # first non-ssp value, stop
                    break

            if sspVal is None:
                status = "smallest value not ssp"
            elif nonsspVal is None:
                status = "ssp at largest value tested"
            else:
                status = "bracket found"

            print(f"{runtype} {solver['name']} {k_name}: sspVal = {sspVal}, nonsspVal = {nonsspVal} ({status})")
            brackets.append({'name': solver['name'], 'exe': solver['exe'], 'kvalue': k_val, 'kname': k_name,
                             'runtype': runtype, 'sspVal': sspVal, 'nonsspVal': nonsspVal, 'status': status})
    return brackets


## ----------------------------------------------------------------------------------------------------
# Stage 1 (brackets fro [ssp, nonssp]) 
# Stage 2 (bisecting the bracket in stage 1). 
# Comment out to skip recomputing the largest ssp-preserving step sizes and rtols. 
# The results are saved in population_brackets.xlsx and population_bisection.xlsx.
## ----------------------------------------------------------------------------------------------------
# Stage 1: find the brackets (smallest to largest, stop at the first non-ssp value)
brackets_fixed    = find_brackets(solvertype, "fixed",    search_fixed,    diff_coef)
brackets_adaptive = find_brackets(solvertype, "adaptive", search_adaptive, diff_coef)
pd.DataFrame(brackets_fixed + brackets_adaptive).to_excel("population_brackets.xlsx", index=False)

# Stage 2: bisection on each bracket found (to read the brackets back from the saved file instead, use
# brackets = pd.read_excel("population_brackets.xlsx").to_dict('records') and change loop to use for b in brackets)
bisection_results = []
for b in brackets_fixed + brackets_adaptive:
    if b['status'] == "bracket found":                       # only bisect where a bracket exists
        bisection_results.append(bisection_midval(b, b['runtype']))
    else:                                                     # nothing to bisect: keep Stage 1 result
        bisection_results.append({'name': b['name'], 'kname': b['kname'], 'runtype': b['runtype'],
                                  'largest_ssp': b['sspVal'], 'smallest_nonssp': b['nonsspVal'],
                                  'status': b['status']})
pd.DataFrame(bisection_results).to_excel("population_bisection.xlsx", index=False)


# save the stats of every run: this includes the runs in Stage 1 and Stage 2, for both fixed and adaptive time stepping
RunStatsAllDf = pd.DataFrame.from_records(all_runs)
RunStatsAllDf.to_excel('population_all_runs.xlsx', index=False)

# keep the bisection value and all values in the params list below it
RunStatsFixed    = []
RunStatsAdaptive = []
for k_name, k_val in diff_coef.items():
    for solver in solvertype:

        # bisection value for this method and diffusion coefficient
        for b in bisection_results:
            if (b['name'] == solver['name']) and (b['kname'] == k_name) and (b['runtype'] == "adaptive"):
                topAdapt = b['largest_ssp']
            if (b['name'] == solver['name']) and (b['kname'] == k_name) and (b['runtype'] == "fixed"):
                topFixed = b['largest_ssp']

        # pick the runs for this method and diffusion coefficient: here we are storing the stats for the bisection
        # ssp-value and all values below it because those values were checked and are ssp-preserving.
        for stat in all_runs:
            if (stat['IMEX_method'] == solver['name']) and (stat['diff_coef'] == k_val):
                if (stat['Runtype'] == "adaptive"):
                    if ((stat['runVal'] in adaptive_params) and (stat['runVal'] < topAdapt)) or (stat['runVal'] == topAdapt):
                        RunStatsAdaptive.append(stat)
                if (stat['Runtype'] == "fixed"):
                    if ((stat['runVal'] in fixed_params) and (stat['runVal'] < topFixed)) or (stat['runVal'] == topFixed):
                        RunStatsFixed.append(stat)

RunStatsFixedDf    = pd.DataFrame.from_records(RunStatsFixed)
RunStatsAdaptiveDf = pd.DataFrame.from_records(RunStatsAdaptive)

# save each dataframe as a separate Excel file
RunStatsFixedDf.to_excel('population_stats_fixed.xlsx', index=False)
RunStatsAdaptiveDf.to_excel('population_stats_adaptive.xlsx', index=False)


# ===============================================================================================================================
#  Generate plots to test the efficiency and accuracy of the IMEX SSP methods
# ===============================================================================================================================
df_fixed    = pd.read_excel('population_stats_fixed' + '.xlsx')    # excel file with the fixed step size runs
df_adaptive = pd.read_excel('population_stats_adaptive' + '.xlsx') # excel file with the adaptive runs

colors   = ['red', 'black', 'blue', 'green', 'orange'] 
diff_coef2 = {'diffk0':0.0, 'diffk02':0.02, 'diffk04':0.04}

x_metrics = [('StepAttempts', 'step-attempts','step_attempts'), 
             ('Implicit_solves', 'implicit-solves','implicit_solves'), 
             ('runtime', 'runtime','runtime')]

y_metrics = [('error', 'error','error')]

for k_name, k_val in diff_coef2.items():
    for x_metric, x_label, x_filename in x_metrics:
        for y_metric, y_label, y_filename in y_metrics:
            fig, ax = plt.subplots( figsize=(7, 6))

            #filter data by fixed and adaptive tests
            data_fixed = df_fixed[(df_fixed["diff_coef"] == k_val)]
            data_adaptive = df_adaptive[(df_adaptive["diff_coef"] == k_val)]

            # fixed run
            for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
                x = valid_data[x_metric]
                y = valid_data[y_metric]
                ax.plot(x, y, color = colors[i], marker='o', markersize=5, linestyle='-', linewidth=2, label=f"{SSPmethodFix}-h")

            # adaptive run
            for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
                x = valid_data[x_metric]
                y = valid_data[y_metric]
                ax.plot(x, y, color = colors[i], marker='*', markersize=5, linestyle='-.', linewidth=2, label=f"{SSPmethodAdapt}-rtol")

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', labelsize=20)

            #remove duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize=20)

            ax.set_xlabel(f'{x_label}', fontsize=20)
            ax.set_ylabel(f'{y_label}', fontsize=20)
            plt.savefig(f"{x_filename}_{y_filename}_population_{k_name}.png", bbox_inches="tight")
            plt.close(fig)
