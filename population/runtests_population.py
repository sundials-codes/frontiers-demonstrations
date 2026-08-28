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
def runtest(solver, modetype, runV, kVal, kName, showcommand=True, sspcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

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
             'lmax_1dev': 0.0, 'error': 0.0, 'runtime':0.0, 'avg_dt':0.0, 'sspCondition': " "}

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
        # stats['maxIntStep']       = 0
        stats['avg_dt']           = 0
        # stats['Negative_model']   = 0 

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
            # elif (("Largest" in txt) and ("average" in txt) and ("step" in txt) and ("size" in txt)):
                # stats['maxIntStep'] = float(txt[7])         #last internal step size used in adaptive run
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
        if (kName == "diffk02"):
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
        # # check the SSP condition (smooth final profile and not negative values
        ##  at all time steps for each diffusion coefficient)
        # # ======================================================================
        # # running python file to plot pressure and density 
        sspcommand = " python ./plot_population.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 
        ssp_stdout_lines = ssp_result.stdout.decode('utf-8').splitlines()

        ## uncomment the following lines if you want to plot the time evolution for each method, each sigma, each rtol or dt
        ## uncomment the corresponding lines in plot_linear_adv_rec.py to save the frames as png files
        ## rename final solution plot file to include the diffusion coefficient and run type
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

            # ignore errors greater than 10  
            if stats['error'] > 10.0:
                stats['ReturnCode'] = 1

            # assessing SSPness based on positivity at all time steps and smooth profile at final time step
            if (kVal==0.0) or (kVal==0.02) or (kVal==0.04):
                if (stats['Negative_model'] == 0):
                    if (kVal==0.0) and (stats['lmax_1dev'] >= 49.28) and (stats['lmax_1dev'] <= 49.45):
                        stats['sspCondition'] = str('ssp')
                        ssp_cond = 0
                    elif (kVal==0.02) and (stats['lmax_1dev'] >= 1.2) and (stats['lmax_1dev'] <= 1.66):
                        stats['sspCondition'] = str('ssp')
                        ssp_cond = 0
                    elif (kVal==0.04) and (stats['lmax_1dev'] >= 0.7) and (stats['lmax_1dev'] <= 1.05):
                        stats['sspCondition'] = str('ssp')
                        ssp_cond = 0
                    else:
                        stats['sspCondition'] = 'not ssp'
                        ssp_cond = 1
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

## These are points for which the method attains non-negativity at all time steps and a smooth solution at the final time
## step for each diffusion coefficient.
adaptive_params = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3] # relative tolerances
fixed_params    = [] # fixed step sizes
for i in range(-14, -2, 1): 
    fixed_params.append(0.25 * (2 ** i))
#end


## -------------------------------------------------------------------------------------------------------------
# This section uses the Bisection Method to compute the step size or rtol at which the method
# switches from ssp to nonssp. Here, ssp refers to positiviy at all time steps and a smooth final profile.
# The step size or rtol values are rounded to 3 significant figures. 
# The values computed in this section can be appended to the run values (rtol or step sizes)
# to generate the plots. The next block of codes (with adaptive and fixed runs) was computed first 
# to determine the interval to bisect before this section was run. You can uncomment the block of codes to 
# generated the step size or relative tolerance values at which the methods switch from ssp (0) to nonssp (1).
# You do not need to run this section unless you want to recompute the bisection values.
# In that case, adaptive params can run from 10^{r}, r = 0, -1, ..., -8, and
#               fixed params can run from 0.25 * 2^{r}, r = 4, -3, ..., -14.
## -------------------------------------------------------------------------------------------------------------
def round_to_sf(x, sf):
    """
    Converts a number to three significant figures

    Input:
        x:  number to round
        sf: number of significant figures to run the number to.

    Output: returns the rounded number
    """
    return round(x,-int(floor(log10(abs(x)))) + (sf - 1))


def bisection_midval(solvers, runtype, paramList):
    """
    Use bisection method to determine values at which methods switch between SSP and Non-SSP.

    Input:
        solvers (list):   List of solvers with 'name', 'exe', 'sspVal', 'nonsspVal', and 'kvalue'.
        runtype (str):    adaptive or fixed.
        paramList (list): rtols or fixed_h

    Output: returns the bisection values
    """
     
    for solver in solvers:
        name = solver['name']
        kval = solver['kvalue']
        kname = solver['kname']

        _,condLow = runtest(solver, runtype, solver['sspVal'], kval, kname, showcommand=True, sspcommand=True)
        _,condHigh = runtest(solver, runtype,solver['nonsspVal'], kval, kname, showcommand=True, sspcommand=True)

        iter = 0
        preMidVal = None
        second_to_preMidVal = None
        while True:
            midVal = (solver['sspVal'] + solver['nonsspVal'])/2.0
            midVal = round_to_sf(midVal,4)

            # end run if midpoint value is the same as previous one and store the last midpoint point value 
            # as well as the previous distinct midpoint value
            if midVal == preMidVal:
                if second_to_preMidVal is not None:
                    paramList.append(second_to_preMidVal)
                paramList.append(midVal)
                print("Mid value did not change after rounding.")
                break
            
            # update midpoint values
            second_to_preMidVal = preMidVal
            preMidVal           = midVal

            # the bisection method
            _,condMid =  runtest(solver, runtype, midVal, kval, kname, showcommand=True, sspcommand=True)
            if (condMid==0):
                solver['sspVal'] = midVal
            elif (condMid==1):
                solver['nonsspVal'] = midVal

            # end run both values have the same ssp condition and store the last midpoint point value 
            #  as well as the previous distinct midpoint value
            _,condLow  = runtest(solver, runtype, solver['sspVal'], kval, kname, showcommand=True, sspcommand=True)
            _,condHigh = runtest(solver, runtype, solver['nonsspVal'], kval, kname, showcommand=True, sspcommand=True)
            if (condLow==condHigh):
                if second_to_preMidVal is not None:
                    paramList.append(second_to_preMidVal)
                paramList.append(midVal)
                print(f"Both values have the same SSP condition ({condLow}).")
                break
            iter += 1

        # print results
        print(f"{runtype} run with {name}, {kval}, iter {iter} : SSP-value & cond = {solver['sspVal'],condLow}, NonSSP-value & cond = {solver['nonsspVal'], condHigh}")

# # #-------------------------------------- adaptive runs -----------------------------------------
# solvernames_adaptK0 = [{'name': 'SSP212',  'exe': SSP212,             'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        {'name': 'SSP312',  'exe': SSP312,             'sspVal': 5e-2, 'nonsspVal': 1e-1, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        {'name':  'SSPL312', 'exe': SSPL312,           'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        {'name': 'SSP423',  'exe': SSP423,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.0, 'kname': 'diffk0'} ]

# solvernames_adaptK2 = [{'name': 'SSP212',  'exe': SSP212,             'sspVal': 1e-3, 'nonsspVal': 2e-3, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP312',  'exe': SSP312,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSPL312', 'exe': SSPL312,            'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP423',  'exe': SSP423,             'sspVal': 5e-3, 'nonsspVal': 1e-2, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP923',  'exe': SSP923,             'sspVal': 5e-4, 'nonsspVal': 1e-3, 'kvalue': 0.02, 'kname': 'diffk02'} ]

# solvernames_adaptK4 = [{'name': 'SSP212', 'exe': SSP212,              'sspVal': 5e-4, 'nonsspVal': 1e-3, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP312', 'exe': SSP312,              'sspVal': 5e-1, 'nonsspVal': 1.0, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSPL312','exe': SSPL312,             'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP423', 'exe': SSP423,              'sspVal': 5e-3, 'nonsspVal': 1e-2, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP923', 'exe': SSP923,              'sspVal': 5e-4, 'nonsspVal': 1e-3, 'kvalue': 0.04, 'kname': 'diffk04'}  ]

# bisection_midval(solvernames_adaptK0, "adaptive", paramList = adaptive_params)
# bisection_midval(solvernames_adaptK2, "adaptive", paramList = adaptive_params)
# bisection_midval(solvernames_adaptK4, "adaptive", paramList = adaptive_params)


# # #-------------------------------------- fixed runs -----------------------------------------
# solvernames_fixedK0 = [{'name': 'SSP212',  'exe': SSP212,            'sspVal': 0.5, 'nonsspVal': 1.0, 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        {'name': 'SSP312',  'exe': SSP312,            'sspVal': 1.0, 'nonsspVal': 1.3, 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 1.0, 'nonsspVal': 2.0, 'kvalue': 0.0, 'kname': 'diffk0'} ]

# solvernames_fixedK2 = [{'name': 'SSP212',  'exe': SSP212,            'sspVal': 0.015625,'nonsspVal': 0.03125,'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 1.0,     'nonsspVal': 2.0,    'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP423',  'exe': SSP423,            'sspVal': 0.5,    'nonsspVal': 0.65,    'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP923',  'exe': SSP923,            'sspVal': 0.03125, 'nonsspVal': 0.0625, 'kvalue': 0.02, 'kname': 'diffk02'} ]

# solvernames_fixedK4 = [{'name': 'SSP212',  'exe': SSP212,            'sspVal': 0.015625, 'nonsspVal': 0.03125, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 1.0,      'nonsspVal': 2.0,     'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP423',  'exe': SSP423,            'sspVal': 0.5,      'nonsspVal': 1.0,     'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP923',  'exe': SSP923,            'sspVal': 0.03125,  'nonsspVal': 0.0625,  'kvalue': 0.04, 'kname': 'diffk04'} ]

# bisection_midval(solvernames_fixedK0, "fixed", paramList = fixed_params)
# bisection_midval(solvernames_fixedK2, "fixed", paramList = fixed_params)
# bisection_midval(solvernames_fixedK4, "fixed", paramList = fixed_params)


## ----------------------------------------------------------------------------------------------------
# This section generates the data for each method, diffusion coefficient with different fixed step
# sizes and rtols
## ----------------------------------------------------------------------------------------------------
## Diffusion coefficients
diff_coef = {'diffk0':0.0, 'diffk02':0.02, 'diffk04':0.04}

solvertype = [{'name': 'SSP212',       'exe': SSP212},
              {'name': 'SSP312',       'exe': SSP312},
              {'name': 'SSPL312',      'exe': SSPL312},
              {'name': 'SSP423',       'exe': SSP423},
              {'name': 'SSP923',       'exe': SSP923}]

# run tests and collect results as a pandas data frame
fname = 'population_stats' 
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_val in adaptive_params:
        for solver_adapt in solvertype:
            adaptive_stat, _ = runtest(solver_adapt, "adaptive", runV_val, k_val, k_name, showcommand=True, sspcommand=True)
            RunStats.append(adaptive_stat)

    for runV_val in fixed_params:
        for solver_adapt in solvertype:
            fixed_stat, _ = runtest(solver_adapt, "fixed", runV_val, k_val, k_name, showcommand=True, sspcommand=True)
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
df = pd.read_excel('population_stats' + '.xlsx') # excel file
methods = df['IMEX_method'].unique()

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
            col_data = df[(df["diff_coef"] == k_val)]
            data_fixed = col_data[col_data["Runtype"] == "fixed"]
            data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

            # fixed run
            for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
                valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
                x = valid_data[x_metric]
                y = valid_data[y_metric]
                ax.plot(x, y, color = colors[i], marker='o', markersize=5, linestyle='-', label=f"{SSPmethodFix}-h")

            # adaptive run
            for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
                valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
                x = valid_data[x_metric]
                y = valid_data[y_metric]
                ax.plot(x, y, color = colors[i], marker='*', markersize=5, linestyle='-.', label=f"{SSPmethodAdapt}-rtol")

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(axis='both', labelsize=13)

            #remove duplicates
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize=10)

            fig.supxlabel(f'{x_label}', fontsize=11)
            fig.supylabel(f'{y_label}', fontsize=11)
            plt.savefig(f"{x_filename}_{y_filename}_population_{k_name}.png", bbox_inches="tight")
            plt.close(fig)

