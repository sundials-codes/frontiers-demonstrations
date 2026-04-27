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

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 
             'runVal': runV, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 
             'Implicit_RHS': 0, 'Implicit_solves':0, 'maxIntStep': 0.0,  'Negative_model': 0, 
             'lmax_1dev': 0.0, 'error': 0.0, 'runtime':0.0, 'sspCondition': " "}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %e --k %.2f" % (solver['exe'], runV, kVal)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f --k %.2f" % (solver['exe'], runV, kVal)
    
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
        if ("test failed repeatedly" in line):
            sundials_failed = True

    if sundials_failed == True:
        if (modetype == "adaptive"):
            print("SUNDIALS failed for  %s  --rtol %e --k %.2f" % (solver['exe'], runV, kVal))
        elif (modetype == "fixed"):
            print("SUNDIALS failed for %s  --fixed_h %.6f --k %.2f" % (solver['exe'], runV, kVal))
        stats['ReturnCode']       = 1
        stats['error']            = 0
        stats['Steps']            = 0
        stats['StepAttempts']     = 0
        stats['ErrTestFails']     = 0
        stats['Explicit_RHS']     = 0 
        stats['Implicit_RHS']     = 0   
        stats['runtime']          = 0     # runtime should be 0 is test failed
        stats['Implicit_solves']  = 0 
        stats['maxIntStep']       = 0
        # stats['Negative_model']   = 0 

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
            elif (("Largest" in txt) and ("average" in txt) and ("step" in txt) and ("size" in txt)):
                stats['maxIntStep'] = float(txt[7])         #last internal step size used in adaptive run
        sum_negLines = 0
        for line in stdout_lines:
            txt = line.split()
            if (("Model" in txt) and ("has" in txt) and ("a" in txt) and ("negative" in txt) and ("time" in txt) and ("step" in txt) and ("t" in txt)):
                sum_negLines += 1
        stats['Negative_model'] = sum_negLines            

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
        

        # ====== in the plot script, only one diffusion coefficient can be true at a time =========
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
                elif "diffk04 =" in line:
                    val = "True" 
                    modified_lines.append(f"diffk04 = {val}\n")
                elif "diffk06 =" in line:
                    val = "False" 
                    modified_lines.append(f"diffk06 = {val}\n")
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
        # # select the run type you want to use
        # # ======================================================================
        if (modetype=="adaptive"):
            # adaptiveRun = True and fixedRun = False to compute the Lmax error
            with open(datafile, "r") as file:
                original_lines = file.readlines()

            modified_lines = []
            for line in original_lines:
                if "AdaptiveRun =" in line:
                    val = "True" #if modetype == "adaptive" else "False"
                    modified_lines.append(f"AdaptiveRun = {val}\n")
                elif "FixedRun =" in line:
                    val = "False" #if modetype == "adaptive" else "True"
                    modified_lines.append(f"FixedRun = {val}\n")
                else:
                    modified_lines.append(line)
                # end
                
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
                
            # running python file to plot pressure and density 
            sspcommand = " python ./plot_population.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 
            ssp_stdout_lines = ssp_result.stdout.decode('utf-8').splitlines()
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
                if (kVal==0.02) and (stats['lmax_1dev'] >= 1.2) and (stats['lmax_1dev'] <= 1.7) and (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                elif (kVal==0.04) and (stats['lmax_1dev'] >= 0.7) and (stats['lmax_1dev'] <= 1.5) and (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                else:
                    stats['sspCondition'] = str('not ssp')  
                    ssp_cond = 1   

                # assessing SSPness based on positivity at all time steps
                if (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                else:
                    stats['sspCondition'] = str('not ssp')  
                    ssp_cond = 1 
                #end
            # #end

        elif (modetype == "fixed"):
            # FixedRun = True and AdaptiveRun = False to compute the Lmax error
            with open(datafile, "r") as file:
                original_lines = file.readlines()

            modified_lines = []
            for line in original_lines:
                if "FixedRun =" in line:
                    val = "True" #if modetype == "fixed" else "False"
                    modified_lines.append(f"FixedRun = {val}\n")
                elif "AdaptiveRun =" in line:
                    val = "False" #if modetype == "fixed" else "True"
                    modified_lines.append(f"AdaptiveRun = {val}\n")
                else:
                    modified_lines.append(line)
                
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

            ## running python file to plot pressure and density
            sspcommand = " python ./plot_population.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)  
            ssp_stdout_lines = ssp_result.stdout.decode('utf-8').splitlines()
            for line in ssp_stdout_lines:
                txt = line.split()
                if (("Lmax" in txt) and ("of" in txt) and ("first" in txt) and ("derivative" in txt) and ("final" in txt)):
                    stats['lmax_1dev'] = float(line.split('=')[-1].strip())
                elif (("Lmax" in txt) and ("error" in txt) and ("using" in txt) and ("reference" in txt) and ("solution" in txt)):
                    stats['error'] = float(line.split('=')[-1].strip())

               # ignore errors greater than 10  
                if stats['error'] > 10.0:
                    stats['ReturnCode'] = 1
                #end

                # assessing SSPness based on positivity at all time steps and smooth profile at final time step
                if (kVal==0.02) and (stats['lmax_1dev'] >= 1.2) and (stats['lmax_1dev'] <= 1.7) and (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                elif (kVal==0.04) and (stats['lmax_1dev'] >= 0.7) and (stats['lmax_1dev'] <= 1.5) and (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                else:
                    stats['sspCondition'] = str('not ssp')  
                    ssp_cond = 1   

                # assessing SSPness based on positivity at all time steps
                if (stats['Negative_model'] == 0):
                    stats['sspCondition'] = str('ssp')
                    ssp_cond = 0
                else:
                    stats['sspCondition'] = str('not ssp')  
                    ssp_cond = 1 
            #end
        # # end : end of run type selection
        
    return stats, ssp_cond
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212       = "./population   --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP312       = "./population   --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSPL312      = "./population   --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP423       = "./population   --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"  
SSP923       = "./population   --IMintegrator ARKODE_SSP_ESDIRK_9_2_3       --EXintegrator ARKODE_SSP_ERK_9_2_3"    

adaptive_params = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0] ## relative tolerances
# adaptive_params = [1.0, 1e-1, 1e-2] ## relative tolerances
fixed_params    = [] # fixed time step sizes
for i in range(4, -10, -1): 
    fixed_params.append(0.25 * (2 ** i))
#end


## ----------------------------------------------------------------------------------------------------
# This section uses the Bisection Method to compute the step size or rtol at which the method
# switches from ssp to nonssp. The step size or rtol values are rounded to 3 significant figures.
# The values computed in this section are then appended to the run values (rtol or step sizes)
# to generate the plots. The next section was computed first to determine the interval to bisect 
# before this section was run. You would not need to that.
## ----------------------------------------------------------------------------------------------------
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
            # as well as the previous distinct midpoint value
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

# # -------------------------------------- adaptive runs -----------------------------------------
# solvernames_adaptK0 = [#{'name': 'SSP212',  'exe': SSP212,              'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        #{'name': 'SSP312',  'exe': SSP312,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        {'name':  'SSPL312', 'exe': SSPL312,            'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.0, 'kname': 'diffk0'},
#                        #{'name': 'SSP423',  'exe': SSP423,             'sspVal': 1e-3, 'nonsspVal': 1e-2, 'kvalue': 0.0, 'kname': 'diffk0'}, 
#                        #{'name': 'SSP923',  'exe': SSP923,              'sspVal': 1e-3, 'nonsspVal': 1e-2, 'kvalue': 0.0, 'kname': 'diffk0'} 
#                        ]

# solvernames_adaptK2 = [{'name': 'SSP212',  'exe': SSP212,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP312',  'exe': SSP312,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSPL312', 'exe': SSPL312,            'sspVal': 5e-2, 'nonsspVal': 1e-1, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP423',  'exe': SSP423,             'sspVal': 1e-3, 'nonsspVal': 1e-2, 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP923',  'exe': SSP923,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02, 'kname': 'diffk02'} ]

# solvernames_adaptK4 = [{'name': 'SSP212', 'exe': SSP212,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04, 'kname': 'diffk04'},
#                     #    {'name': 'SSP312', 'exe': SSP312,           'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 5e-2, 'nonsspVal': 1e-1, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP423', 'exe': SSP423,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP923', 'exe': SSP923,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04, 'kname': 'diffk04'}  ]

# bisection_midval(solvernames_adaptK0, "adaptive", paramList = adaptive_params)
# bisection_midval(solvernames_adaptK2, "adaptive", paramList = adaptive_params)
# bisection_midval(solvernames_adaptK4, "adaptive", paramList = adaptive_params)


# # -------------------------------------- fixed runs -----------------------------------------
# solvernames_fixedK0 = [{'name': 'SSP212',  'exe': SSP212,            'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        {'name': 'SSP312',  'exe': SSP312,            'sspVal': 0.25*(2**3), 'nonsspVal': 0.25*(2**4), 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        {'name': 'SSP423',  'exe': SSP423,            'sspVal': 0.25*(2**3), 'nonsspVal': 0.25*(2**4), 'kvalue': 0.0, 'kname': 'diffk0'} ,
#                        #{'name': 'SSP923',  'exe': SSP923,            'sspVal': 0.25*(2**3), 'nonsspVal': 0.25*(2**4), 'kvalue': 0.0, 'kname': 'diffk0'} 
#                        ]

# solvernames_fixedK2 = [{'name': 'SSP212',  'exe': SSP212,             'sspVal': 0.25*(2**1), 'nonsspVal': 0.25*(2**2), 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP312',  'exe': SSP312,             'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSPL312', 'exe': SSPL312,           'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP423',  'exe': SSP423,             'sspVal': 0.25*(2**0), 'nonsspVal': 0.25*(2**1), 'kvalue': 0.02, 'kname': 'diffk02'},
#                        {'name': 'SSP923',  'exe': SSP923,             'sspVal': 0.25*(2**-7), 'nonsspVal': 0.25*(2**-6), 'kvalue': 0.02, 'kname': 'diffk02'} ]

# solvernames_fixedK4 = [{'name': 'SSP212',  'exe': SSP212,             'sspVal': 0.25*(2**1), 'nonsspVal': 0.25*(2**2), 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP312',  'exe': SSP312,             'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSPL312', 'exe': SSPL312,            'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP423',  'exe': SSP423,             'sspVal': 0.25*(2**0), 'nonsspVal': 0.25*(2**1), 'kvalue': 0.04, 'kname': 'diffk04'},
#                        {'name': 'SSP923',  'exe': SSP923,             'sspVal': 0.25*(2**-8), 'nonsspVal': 0.25*(2**-7), 'kvalue': 0.04, 'kname': 'diffk04'} ]

# bisection_midval(solvernames_fixedK0, "fixed", paramList = fixed_params)
# bisection_midval(solvernames_fixedK2, "fixed", paramList = fixed_params)
# bisection_midval(solvernames_fixedK4, "fixed", paramList = fixed_params)


## ----------------------------------------------------------------------------------------------------
# This section generates the data for each method, diffusion coefficient with different fixed step
# sizes and rtols
## ----------------------------------------------------------------------------------------------------
sorted_adaptive_params = sorted(adaptive_params) ## relative tolerances
sorted_fixed_params    = sorted(fixed_params) ## fixed time step sizes

## Diffusion coefficients
# diff_coef = {'diffk0':0.0, 'diffk02':0.02, 'diffk04':0.04}
diff_coef = { 'diffk04':0.04}

## Integrator types
solvertype = [{'name': 'SSP212',       'exe': SSP212},
              {'name': 'SSP312',       'exe': SSP312},
              {'name': 'SSPL312',      'exe': SSPL312},
              {'name': 'SSP423',       'exe': SSP423},
              {'name': 'SSP923',       'exe': SSP923}]

# run tests and collect results as a pandas data frame
fname = 'population_stats' 
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_val in sorted_adaptive_params:
        for solver_adapt in solvertype:
            adaptive_stat, _ = runtest(solver_adapt, "adaptive", runV_val, k_val, k_name, showcommand=True, sspcommand=True)
            RunStats.append(adaptive_stat)

    for runV_val in sorted_fixed_params:
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

# colors   = ['red', 'black', 'blue', 'green', 'orange'] 
# markers  = ['o', '*', 's', '^', '+']
# modetype = ['fixed', 'adaptive']

# --------------------------- accepted steps vs erroru ----------------------------------
#create a figure of subplots (columns are diffusion coefficients parameters and rows are methods)
fig, axes = plt.subplots(1, len(diff_coef), figsize=(15, 15))
if len(diff_coef) == 1:
    axes = [axes]
for col_ind, (kValName, kVal) in enumerate(diff_coef.items()):

    #filter data by fixed and adaptive tests
    col_data = df[(df["diff_coef"] == kVal)]
    data_fixed = col_data[col_data["Runtype"] == "fixed"]
    data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

    # fixed run
    for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
        SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
        valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
        x = valid_data['StepAttempts']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='-', label=f"{SSPmethodFix}-h")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['StepAttempts'], sspness['error'], marker='x', linewidth=2, linestyle='none',color=method_line_color, label='_nolegend_')

    #adaptive run
    for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
        SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
        valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
        x = valid_data['StepAttempts']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='--', label=f"{SSPmethodAdapt}-rtol")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['StepAttempts'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color, label='_nolegend_')
        

    # each column should correspond to a stiffness parameter
    axes[col_ind].set_title(f"d = {kVal}", fontsize=18)
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].legend(loc="best", ncol=2, fontsize=18)
    axes[col_ind].tick_params(axis='both', labelsize=15)
#end
fig.supxlabel(' StepAttempts ', fontsize=18)
fig.supylabel(' error ', fontsize=18)
# fig.suptitle("StepAttempts vs error", fontsize=20)
fig.tight_layout()
plt.savefig("StepAttempts_error_population.png")


# --------------------------- implicit solves vs erroru ----------------------------------
#create a figure of subplots (columns are diffusion coefficients parameters and rows are methods)
fig, axes = plt.subplots(1, len(diff_coef), figsize=(15, 15))
if len(diff_coef) == 1:
    axes = [axes]
for col_ind, (kValName, kVal) in enumerate(diff_coef.items()):

    #filter data by fixed and adaptive tests
    col_data = df[(df["diff_coef"] == kVal)]
    data_fixed = col_data[col_data["Runtype"] == "fixed"]
    data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

    # fixed run
    for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
        SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
        valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
        x = valid_data['Implicit_solves']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='-', label=f"{SSPmethodFix}-h")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['Implicit_solves'], sspness['error'], marker='x', linewidth=2, linestyle='none',color=method_line_color, label='_nolegend_')

    #adaptive run
    for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
        SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
        valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
        x = valid_data['Implicit_solves']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='--', label=f"{SSPmethodAdapt}-rtol")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['Implicit_solves'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color, label='_nolegend_')
        

    # each column should correspond to a stiffness parameter
    axes[col_ind].set_title(f"d = {kVal}", fontsize=18)
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].legend(loc="best", ncol=2,fontsize=18)
    axes[col_ind].tick_params(axis='both', labelsize=15)
#end
fig.supxlabel(' Implicit_solves ', fontsize=18)
fig.supylabel(' error ', fontsize=18)
# fig.suptitle("Implicit_solves vs error", fontsize=20)
fig.tight_layout()
plt.savefig("Implicit_solves_error_population.png")


# --------------------------- runtime vs erroru ----------------------------------
#create a figure of subplots (columns are diffusion coefficients parameters and rows are methods)
fig, axes = plt.subplots(1, len(diff_coef), figsize=(15, 15))
if len(diff_coef) == 1:
    axes = [axes]
for col_ind, (kValName, kVal) in enumerate(diff_coef.items()):

    #filter data by fixed and adaptive tests
    col_data = df[(df["diff_coef"] == kVal)]
    data_fixed = col_data[col_data["Runtype"] == "fixed"]
    data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

    # fixed run
    for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
        SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
        valid_data = SSPmethodFix_data[SSPmethodFix_data['ReturnCode'] != 1]
        x = valid_data['runtime']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='-', label=f"{SSPmethodFix}-h")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none',color=method_line_color, label='_nolegend_')

    #adaptive run
    for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
        SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
        valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
        x = valid_data['runtime']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='--', label=f"{SSPmethodAdapt}-rtol")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color, label='_nolegend_')
        

    # each column should correspond to a stiffness parameter
    axes[col_ind].set_title(f"d = {kVal}", fontsize=18)
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].legend(loc="best", ncol=2,fontsize=18)
    axes[col_ind].tick_params(axis='both', labelsize=15)
#end
fig.supxlabel(' runtime ', fontsize=18)
fig.supylabel(' error ', fontsize=18)
# fig.suptitle("runtime vs error", fontsize=20)
fig.tight_layout()
plt.savefig("runtime_error_population.png")



# --------------------------- runtime vs erroru ----------------------------------
#create a figure of subplots (columns are diffusion coefficients parameters and rows are methods)
fig, axes = plt.subplots(1, len(diff_coef), figsize=(15, 15))
if len(diff_coef) == 1:
    axes = [axes]
for col_ind, (kValName, kVal) in enumerate(diff_coef.items()):

    #filter data by fixed and adaptive tests
    col_data = df[(df["diff_coef"] == kVal)]
    data_adaptive = col_data[col_data["Runtype"] == "adaptive"]

    #adaptive run
    for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
        SSPmethodAdapt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt]
        valid_data = SSPmethodAdapt_data[SSPmethodAdapt_data['ReturnCode'] != 1]
        x = valid_data['runtime']
        y = valid_data['error']
        method_line = axes[col_ind].plot(x, y, marker='.', linestyle='--', label=f"{SSPmethodAdapt}-rtol")
        method_line_color = method_line[0].get_color()
        sspness = valid_data[valid_data['sspCondition'] == "not ssp"]
        axes[col_ind].plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color, label='_nolegend_')
        
    # each column should correspond to a stiffness parameter
    axes[col_ind].set_title(f"d = {kVal}", fontsize=18)
    axes[col_ind].set_xscale('log')
    axes[col_ind].set_yscale('log')
    axes[col_ind].legend(loc="best", ncol=2, fontsize=18)
    axes[col_ind].tick_params(axis='both', labelsize=15)
#end
fig.supxlabel(' runtime ', fontsize=18)
fig.supylabel(' error ', fontsize=18)
# fig.suptitle("runtime vs error", fontsize=20)
fig.tight_layout()
plt.savefig("runtime_error_population_adaptive.png")

