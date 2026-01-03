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
def runtest(solver, modetype, runV, runN, kstiff, knonstiff, kstiffname, showcommand=True, sspcommand=True):
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
             'stiff_param': 0.0, 'nonstiff_param': 0.0, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 
             'Explicit_RHS': 0, 'Implicit_RHS': 0, 'err_rho': 0.0, 'energy_err': 0.0}

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
        stats['err_rho']         = 0 

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

        # ====== in the plot script, only one stiffness value can be true at a time =========
        # so that you can use the correct reference solution for each stiffness parameter
        # ===================================================================================
        # K = 1e6
        if (kstiffname == "ks1e6"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "stiff1e6 =" in line:
                    val = "True" 
                    modified_lines.append(f"stiff1e6 = {val}\n")
                elif "stiff1e8 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e8 = {val}\n")
                elif "stiff1e10 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e10 = {val}\n")
                elif "stiff1e12 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e12 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

        # K = 1e8
        elif (kstiffname == "ks1e8"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "stiff1e6 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e6 = {val}\n")
                elif "stiff1e8 =" in line:
                    val = "True" 
                    modified_lines.append(f"stiff1e8 = {val}\n")
                elif "stiff1e10 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e10 = {val}\n")
                elif "stiff1e12 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e12 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        
        # K = 1e10
        elif (kstiffname == "ks1e10"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "stiff1e6 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e6 = {val}\n")
                elif "stiff1e8 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e8 = {val}\n")
                elif "stiff1e10 =" in line:
                    val = "True" 
                    modified_lines.append(f"stiff1e10 = {val}\n")
                elif "stiff1e12 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e12 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)

        # K = 1e12
        elif (kstiffname == "ks1e12"):
            with open(datafile, "r") as file:
                original_lines = file.readlines()
            modified_lines = []
            for line in original_lines:
                if "stiff1e6 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e6 = {val}\n")
                elif "stiff1e8 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e8 = {val}\n")
                elif "stiff1e10 =" in line:
                    val = "False" 
                    modified_lines.append(f"stiff1e10 = {val}\n")
                elif "stiff1e12 =" in line:
                    val = "True" 
                    modified_lines.append(f"stiff1e12 = {val}\n")
                else:
                    modified_lines.append(line)
            # write the modified line to the python script
            with open(datafile, "w") as f:
                f.writelines(modified_lines)
        # # ==== end using correct reference solution for each stiffness value ====


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
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE) 
            new_fileName = f"hyperbolic_graph_{solver['name']}_{runN}_{kstiffname}.png"
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
                    print("tstar is %f\n" %tstar)
                elif (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt)):
                    print("error %.14e" %float(txt[6]))
                    stats['err_rho'] = float(txt[6])
                elif (("Maximum" in txt) and ("energy" in txt) and ("error" in txt)):
                    print("energy error %.14e" %float(txt[4]))
                    stats['energy_err'] = float(txt[4])
                #end
            # #end

            #tstar should not be None for adaptive runs.
            if tstar is not None:
                ## ==============================================================================
                ## use the t_star to determine time history on the left and right side of the shock
                ## ==============================================================================
                # copy sun.log file into the /sundials/tools folder
                file_to_copy = "sun-%s-%s.log" % (solver['name'], runN) #'./sun.log'
                save_file = "sun-%s-%s-%s" % (solver['name'], runN, kstiffname)
                destination_directory = './../deps/sundials/tools'
                shutil.copy(file_to_copy, destination_directory)

                # change the working directory to sundials/tools
                curent_directory = os.getcwd()
                tools_directory  = os.chdir("../deps/sundials/tools")
                tools_directory  = os.getcwd()

                # add tstar to time histroy plot
                # logcommand = f"./log_example.py {file_to_copy} --tstar %f  --save {save_file}" %(tstar)
                logcommand = f"./log_example.py {file_to_copy} --tstar %f" %(tstar)
                log_result = subprocess.run(shlex.split(logcommand), stdout=subprocess.PIPE)

                # after the tools directory come back to the bin directory
                bin_directory = os.chdir("../../../bin")
                bin_directory  = os.getcwd()

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
            sspcommand = " python ./plot_hyperbolic_relaxation.py"
            ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)   
            new_fileName = f"hyperbolic_graph_{solver['name']}_{runN}_{kstiffname}.png"
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
                if (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt)):
                    print("error %.14e" %float(txt[6]))
                    stats['err_rho'] = float(txt[6])
                elif (("Maximum" in txt) and ("energy" in txt) and ("error" in txt)):
                    print("energy error %.14e" %float(txt[4]))
                    stats['energy_err'] = float(txt[4])
                #end
            #end
        # # end : end of run type selection
        
    return stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2        --output 2" 
SSP312  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2        --output 2"           
SSPL312 = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2  --output 2"  
SSP423  = "  ./hyperbolic_relaxation  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3        --output 2"    

## common testing parameters
adaptive_params = {'r1': 1e-5, 'r2':1e-4, 'r3':1e-3, 'r4':1e-2, 'r5':1e-1, 'r6':1.0} #relative tolerances
fixed_params    = {} #fixed time step sizes
for i in range(5, -1, -1):
    fixed_params[f"h{i}"] = 0.01/(2.0**i)

## stiffness parameters
nonstiff_params = [1e2]
stiff_params    = {'ks1e6': 1e6, 'ks1e8': 1e8, 'ks1e10': 1e10, 'ks1e12': 1e12}

## Integrator types
solvertype = [{'name': 'SSP212',  'exe': SSP212},
              {'name': 'SSP312',  'exe': SSP312},
              {'name': 'SSPL312', 'exe': SSPL312},
              {'name': 'SSP423',  'exe': SSP423}]

# run tests and collect results as a pandas data frame
fname = 'hyperbolic_relaxation_stats' 
RunStats = []

for knonstiff in nonstiff_params:
    for kstiffname, kstiff in stiff_params.items():
        for runname, runvalue in adaptive_params.items():
            for solver_adapt in solvertype:
                adaptive_stat = runtest(solver_adapt, "adaptive", runvalue, runname, kstiff, knonstiff, kstiffname, showcommand=True, sspcommand=True)
                RunStats.append(adaptive_stat)

for knonstiff in nonstiff_params:
    for kstiffname, kstiff in stiff_params.items():
        for runname, runvalue in fixed_params.items():
            for solver_fixed in solvertype:
                fixed_stat = runtest(solver_fixed, "fixed", runvalue, runname, kstiff, knonstiff, kstiffname, showcommand=True, sspcommand=True)
                RunStats.append(fixed_stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)




