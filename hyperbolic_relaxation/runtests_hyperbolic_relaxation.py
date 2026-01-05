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
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'nonstiff_param': 0.0, 'stiff_param': 0.0,
             'runVal': runV, 'runtime':0.0, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 
             'Implicit_RHS': 0, 'Implicit_solves': 0, 'err_rho': 0.0, 'energy_err': 0.0}

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
        stats['Implicit_solves'] = 0  
        stats['err_rho']         = 0 
        stats['energy_err']      = 0

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

        # number of implicit solves for each method
        if (solver['name']== 'SSP212'):
            stats['Implicit_solves'] = 2
        elif (solver['name']== 'SSP312'):
            stats['Implicit_solves'] = 3
        elif (solver['name']== 'SSPL312'):
            stats['Implicit_solves'] = 3
        elif (solver['name']== 'SSP423'):
            stats['Implicit_solves'] = 3
        # end

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
                    # print("tstar is %f\n" %tstar)
                elif (("Lmax" in txt) and ("reference" in txt) and ("solution" in txt)):
                    # print("error %.14e" %float(txt[6]))
                    stats['err_rho'] = float(txt[6])
                elif (("Maximum" in txt) and ("energy" in txt) and ("error" in txt)):
                    # print("energy error %.14e" %float(txt[4]))
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
                # # destination_directory = os.getcwd()
                # print('destination_directory:', destination_directory)
                # print("CWD:", os.getcwd())
                # print("Looking for:", file_to_copy)
                # print("Exists?", os.path.exists(file_to_copy))
                shutil.copy(file_to_copy, destination_directory)
                
                # change the working directory to sundials/tools
                curent_directory = os.getcwd()
                # print('curent_directory:', curent_directory)
                tools_directory  = os.chdir("../deps/sundials/tools")
                tools_directory  = os.getcwd()
                # print('tools_directory:', tools_directory)

                # add tstar to time histroy plot
                logcommand = f"./log_example.py {file_to_copy} --tstar %f  --save {save_file}" %(tstar)
                # logcommand = f"./log_example.py {file_to_copy} --tstar %f" %(tstar)
                log_result = subprocess.run(shlex.split(logcommand), stdout=subprocess.PIPE)

                # after the tools directory come back to the bin directory
                bin_directory = os.chdir("../../../bin")
                bin_directory  = os.getcwd()
                # print('bin_directory:', bin_directory)

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
                    # print("error %.14e" %float(txt[6]))
                    stats['err_rho'] = float(txt[6])
                elif (("Maximum" in txt) and ("energy" in txt) and ("error" in txt)):
                    # print("energy error %.14e" %float(txt[4]))
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



# ===============================================================================================================================
#  Generate plots to test the efficiency and accuracy of the IMEX SSP methods
# ===============================================================================================================================
df = pd.read_excel('hyperbolic_relaxation_stats' + '.xlsx') # excel file
stiff_param = {'ks1e6': 1e6, 'ks1e8': 1e8, 'ks1e10': 1e10, 'ks1e12': 1e12}

fixed_accuracy   = True
fixed_efficiency = True
fixed_time       = True

adaptive_rejectSteps = True
adaptive_efficiency  = True
adaptive_time        = True

for stiffNm, stiffVal in stiff_param.items():
    # -------------------------------------------------- fixed runs ----------------------------------------------------------------
    data_fixed = df[(df["stiff_param"] == stiffVal) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "nonstiff_param", "stiff_param", 
                                                                                   "runVal", "runtime", "Steps", "StepAttempts", "ErrTestFails",
                                                                                   "Explicit_RHS", "Implicit_RHS", "Implicit_solves", 
                                                                                   "err_rho", "energy_err"]]
    linestyles = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle(['o', '*', 's', '^'])
    # accuracy plot
    if (fixed_accuracy):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['runVal'], SSPmethodFix_data['err_rho'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('h')
        plt.ylabel('$L_{\\infty}$ error')
        plt.title(f"step size vs error for K = {stiffVal}")
        plt.legend(loc="best")
        plt.savefig(f"accuracy_hyperbolic_{stiffNm}_fixedRun.pdf")
        # plt.show()
    
    # # efficiency plot (number of implicit solves)
    # if (fixed_efficiency):
    #     plt.figure()
    #     for SSPmethodFix in data_fixed['IMEX_method'].unique():
    #         SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
    #         plt.plot(SSPmethodFix_data['err_rho'], SSPmethodFix_data['Implicit_solves'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
    #     plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.xlabel('$L_{\\infty}$ error')
    #     plt.ylabel('number of Implicit Solves')
    #     plt.title(f"error vs implicit solves for K = {stiffVal}")
    #     plt.legend(loc="best")
    #     plt.savefig(f"efficiency_hyperbolic_{stiffNm}_fixedRun.pdf")
    #     # plt.show()

    # efficiency plot (runtime)
    if (fixed_time):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['runtime'], SSPmethodFix_data['err_rho'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.title(f"runtime vs error for K = {stiffVal}")
        plt.legend(loc="best")
        plt.savefig(f"time_hyperbolic_{stiffNm}_fixedRun.pdf")
        # plt.show()


    # -------------------------------------------------- adaptive runs ----------------------------------------------------------------
    data_adaptive = df[(df["stiff_param"] == stiffVal) & (df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "nonstiff_param", "stiff_param", 
                                                                                   "runVal", "runtime", "Steps", "StepAttempts", "ErrTestFails",
                                                                                   "Explicit_RHS", "Implicit_RHS", "Implicit_solves", 
                                                                                   "err_rho", "energy_err"]]
    linestyles = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle(['o', '*', 's', '^'])
    # rejected steps
    if (adaptive_rejectSteps):
        plt.figure()
        for SSPmethodFix in data_adaptive['IMEX_method'].unique():
            SSPmethodFix_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['runVal'], SSPmethodFix_data['ErrTestFails'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        # plt.yscale('log')
        plt.xlabel('rtol')
        plt.ylabel('number of rejected steps')
        plt.title(f"rtol vs rejected steps for K = {stiffVal}")
        plt.legend(loc="best")
        plt.savefig(f"rejectSteps_hyperbolic_{stiffNm}_adaptiveRun.pdf")
        # plt.show()
    
    # # efficiency plot (number of implicit solves)
    # if (adaptive_efficiency):
    #     plt.figure()
    #     for SSPmethodFix in data_adaptive['IMEX_method'].unique():
    #         SSPmethodFix_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodFix]
    #         plt.plot(SSPmethodFix_data['err_rho'], SSPmethodFix_data['Implicit_solves'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
    #     plt.xscale('log')
    #     # plt.yscale('log')
    #     plt.xlabel('$L_{\\infty}$ error')
    #     plt.ylabel('number of Implicit Solves')
    #     plt.title(f"error vs implicit solves for K = {stiffVal}")
    #     plt.legend(loc="best")
    #     plt.savefig(f"efficiency_hyperbolic_{stiffNm}_adaptiveRun.pdf")
    #     # plt.show()

    # efficiency plot (runtime)
    if (adaptive_time):
        plt.figure()
        for SSPmethodFix in data_adaptive['IMEX_method'].unique():
            SSPmethodFix_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['runtime'], SSPmethodFix_data['err_rho'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.title(f"runtime vs error for K = {stiffVal}")
        plt.legend(loc="best")
        plt.savefig(f"time_hyperbolic_{stiffNm}_adaptiveRun.pdf")
        # plt.show()


# =================================================================================================================
# determine the energy error for each method as the stiffness parameter changed, for fixed and adaptive runs
# =================================================================================================================
fixed_energy_err    = True
adaptive_energy_err = True

# ---------- fixed runs -----------
for runNm, runVal in fixed_params.items():
    data_fixed = df[(df["runVal"] == runVal) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "nonstiff_param", "stiff_param", 
                                                                                   "runVal", "runtime", "Steps", "StepAttempts", "ErrTestFails",
                                                                                   "Explicit_RHS", "Implicit_RHS", "Implicit_solves", 
                                                                                   "err_rho", "energy_err"]]
    linestyles = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle(['o', '*', 's', '^'])
    if (fixed_energy_err):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['stiff_param'], SSPmethodFix_data['energy_err'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('stiffness paramater')
        plt.ylabel('energy error')
        plt.title(f"stiffness paramater vs energy error for h = {runVal}")
        plt.legend(loc="best")
        plt.savefig(f"stiffness_hyperbolic_{runNm}_fixedRun.pdf")
        # plt.show()

# ---------- adaptive runs -----------
for runNm, runVal in adaptive_params.items():
    data_adaptive = df[(df["runVal"] == runVal) & (df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "nonstiff_param", "stiff_param", 
                                                                                   "runVal", "runtime", "Steps", "StepAttempts", "ErrTestFails",
                                                                                   "Explicit_RHS", "Implicit_RHS", "Implicit_solves", 
                                                                                   "err_rho", "energy_err"]]
    linestyles = itertools.cycle(['-', '--', ':', '-.'])
    markers = itertools.cycle(['o', '*', 's', '^'])
    if (adaptive_energy_err):
        plt.figure()
        for SSPmethodFix in data_adaptive['IMEX_method'].unique():
            SSPmethodFix_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodFix]
            plt.plot(SSPmethodFix_data['stiff_param'], SSPmethodFix_data['energy_err'], marker = next(markers), markersize=5, linestyle=next(linestyles), label=SSPmethodFix)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('stiffness paramater')
        plt.ylabel('energy error')
        plt.title(f"stiffness paramater vs energy error for rtol = {runVal}")
        plt.legend(loc="best")
        plt.savefig(f"stiffness_hyperbolic_{runNm}_adaptiveRun.pdf")
        # plt.show()
        
print("Accuracy and efficiency plots generated!\n")

    

    








