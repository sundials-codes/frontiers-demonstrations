#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
# --------------------------------------------------------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2024, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
#----------------------------------------------------------------------------------------------------------------------------------
# ReadME: This script extracts the rtol and fixed_h for which a method is SSP and stores then a list.
#         The data is then used to plot the numerical solution at the final time step for each method.
#-----------------------------------------------------------------------------------------------------------------------------------

# imports
import sys, os
import subprocess
import time
import shlex
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

## ------------------------ Locate excel file --------------------
if os.path.exists("population_density_imex_stats.xlsx"):
    print("File found, extracting data from file.")
else:
    sys.exit("Warning: population_density_imex_stats.xlsx not found.")

## ---------------- Extract data from excel file -----------------
fixed_ssp_params    = []
adaptive_ssp_params = []
kVal                = {"k2": 0.02,"k4": 0.04} #diffusion coefficient
runtype             = ["adaptive", "fixed"]
IMEXmethods         = ["SSP-ARK-2-1-2", "SSP-ARK-3-1-2", "SSP-LSPUM-ARK-3-1-2", "SSP-ARK-4-2-3"]
df = pd.read_excel('population_density_imex_stats' + '.xlsx')

# ---------------------- adaptive runs -----------------------------
data_adaptive = df[(df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "diff_coef","runVal", "Negative_model", "sspCondition"]]
#add only the runval values corresponding to an ssp condition to the list
adaptive_ssp_params.extend(data_adaptive.loc[data_adaptive["Negative_model"] == 0, "runVal"].tolist()) 
#remove duplicated values
adaptive_ssp_params = list(dict.fromkeys(adaptive_ssp_params))


# ---------------------- fixed runs --------------------------------
data_fixed = df[(df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "diff_coef",
                                             "runVal", "Negative_model", "sspCondition"]]
#add only the runval values corresponding to an ssp condition to the list
fixed_ssp_params.extend(data_fixed.loc[data_fixed["Negative_model"] == 0, "runVal"].tolist()) 
#remove duplicated values
fixed_ssp_params = list(dict.fromkeys(fixed_ssp_params))


## ---------------------------- function to run commands and plot ----------------------------
def runtest(solver, runtype, runV, kVal, showcommand=True, sspcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    and plots the solution at the final time step.
  

    Input: solver:            imex scheme to tun
           runtype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient

    Output: returns the plot
    """

    if (runtype == "adaptive"):
        runcommand = " %s  --rtol %.6f --k %.2f" % (solver['exe'], runV, kVal)
    elif (runtype == "fixed"):
        runcommand = " %s  --fixed_h %.6f --k %.2f" % (solver['exe'], runV, kVal)
    
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Running: " + runcommand + " SUCCESS")

            # executabel spits out a text file containing the solution at all time steps
            datafile  = "population.txt"
            if not os.path.isfile(datafile):
                msg = "Error: file " + datafile + " does not exist"
                sys.exit(msg)
            
            #  read solution file, storing each line as a string in a list
            with open(datafile, "r") as file:
                lines = file.readlines()

                # extract header information
                title  = lines.pop(0)  # Title
                T0     = float(lines.pop(0).split()[2])   # initial time
                Tf     = float(lines.pop(0).split()[2])   # final time
                N      = int(lines.pop(0).split()[2])     # spatial dimension
                xl     = float(lines.pop(0).split()[2])   # left endpoint on spatial grid
                xr     = float(lines.pop(0).split()[2])   # right endpoint on spatial grid 
                diff_k = float(lines.pop(0).split()[2])   # diffusion coefficient 
                x      = np.linspace(xl, xr, N)

                lastline  = (lines[-1])
                num_steps = lastline.split(':')
                nsteps    = int(num_steps[1].strip()) # total number of steps taken

                dt = (Tf-T0)/nsteps                   # temporal step size
                dx = (xr - xl)/N                      # spatial step size
                
                # allocate solution data as 2D Python arrays
                t    = np.zeros((nsteps), dtype=float)
                pSol = np.zeros((nsteps, N), dtype=float)

                # store remaining data into numpy arrays
                it  = 0
                for i in range(0, len(lines)):
                    if "Time step" in lines[i]:
                        get_t  = lines[i].split(':')
                        time_t = get_t[1].strip()
                        i = i + 1
                        pSol[it,:] = np.array(list(map(float, lines[i].split()))) #to remove single quotes around the vectors since each vector is a line
                        t[it] = time_t #(it + 1) * dt
                        it = it + 1
        
    # return the solution at the final tiem step
    return x, pSol[-1, :]
## end of function


SSP_ARK_212       = "./population_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./population_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./population_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./population_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"   

## Integrator types
solvertype = [{'name': 'SSP-ARK-2-1-2',       'exe': SSP_ARK_212},
              {'name': 'SSP-ARK-3-1-2',       'exe': SSP_ARK_312},
              {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312},
              {'name': 'SSP-ARK-4-2-3',       'exe': SSP_ARK_423}]


# # ---------------------- adaptive runs (in a single graph) -----------------------------
# for kvalue in kVal:
#     for runvalue in adaptive_ssp_params:
#         for solver in solvertype:
#             x, final_sol = runtest(solver, "adaptive", runvalue, kvalue, showcommand=True, sspcommand=True)
#             plt.plot(x, final_sol, label=solver['name'])

#         plt.title(f"Adaptive solution at final step with (value = {runvalue}, k = {kvalue})") 
#         plt.xlabel(r"$x$")
#         plt.legend()
#         plt.show()


# # ---------------------- fixed runs (in a single graph) -----------------------------
# for kvalue in kVal:
#     for runvalue in fixed_ssp_params:
#         for solver in solvertype:
#             x, final_sol = runtest(solver, "fixed", runvalue, kvalue, showcommand=True, sspcommand=True)
#             plt.plot(x, final_sol, label=solver['name'])

#         plt.title(f"Fixed solution at final step with (value = {runvalue}, k = {kvalue})") 
#         plt.xlabel(r"$x$")
#         plt.legend()
#         plt.show()

# ---------------------- adaptive runs -----------------------------
for kname, kvalue in kVal.items():
    for ii, runvalue in enumerate(adaptive_ssp_params):
        ## subplots 
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 4, figure=fig)
        for i, solver in enumerate(solvertype):
            x, final_sol = runtest(solver, "adaptive", runvalue, kvalue, showcommand=True, sspcommand=True)
            ax = fig.add_subplot(gs[0, i]) 
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"pSol")
            ax.set_title(f"{solver['name']}")
            ax.plot(x, final_sol)
        filename = f"Plot_adaptive_{kname}_{ii}.png"
        plt.suptitle(f"Adaptive solution at final step with (value = {runvalue}, k = {kvalue})")        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.savefig(filename)
        plt.close()


# ---------------------- fixed runs -----------------------------
for kname, kvalue in kVal.items():
    for ii, runvalue in enumerate(fixed_ssp_params):
        ## subplots 
        fig = plt.figure(figsize=(10, 5))
        gs = GridSpec(1, 4, figure=fig)
        for i, solver in enumerate(solvertype):
            x, final_sol = runtest(solver, "fixed", runvalue, kvalue, showcommand=True, sspcommand=True)
            ax = fig.add_subplot(gs[0, i]) 
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"pSol")
            ax.set_title(f"{solver['name']}")
            ax.plot(x, final_sol)
        filename = f"Plot_fixed_{kname}_{ii}.png"
        plt.suptitle(f"Fixed solution at final step with (value = {runvalue}, k = {kvalue})")        
        plt.tight_layout(rect=[0, 0, 1, 0.95]) 
        plt.savefig(filename)
        plt.close()

