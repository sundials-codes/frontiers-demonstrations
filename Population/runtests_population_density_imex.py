#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
#------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import shlex
import sys
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, rtol, k, commonargs, showcommand=True):
    stats = {'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': k, 'rtol': rtol,
             'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0}
    
    runcommand = " %s  --rtol %e  --k %.2f" % (solver['exe'], rtol, k)
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
    stats['ReturnCode'] = result.returncode

    if (result.returncode != 0):
        print("Run command " + runcommand + " FAILURE: " + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Run command " + runcommand + " SUCCESS")
        lines = str(result.stdout).split('\\n')
        for line in lines:
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
        
    return stats


# filename to hold run statistics
fname = "population_density_imex"

# shortcuts to executable/configuration of different embedded IMEX SSP methods
ARKODE_SSP_2_1_2       = "./population_density_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
ARKODE_SSP_3_1_2       = "./population_density_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
ARKODE_SSP_LSPUM_3_1_2 = "./population_density_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
ARKODE_SSP_4_2_3       = "./population_density_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"            

## common testing parameters
common = " --output 2"

## Relative tolerances
rtols = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5]

## Diffusion coefficients
diff_coef = [0.00, 0.02, 0.04]

## Integrator types
solvertype = [{'name': 'IMEX_SSP_212',       'exe': ARKODE_SSP_2_1_2},
              {'name': 'IMEX_SSP_312',       'exe': ARKODE_SSP_3_1_2},
              {'name': 'IMEX_SSP_LSPUM_312', 'exe': ARKODE_SSP_LSPUM_3_1_2},
              {'name': 'IMEX_SSP_423',       'exe': ARKODE_SSP_4_2_3}]

# run tests and collect results as a pandas data frame
RunStats = []
for k in diff_coef:
    for rtol in rtols:
        for solver in solvertype:
            stat = runtest(solver, rtol, k, common)
            RunStats.append(stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)


# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)

##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel('population_density_imex.xlsx') # excel file

diff_coeff = [0, 0.02, 0.04] #diffusion coefficients

## plot the different rtols against the number of RHS function evaluations for both the implicit and explicit methods
for dck in diff_coeff:
    fig  = plt.figure(figsize=(10, 5))
    gs   = GridSpec(1, 2, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0])  # implicit method 
    ax01 = fig.add_subplot(gs[0, 1])  # explicit method

    data_implicit = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "rtol", "Implicit_RHS"]]
    for IMmethod in data_implicit['IMEX_method'].unique():
        IMmethod_data = data_implicit[data_implicit['IMEX_method'] == IMmethod]
        ax00.plot(IMmethod_data['rtol'], IMmethod_data['Implicit_RHS'], marker='o',label=IMmethod)
        ax00.set_xscale('log')
        ax00.set_yscale('log')
        ax00.set_xlabel('rtol')
        ax00.set_ylabel('Implicit RHS fn evals')
        ax00.set_title('IM-RHS vs rtol')
        ax00.legend()

    data_explicit = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "rtol", "Explicit_RHS"]]
    for EXmethod in data_explicit['IMEX_method'].unique():
        EXmethod_data = data_explicit[data_explicit['IMEX_method'] == EXmethod]
        ax01.plot(EXmethod_data['rtol'], EXmethod_data['Explicit_RHS'], marker='x',label=EXmethod)
        ax01.set_xscale('log')
        ax01.set_yscale('log')
        ax01.set_xlabel('rtol')
        ax01.set_ylabel('Explicit RHS fn evals')
        ax01.set_title('EX-RHS vs rtol')
        ax01.legend()

    plt.suptitle('RHS fn evals for k = %.2f' %dck)
    plt.savefig("RHS fn evals for k = %.2f.pdf"%dck)
    plt.show()



