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
        # print(lines)
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

# shortcuts to executable/configuration of different solver types
###LSRK_SSP methods
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
solvertype = [{'name': 'IMEX_SSP(2,1,2)',       'exe': ARKODE_SSP_2_1_2},
              {'name': 'IMEX_SSP(3,1,2)',       'exe': ARKODE_SSP_3_1_2},
              {'name': 'IMEX_SSP_LSPUM(3,1,2)', 'exe': ARKODE_SSP_LSPUM_3_1_2},
              {'name': 'IMEX_SSP(4,2,3)',       'exe': ARKODE_SSP_4_2_3}]


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
