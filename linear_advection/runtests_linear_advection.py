#!/usr/bin/env python
#------------------------------------------------------------
# Programmer(s):  Sylvia Amihere and Daniel Reynolds @ SMU
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
def runtest(solver, rtol, nx, commonargs, showcommand=True, sspcommand=True):
    stats = {'ReturnCode': 0, 'method': solver['name'], 'nx': nx, 'rtol': rtol,
             'dx': 0.0, 'CurrentTime': 0.0, 'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0,
             'RHSFE': 0, 'InitialDT': 0.0, 'CurrentDT': 0.0, 'LastDT': 0.0, 'AvgStepSize':0.0, 'args': commonargs, 
             'sspCondition': " "}
    
    runcommand = " %s --nx %i --rtol %e %s" % (solver['exe'], nx, rtol, commonargs)
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
            if ("dx" in txt):
                stats['dx'] = float(txt[2]) #spatial step size
            elif (("Current" in txt) and ("time" in txt)):
                stats['CurrentTime'] = float(txt[3])
            elif ("Steps" in txt):
                stats['Steps'] = int(txt[2])
            elif (("Step" in txt) and ("attempts" in txt)):
                stats['StepAttempts'] = int(txt[3])
            elif (("Error" in txt) and ("Fails" in txt)):
                stats['ErrTestFails'] = float(txt[4])
            elif (("RHS" in txt) and ("evals" in txt)):
                stats['RHSFE'] = int(txt[4])       #right hand side evaluations
            elif (("Initial" in txt) and ("step" in txt)):
                stats['InitialDT'] = float(txt[4]) #temporal step size
            elif (("Current" in txt) and ("step" in txt)):
                stats['CurrentDT'] = float(txt[4]) #temporal step size
            elif (("Last" in txt) and ("step" in txt)):
                stats['LastDT'] = float(txt[4])    #temporal step size

        if 'Steps' in stats and 'CurrentTime' in stats:
            stats['AvgStepSize'] = stats['CurrentTime']/stats['Steps']
        else:
            print("Error: 'Steps' or 'CurrentTime' not found in stats.")


        ## running python file to determine the ssp condition
        sspcommand = "python ./plot_advection.py"
        ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)

        if (sspcommand):
            print("Run ssp command " + sspcommand + " SUCCESS")
        pylines = str(ssp_result.stdout).split('\\n')
        # print(pylines)

        for pyline in pylines:
            txt = pyline.split()
            # print(txt)
            if (("Fails" in txt) and ("condition." in txt)):
                stats['sspCondition'] = str('stable not ssp')
                # print("yes "+txt[0])
            elif (("Satisfies" in txt) and ("condition" in txt)):
                stats['sspCondition'] = str('stable and ssp')
                # print("other "+txt[0])
        
    return stats


# filename to hold run statistics
fname = "linear_advection"


# shortcuts to executable/configuration of different solver types
###LSRK_SSP methods
LSRK_SSP_stg10_ord4 = "./linear_advection --integrator ARKODE_LSRK_SSP_10_4" #ARKODE_LSRK_SSP_10_4
LSRK_SSP_stg4_ord3 = "./linear_advection --ARKODE_LSRK_SSP_S_3 "             #ARKODE_LSRK_SSP_S_3 with 4 stages
LSRK_SSP_stg9_ord3 = "./linear_advection --ARKODE_LSRK_SSP_S_3 --stages 9"   #ARKODE_LSRK_SSP_S_3 with 9 stages
LSRK_SSP_stg2_ord2 = "./linear_advection --ARKODE_LSRK_SSP_S_2 "             #ARKODE_LSRK_SSP_S_2 with 2 stages
LSRK_SSP_stg10_ord2 = "./linear_advection --ARKODE_LSRK_SSP_S_2 --stages 10" #ARKODE_LSRK_SSP_S_2 with 10 stages

###ERK_SSP methods
ERK_SSP_stg10_ord4 = "./linear_advection --ARKODE_SSP_ERK_10_3_4" #ARKODE_SSP_ERK_10_3_4
ERK_SSP_stg9_ord3 = "./linear_advection --ARKODE_SSP_ERK_9_2_3"   #ARKODE_SSP_ERK_9_2_3
ERK_SSP_stg10_ord2 = "./linear_advection --ARKODE_SSP_ERK_10_1_2" #ARKODE_SSP_ERK_10_1_2
ERK_SSP_stg4_ord3 = "./linear_advection --ARKODE_SSP_ERK_4_2_3"   #ARKODE_SSP_ERK_4_2_3
ERK_SSP_stg2_ord2 = "./linear_advection --ARKODE_SSP_ERK_2_1_2"   #ARKODE_SSP_ERK_2_1_2

ssp_condition = "python ./plot_advection.py"

# common testing parameters
output = " --output 2"
nout = " --nout 100"
common = output + nout 

## Dimension or mesh sizes and relative tolerances
dim_mesh    = [150, 200, 250, 300, 350] 
rtols = [1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6]


## Integrator types
solvertype = [{'name': 'LSRK_SSP_stg10_ord4', 'exe': LSRK_SSP_stg10_ord4},
              {'name': 'LSRK_SSP_stg4_ord3', 'exe': LSRK_SSP_stg4_ord3},
              {'name': 'LSRK_SSP_stg9_ord3', 'exe': LSRK_SSP_stg9_ord3},
              {'name': 'LSRK_SSP_stg2_ord2', 'exe': LSRK_SSP_stg2_ord2},
              {'name': 'LSRK_SSP_stg10_ord2', 'exe': LSRK_SSP_stg10_ord2},
              {'name': 'ERK_SSP_stg10_ord4', 'exe': ERK_SSP_stg10_ord4},
              {'name': 'ERK_SSP_stg9_ord3', 'exe': ERK_SSP_stg9_ord3},
              {'name': 'ERK_SSP_stg10_ord2', 'exe': ERK_SSP_stg10_ord2},
              {'name': 'ERK_SSP_stg4_ord3', 'exe': ERK_SSP_stg4_ord3},
              {'name': 'ERK_SSP_stg2_ord2', 'exe': ERK_SSP_stg2_ord2}]

# solvertype = [{'name': 'LSRK_SSP_stg10_ord4', 'exe': LSRK_SSP_stg10_ord4}]

# sspStat = [{'exe': ssp_condition}]


# run tests and collect results as a pandas data frame
RunStats = []
for rtol in rtols:
    for nx in dim_mesh:
        for solver in solvertype:
            stat = runtest(solver, rtol, nx, common)
            RunStats.append(stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)


# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)
