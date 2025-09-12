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
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec
# fixedRun=False,
# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, runN, runV, kName, kVal, commonargs, showcommand=True, sspcommand=True, adaptiveRun=False, fixedRun=True):
    stats = {'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 'runVal': runV,
             'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0,
             'Nonlinear_Solves':0, 'Negative_model': 0}
    
    if (adaptiveRun):
        runcommand = " %s  --rtol %e  --k %.2f" % (solver['exe'], runV, kVal)
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
        stats['ReturnCode'] = result.returncode
    elif(fixedRun):
        runcommand = " %s  --fixed_h %.2f  --k %.2f" % (solver['exe'], runV, kVal)
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
        stats['ReturnCode'] = result.returncode
    ## end if-else statement

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
            elif (("NLS" in txt) and ("iters" in txt) and ("per" not in txt) and ("step" not in txt)):
                stats['Nonlinear_Solves'] = int(txt[3])   #right hand side evaluations for implicit method
            elif (("Model" in txt) and ("has" in txt) and ("a" in txt) and ("negative" in txt) and ("time" in txt) and ("step" in txt) and ("t" in txt) and ("10.00" in txt)):
                stats['Negative_model'] = 1               #right hand side evaluations for implicit method


    ## running python file to determine the if the graph is smooth and positive or not (ssp condition)
    sspcommand = " python ./plot_population.py"
    if (sspcommand):
        print("Run solution graph: " + sspcommand )
        subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)
        new_fileName = f"soln_graph_{solver['name']}_{runN}_{kName}.png"
        # if (adaptiveRun):
        #     new_fileName = f"soln_graph_{solver['name']}_{runN}_{kName}.png"
        # elif(fixedRun):
        #     new_fileName = f"soln_graph_{solver['name']}_{runN}_{kName}.png"
        #end if-else statement

        #rename plot file
        if os.path.exists("populationModel_frames.png"):
            os.rename("populationModel_frames.png", new_fileName)
            print(f"Plot saved as: {new_fileName}")
        else:
            print("Warning: populationModel_frames.png not found.")
        #end if-else statement 
    #end if statement
        
    return stats
## end of function


# filename to hold run statistics
fname = "population_density_imex"

# shortcuts to executable/configuration of different embedded IMEX SSP methods
ARKODE_SSP_2_1_2       = "./population_density_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
ARKODE_SSP_3_1_2       = "./population_density_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
ARKODE_SSP_LSPUM_3_1_2 = "./population_density_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
ARKODE_SSP_4_2_3       = "./population_density_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"            

## common testing parameters
common = " --output 2"

adaptRun = False  #if True ensure that adaptiveRun is True and vice versa
fixh_Run = True #if True ensure that fixedRun is True and vice versa

if (adaptRun):
    ## Relative tolerances
    runParams = {'r1':1.e-1, 'r2':1.e-2, 'r3':1.e-3, 'r4':1.e-4, 'r5':1.e-5}
elif(fixh_Run):
    ## fixed time step sizes
    runParams = {'h1':0.25, 'h2':0.50,  'h3':0.75,  'h4':1.00,  'h5':1.25,  'h6':1.50,  'h7':1.75,  'h8':2.00, 
                 'h9':2.25, 'h10':2.50, 'h11':2.75, 'h12':3.00, 'h13':3.25, 'h14':3.50, 'h15':3.75, 'h16':4.00}
##end if-else statement

## Diffusion coefficients
diff_coef = {'k0':0.00, 'kpt02':0.02, 'kpt04':0.04}

## Integrator types
solvertype = [{'name': 'IMEX_SSP_212',       'exe': ARKODE_SSP_2_1_2},
              {'name': 'IMEX_SSP_312',       'exe': ARKODE_SSP_3_1_2},
              {'name': 'IMEX_SSP_LSPUM_312', 'exe': ARKODE_SSP_LSPUM_3_1_2},
              {'name': 'IMEX_SSP_423',       'exe': ARKODE_SSP_4_2_3}]

# run tests and collect results as a pandas data frame
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_name, runV_val in runParams.items():
            for solver_adapt in solvertype:
              stat = runtest(solver_adapt, runV_name, runV_val, k_name, k_val, common, showcommand=True, sspcommand=True, adaptiveRun=False, fixedRun=True)
              RunStats.append(stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)


# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)


##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel('population_density_imex.xlsx') # excel file

diff_coeff = [0.00, 0.02, 0.04] #diffusion coefficients

# marker     = itertools.cycle(('s', 'v', 'o', 'P')) #different markers for each method
# lines      = ["-","--","-.",":"]                   #different linestyles for each method
# linecycler = cycle(lines)

## plot the different rtols against the number of RHS function evaluations for both the implicit and explicit methods
for dck in diff_coeff:
    fig  = plt.figure(figsize=(10, 5))
    gs   = GridSpec(1, 2, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0])  # implicit method 
    ax01 = fig.add_subplot(gs[0, 1])  # explicit method

    data_implicit = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Negative_model"]]
    for IMmethod in data_implicit['IMEX_method'].unique():
        IMmethod_data = data_implicit[data_implicit['IMEX_method'] == IMmethod]
        # ax00.plot(IMmethod_data['runVal'], IMmethod_data['Nonlinear_Solves'], marker=next(marker), linestyle=next(linecycler), linewidth = '2', label=IMmethod)
        if (IMmethod_data['Negative_model'] == 1).any():
            marker = '*'
        else:
            marker = 'o'
        #end if-else statement
        ax00.plot(IMmethod_data['runVal'], IMmethod_data['Nonlinear_Solves'], marker=marker, label=IMmethod)
        ax00.set_xscale('log')
        ax00.set_yscale('log')
        ax00.set_xlabel('runVal')
        ax00.set_ylabel('Nonlinear solves')
        ax00.set_title('IM-solves vs runVal')
        ax00.legend()

    data_explicit = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "runVal", "Explicit_RHS", "Negative_model"]]
    for EXmethod in data_explicit['IMEX_method'].unique():
        EXmethod_data = data_explicit[data_explicit['IMEX_method'] == EXmethod]
        # ax01.plot(EXmethod_data['runVal'], EXmethod_data['Explicit_RHS'], marker=next(marker), linestyle=next(linecycler), linewidth = '2', label=EXmethod)
        if (EXmethod_data['Negative_model'] == 1).any():
            marker = '*'
        else:
            marker = 'o'
        #end if-else statement
        ax01.plot(EXmethod_data['runVal'], EXmethod_data['Explicit_RHS'], marker=marker, label=EXmethod)
        ax01.set_xscale('log')
        ax01.set_yscale('log')
        ax01.set_xlabel('runVal')
        ax01.set_ylabel('Explicit RHS solves')
        ax01.set_title('EX-solves vs runVal')
        ax01.legend()

    plt.suptitle('RHS fn evals for k = %.2f' %dck)
    plt.savefig("RHS fn evals for k = %.2f.pdf"%dck)
    plt.show()



