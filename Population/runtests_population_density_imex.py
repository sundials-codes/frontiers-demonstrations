#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: If running fixed step sizes, ensure that fixedRun = True (2 locations) and fixhRun = True (1 location) 
#         in this script,runtests_population_density_imex.py Also, ensure that FixedRun = True (1 location) in the script,
#         plot_population.py . This means that adaptiveRun = False (2 locations) and adaptRun = False (1 location) in 
#         this script, runtests_population_density_imex.py and, AdaptiveRun = False (1 location) in the script plot_population.py 
#
#         Similarly, if running adaptive step sizes, ensure that adaptiveRun = True (2 locations) and
#         adaptRun = True (1 location) in this script, runtests_population_density_imex.py . Also, ensure that 
#         AdaptiveRun = True (1 location) in the script, plot_population.py . This means that fixedRun = False (2 locations) and 
#         fixhRun = False (1 location) in this script, runtests_population_density_imex.py and, FixedRun = False (1 location) 
#         in the script plot_population.py 
#-------------------------------------------------------------------------------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import shlex
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, runN, runV, kName, kVal, commonargs, showcommand=True, sspcommand=True, adaptiveRun=True, fixedRun=False):
    stats = {'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 'runVal': runV,
             'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0,
             'Nonlinear_Solves':0, 'Negative_model': 0, 'lmax_1dev': 0.0, 'error': 0.0, 'sspCondition': " "}
    
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
    ssp_result = subprocess.run(shlex.split(sspcommand), stdout=subprocess.PIPE)
    if (sspcommand):
        print("Run solution graph: " + sspcommand + " SUCCESS")
        new_fileName = f"soln_graph_{solver['name']}_{runN}_{kName}.png"

        ## rename plot file
        if os.path.exists("populationModel_frames.png"):
            os.rename("populationModel_frames.png", new_fileName)
            print(f"Plot saved as: {new_fileName}")
        else:
            print("Warning: populationModel_frames.png not found.")
        ## end if-else statement 
    ## end if statement
    pylines = str(ssp_result.stdout).split()
    lmax_1dev = float(pylines[8].replace('\\nLmax', ''))
    lmax_error = float(pylines[13].replace("\\n'", ''))
    # lmax_error = f"{lmax_error:e}"

    stats['lmax_1dev'] = lmax_1dev #lmax val for first derivative
    stats['error']     = lmax_error # lmax error after comparing solution at final time step with reference solution

    if (kVal==0.02) and (lmax_1dev >= 1.2) and (lmax_1dev <= 1.7) and (stats['Negative_model'] == 0):
        stats['sspCondition'] = str('ssp')
    elif (kVal==0.04) and (lmax_1dev >= 0.7) and (lmax_1dev <= 1.5) and (stats['Negative_model'] == 0):
        stats['sspCondition'] = str('ssp')
    else:
        stats['sspCondition'] = str('not ssp')
    ## end if else statement        
        
    return stats
## end of function




# shortcuts to executable/configuration of different embedded IMEX SSP methods
ARKODE_SSP_212       = "./population_density_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
ARKODE_SSP_312       = "./population_density_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
ARKODE_SSP_LSPUM_312 = "./population_density_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
ARKODE_SSP_423       = "./population_density_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"            

## common testing parameters
common = " --output 2"

adaptRun = True  #if True ensure that adaptiveRun is True and vice versa
fixhRun  = False #if True ensure that fixedRun is True and vice versa

if (adaptRun):
    runParams = {'r1':1.e-1, 'r2':1.e-2, 'r3':1.e-3, 'r4':1.e-4, 'r5':1.e-5} ## Relative tolerances
    xlabel_name = 'rtol'
    fname_apd = 'adaptiveRun'
elif(fixhRun):
    runParams = {'h1':0.25, 'h2':0.50, 'h3':0.75, 'h4':1.00,  'h5':1.25,  'h6':1.50, 
                 'h7':1.75, 'h8':2.00, 'h9':2.25, 'h10':2.50, 'h11':2.75, 'h12':3.00} ## fixed time step sizes
    xlabel_name = 'h'
    fname_apd = 'fixedRun'
##end if-else statement

## Diffusion coefficients
diff_coef = {'kpt02':0.02, 'kpt04':0.04}

## Integrator types
solvertype = [{'name': 'IMEX_SSP_212',       'exe': ARKODE_SSP_212},
              {'name': 'IMEX_SSP_312',       'exe': ARKODE_SSP_312},
              {'name': 'IMEX_SSP_LSPUM_312', 'exe': ARKODE_SSP_LSPUM_312},
              {'name': 'IMEX_SSP_423',       'exe': ARKODE_SSP_423}]

# run tests and collect results as a pandas data frame

fname = f'population_density_imex_{fname_apd}' 
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_name, runV_val in runParams.items():
            for solver_adapt in solvertype:
              stat = runtest(solver_adapt, runV_name, runV_val, k_name, k_val, common, showcommand=True, sspcommand=True, adaptiveRun=True, fixedRun=False)
              RunStats.append(stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)


# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)


##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel(f'population_density_imex_{fname_apd}' + '.xlsx') # excel file

diff_coeff = [0.02, 0.04] #diffusion coefficients

# marker     = itertools.cycle(('s', 'v', 'o', 'P')) #different markers for each method
# lines      = ["-","--","-.",":"]                   #different linestyles for each method
# linecycler = cycle(lines)

## plot the different rtols against the number of RHS function evaluations for both the implicit and explicit methods
for dck in diff_coeff:
    # fig  = plt.figure(figsize=(10, 5))
    # gs   = GridSpec(1, 2, figure=fig)
    # ax00 = fig.add_subplot(gs[0, 0])  # implicit method 
    # ax01 = fig.add_subplot(gs[0, 1])  # explicit method

    if (adaptRun):
        data_adaptive = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "error", "Negative_model"]]
        for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
            SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
            # Plot the whole method line with '.' markers
            plt.plot(SSPmethodAdt_data['runVal'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
            # Overlay red 'x' markers where Negative_model == 1
            negative_model = SSPmethodAdt_data[SSPmethodAdt_data['Negative_model'] == 1]
            plt.plot(negative_model['runVal'], negative_model['error'], marker='x', linestyle='none', color='red')
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(xlabel_name)
            plt.ylabel('error')
            plt.title('rtol vs error for k = %.2f' %dck)
            plt.legend()
    elif (fixhRun):
        data_fixed = df[(df["diff_coef"] == dck)][["IMEX_method", "diff_coef", "runVal", "Explicit_RHS", "error", "Negative_model"]]
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            # Plot the whole method line with '.' markers
            plt.plot(SSPmethodFix_data['runVal'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
            # Overlay red 'x' markers where Negative_model == 1
            negative_model = SSPmethodFix_data[SSPmethodFix_data['Negative_model'] == 1]
            plt.plot(negative_model['runVal'], negative_model['error'], marker='x', linestyle='none', color='red')
            # plt.xscale('log')
            plt.yscale('log')
            plt.xlabel(xlabel_name)
            plt.ylabel('error')
            plt.title('cost vs error for k = %.2f' %dck)
            plt.legend()

    plt.savefig("Plot for k = %.2f.pdf"%dck)
    plt.show()



