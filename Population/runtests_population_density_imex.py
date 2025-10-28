#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
# All rights reserved.
# For details, see the LICENSE file.
#------------------------------------------------------------------------------------------------------------------------------------
# ReadME: If running fixed step sizes, ensure that fixedRun = True (1 location) and fixhRun = True (1 location) 
#         in this script,runtests_population_density_imex.py Also, ensure that FixedRun = True (1 location) in the script,
#         plot_population.py . This means that adaptiveRun = False (1 location) and adaptRun = False (1 location) in 
#         this script, runtests_population_density_imex.py and, AdaptiveRun = False (1 location) in the script plot_population.py 
#
#         Similarly, if running adaptive step sizes, ensure that adaptiveRun = True (1 location) and
#         adaptRun = True (1 location) in this script, runtests_population_density_imex.py . Also, ensure that 
#         AdaptiveRun = True (1 location) in the script, plot_population.py . This means that fixedRun = False (1 location) and 
#         fixhRun = False (1 location) in this script, runtests_population_density_imex.py and, FixedRun = False (1 location) 
#         in the script plot_population.py 
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

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, runN, runV, kName, kVal, commonargs, showcommand=True, sspcommand=True):
    def runtype(modetype):
        stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 'runVal': runV,
                'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 'Total Func Eval':0,
                'maxIntStep': 0.0, 'Nonlinear_Solves':0, 'Negative_model': 0, 'runtime':0.0, 'lmax_1dev': 0.0, 'error': 0.0,
                'sspCondition': " "}

        if (modetype == "adaptive"):
            runcommand = " %s  --rtol %e  --k %.2f" % (solver['exe'], runV, kVal)
        elif (modetype == "fixed"):
            runcommand = " %s  --fixed_h %.2f  --k %.2f" % (solver['exe'], runV, kVal)
        
        start_time = time.time()
        result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
        end_time = time.time()
        length_time = end_time - start_time
        stats['Runtype']    = modetype
        stats['ReturnCode'] = result.returncode
        stats['runtime']    = length_time

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
                elif (("Maximum" in txt) and ("internal" in txt) and ("step" in txt) and ("size" in txt)):
                    stats['maxIntStep'] = float(txt[6])         #last internal step size used in adaptive run
            sum_negLines = 0
            for line in lines:
                txt = line.split()
                if (("Model" in txt) and ("has" in txt) and ("a" in txt) and ("negative" in txt) and ("time" in txt) and ("step" in txt) and ("t" in txt)):
                    sum_negLines += 1
            stats['Negative_model'] = sum_negLines               #right hand side evaluations for implicit method
            stats['Total Func Eval'] = stats['Implicit_RHS'] + stats['Explicit_RHS']


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

        pylines = str(ssp_result.stdout).split()
        lmax_1dev = float(pylines[8].replace('\\nLmax', ''))
        lmax_error = float(pylines[13].replace("\\n'", ''))

        stats['lmax_1dev'] = lmax_1dev #lmax val for first derivative
        stats['error']     = lmax_error # lmax error after comparing solution at final time step with reference solution

        if (kVal==0.02) and (lmax_1dev >= 1.2) and (lmax_1dev <= 1.7) and (stats['Negative_model'] == 0):
            stats['sspCondition'] = str('ssp')
        elif (kVal==0.04) and (lmax_1dev >= 0.7) and (lmax_1dev <= 1.5) and (stats['Negative_model'] == 0):
            stats['sspCondition'] = str('ssp')
        else:
            stats['sspCondition'] = str('not ssp')       
            
        return stats
    
    adaptive_stats = runtype("adaptive")
    fixed_stats    = runtype("fixed")
    return adaptive_stats, fixed_stats
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP_ARK_212       = "./population_density_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./population_density_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./population_density_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./population_density_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"            

## common testing parameters
common = " --output 2"

adaptive_params = {'r1':1.e-1, 'r2':1.e-2, 'r3':1.e-3, 'r4':1.e-4, 'r5':1.e-5} ## Relative tolerances
fixed_params    = {'h1':0.25*(2**-4), 'h2':0.25*(2**-3), 'h3':0.25*(2**-2), 'h4':0.25*(2**-1),  'h5':0.25*(2**0), 
                   'h6':0.25*(2**1), 'h7':0.25*(2**2), 'h8':0.25*(2**3), 'h9':0.25*(2**4)} ## fixed time step sizes

## Diffusion coefficients
diff_coef = {'kpt02':0.02, 'kpt04':0.04}

## Integrator types
solvertype = [{'name': 'SSP-ARK-2-1-2',       'exe': SSP_ARK_212},
              {'name': 'SSP-ARK-3-1-2',       'exe': SSP_ARK_312},
              {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312},
              {'name': 'SSP-ARK-4-2-3',       'exe': SSP_ARK_423}]

# run tests and collect results as a pandas data frame
fname = 'population_density_imex_stats' 
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_name, runV_val in adaptive_params.items():
            for solver_adapt in solvertype:
              adaptive_result, _ = runtest(solver_adapt, runV_name, runV_val, k_name, k_val, common, showcommand=True, sspcommand=True)
              RunStats.append(adaptive_result)

    for runV_name, runV_val in fixed_params.items():
            for solver_adapt in solvertype:
              _, fixed_result = runtest(solver_adapt, runV_name, runV_val, k_name, k_val, common, showcommand=True, sspcommand=True)
              RunStats.append(fixed_result)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)


##---------------------------------------------- Efficiency Plots ---------------------------------------------
df = pd.read_excel('population_density_imex_stats' + '.xlsx') # excel file

diff_coeff = [0.02, 0.04] #diffusion coefficients

adapt_accuracy         = True
adapt_efficiency_time  = True
adapt_efficiency_steps = True
fixed_convergence      = True
fixed_efficiency_work  = True
fixed_efficiency_time  = True

for dck in diff_coeff:    
# # --------------------------------------------------- Run Adaptive Time Steps --------------------------------------------------------------------------------  
    data_adaptive = df[(df["diff_coef"] == dck) & (df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
                                                                                  "Total Func Eval", "maxIntStep", "error", "Negative_model", "sspCondition", "runtime", "Steps"]]
    if (adapt_accuracy):
        plt.figure()
        for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
            SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodAdt_data['runVal'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
            plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('rtol')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Adaptive accuracy plot with d = %.2f.pdf"%dck)
        plt.show()

    if (adapt_efficiency_time):
        plt.figure()
        for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
            SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodAdt_data['runtime'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
            # negative_model = SSPmethodAdt_data[(SSPmethodAdt_data['Negative_model'] == 1) & (SSPmethodAdt_data['sspCondition'] == "not ssp")]
            plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Adaptive efficiency time plot with d = %.2f.pdf"%dck)
        plt.show()

    if (adapt_efficiency_steps):
        plt.figure()
        for SSPmethodAdt in data_adaptive['IMEX_method'].unique():
            SSPmethodAdt_data = data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdt]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodAdt_data['Steps'], SSPmethodAdt_data['error'], marker='.', linestyle='-', label=SSPmethodAdt)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodAdt_data[SSPmethodAdt_data['sspCondition'] == "not ssp"]
            # negative_model = SSPmethodAdt_data[(SSPmethodAdt_data['Negative_model'] == 1) & (SSPmethodAdt_data['sspCondition'] == "not ssp")]
            plt.plot(sspness['Steps'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of steps')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Adaptive efficiency steps plot with d = %.2f.pdf"%dck)
        plt.show()

# # ------------------------------------------------ Run Fixed Time Steps ---------------------------------------------------------------------------            
    data_fixed = df[(df["diff_coef"] == dck) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
                                                                            "Total Func Eval", "error", "maxIntStep", "Negative_model", "sspCondition", "runtime", "Steps"]]
    if (fixed_convergence):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodFix_data['runVal'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
            # negative_model = SSPmethodFix_data[(SSPmethodFix_data['Negative_model'] == 1) & (SSPmethodFix_data['sspCondition'] == "not ssp")]
            plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('h')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Fixed convergence plot with d = %.2f.pdf"%dck)
        plt.show()

    if (fixed_efficiency_time):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodFix_data['runtime'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
            # negative_model = SSPmethodFix_data[(SSPmethodFix_data['Negative_model'] == 1) & (SSPmethodFix_data['sspCondition'] == "not ssp")]
            plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Fixed effciency time plot for d = %.2f.pdf"%dck)
        plt.show()

    if (fixed_efficiency_work):
        plt.figure()
        for SSPmethodFix in data_fixed['IMEX_method'].unique():
            SSPmethodFix_data = data_fixed[data_fixed['IMEX_method'] == SSPmethodFix]
            # Plot the whole method line with '.' markers
            method_line = plt.plot(SSPmethodFix_data['Total Func Eval'], SSPmethodFix_data['error'],marker='.', linestyle='-', label=SSPmethodFix)
            method_line_color = method_line[0].get_color()
            # Overlay red 'x' markers where Negative_model == 1 or "not ssp"
            sspness = SSPmethodFix_data[SSPmethodFix_data['sspCondition'] == "not ssp"]
            # negative_model = SSPmethodFix_data[(SSPmethodFix_data['Negative_model'] == 1) & (SSPmethodFix_data['sspCondition'] == "not ssp")]
            plt.plot(sspness['Total Func Eval'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Total Num of Func Evals')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig("Fixed effciency work for d = %.2f.pdf"%dck)
        plt.show()



