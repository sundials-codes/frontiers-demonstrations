#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
#------------------------------------------------------------------------------------------------------------------------------------
# Copyright (c) 2025, Southern Methodist University.
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
import shlex
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import itertools
from itertools import cycle
from matplotlib.gridspec import GridSpec
from math import log10, floor

# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, modetype, runV, kVal, showcommand=True, sspcommand=True):
    """
    This function runs the population model using both fixed and adaptive time
    stepping with different parameters and stores the stats in an excel file

    Input: solver:            imex scheme to tun
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient

    Output: returns the statistics
    """
    stats = {'Runtype': modetype,'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal, 'runVal': runV,
            'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0, 'Implicit_RHS': 0, 'Total Func Eval':0,
            'maxIntStep': 0.0, 'Nonlinear_Solves':0, 'Negative_model': 0, 'runtime':0.0, 'lmax_1dev': 0.0, 'error': 0.0,
            'sspCondition': " "}

    if (modetype == "adaptive"):
        runcommand = " %s  --rtol %.6f --k %.2f" % (solver['exe'], runV, kVal)
    elif (modetype == "fixed"):
        runcommand = " %s  --fixed_h %.6f --k %.2f" % (solver['exe'], runV, kVal)
    
    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE)
    end_time = time.time()
    length_time = end_time - start_time
    stats['Runtype']    = modetype
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = length_time

    if (result.returncode != 0):
        print("Running: " + runcommand + " FAILURE: \n" + str(result.returncode))
        print(result.stderr)
    else:
        if (showcommand):
            print("Running: " + runcommand + " SUCCESS")
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
            elif (("Largest" in txt) and ("average" in txt) and ("step" in txt) and ("size" in txt)):
                stats['maxIntStep'] = float(txt[7])         #last internal step size used in adaptive run
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
    # if (sspcommand):
    #         print("Run solution graph: " + sspcommand + " SUCCESS")
    #         new_fileName = f"soln_graph_{solver['name']}_{runN}_{kName}.png"

    #         ## rename plot file
    #         if os.path.exists("populationModel_frames.png"):
    #             os.rename("populationModel_frames.png", new_fileName)
    #             print(f"Plot saved as: {new_fileName}")
    #         else:
    #             print("Warning: populationModel_frames.png not found.")

    pylines = str(ssp_result.stdout).split()
    lmax_1dev = float(pylines[8].replace('\\nLmax', ''))
    lmax_error = float(pylines[13].replace("\\n'", ''))

    stats['lmax_1dev'] = lmax_1dev #lmax val for first derivative
    stats['error']     = lmax_error # lmax error after comparing solution at final time step with reference solution

    # # assessing SSPness based on positivity at all time steps and smooth profile at final time step
    # if (kVal==0.02) and (lmax_1dev >= 1.2) and (lmax_1dev <= 1.7) and (stats['Negative_model'] == 0):
    #     stats['sspCondition'] = str('ssp')
    #     ssp_cond = 0
    # elif (kVal==0.04) and (lmax_1dev >= 0.7) and (lmax_1dev <= 1.5) and (stats['Negative_model'] == 0):
    #     stats['sspCondition'] = str('ssp')
    #     ssp_cond = 0
    # else:
    #     stats['sspCondition'] = str('not ssp')  
    #     ssp_cond = 1   

    # assessing SSPness based on positivity at all time steps
    if (stats['Negative_model'] == 0):
        stats['sspCondition'] = str('ssp')
        ssp_cond = 0
    else:
        stats['sspCondition'] = str('not ssp')  
        ssp_cond = 1      
        
    return stats, ssp_cond
## end of function


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP_ARK_212       = "./population_imex  --IMintegrator ARKODE_SSP_SDIRK_2_1_2        --EXintegrator ARKODE_SSP_ERK_2_1_2" 
SSP_ARK_312       = "./population_imex  --IMintegrator ARKODE_SSP_DIRK_3_1_2         --EXintegrator ARKODE_SSP_ERK_3_1_2"           
SSP_LSPUM_ARK_312 = "./population_imex  --IMintegrator ARKODE_SSP_LSPUM_SDIRK_3_1_2  --EXintegrator ARKODE_SSP_LSPUM_ERK_3_1_2"  
SSP_ARK_423       = "./population_imex  --IMintegrator ARKODE_SSP_ESDIRK_4_2_3       --EXintegrator ARKODE_SSP_ERK_4_2_3"     

adaptive_params = [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0] ## relative tolerances
fixed_params    = [0.25*(2**-5),  0.25*(2**-4), 0.25*(2**-3), 0.25*(2**-2), 0.25*(2**-1),
                   0.25*(2**0),   0.25*(2**1),  0.25*(2**2),  0.25*(2**3),  0.25*(2**4)] ## fixed time step sizes


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

        _,condLow = runtest(solver, runtype, solver['sspVal'], kval, showcommand=True, sspcommand=True)
        _,condHigh = runtest(solver, runtype,solver['nonsspVal'], kval, showcommand=True, sspcommand=True)

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
            _,condMid =  runtest(solver, runtype, midVal, kval, showcommand=True, sspcommand=True)
            if (condMid==0):
                solver['sspVal'] = midVal
            elif (condMid==1):
                solver['nonsspVal'] = midVal

            # end run both values have the same ssp condition and store the last midpoint point value 
            # as well as the previous distinct midpoint value
            _,condLow  = runtest(solver, runtype, solver['sspVal'], kval, showcommand=True, sspcommand=True)
            _,condHigh = runtest(solver, runtype, solver['nonsspVal'], kval, showcommand=True, sspcommand=True)
            if (condLow==condHigh):
                if second_to_preMidVal is not None:
                    paramList.append(second_to_preMidVal)
                paramList.append(midVal)
                print(f"Both values have the same SSP condition ({condLow}).")
                break
            iter += 1

        # print results
        print(f"{runtype} run with {name}, {kval}, iter {iter} : SSP-value & cond = {solver['sspVal'],condLow}, NonSSP-value & cond = {solver['nonsspVal'], condHigh}")

# -------------------------------------- adaptive runs -----------------------------------------
solvernames_adaptK0 = [#{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 1e-2, 'nonsspVal': 5e-2, 'kvalue': 0.0},
                       #{'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.0},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.0}
                       #{'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 1e-3, 'nonsspVal': 1e-2, 'kvalue': 0.0} 
                       ]

solvernames_adaptK2 = [{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02},
                       {'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.02},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 5e-2, 'nonsspVal': 1e-1, 'kvalue': 0.02},
                       {'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 1e-3, 'nonsspVal': 1e-2, 'kvalue': 0.02} ]

solvernames_adaptK4 = [{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04},
                    #    {'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,           'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 5e-2, 'nonsspVal': 1e-1, 'kvalue': 0.04},
                       {'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 1e-1, 'nonsspVal': 5e-1, 'kvalue': 0.04} ]

bisection_midval(solvernames_adaptK0, "adaptive", paramList = adaptive_params)
bisection_midval(solvernames_adaptK2, "adaptive", paramList = adaptive_params)
bisection_midval(solvernames_adaptK4, "adaptive", paramList = adaptive_params)


# -------------------------------------- fixed runs -----------------------------------------
solvernames_fixedK0 = [{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.0},
                       {'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,             'sspVal': 0.25*(2**3), 'nonsspVal': 0.25*(2**4), 'kvalue': 0.0},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.0},
                       {'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 0.25*(2**3), 'nonsspVal': 0.25*(2**4), 'kvalue': 0.0} ]

solvernames_fixedK2 = [{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 0.25*(2**1), 'nonsspVal': 0.25*(2**2), 'kvalue': 0.02},
                       {'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,             'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.02},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.02},
                       {'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 0.25*(2**0), 'nonsspVal': 0.25*(2**1), 'kvalue': 0.02} ]

solvernames_fixedK4 = [{'name': 'SSP-ARK-2-1-2', 'exe': SSP_ARK_212,             'sspVal': 0.25*(2**1), 'nonsspVal': 0.25*(2**2), 'kvalue': 0.04},
                       {'name': 'SSP-ARK-3-1-2', 'exe': SSP_ARK_312,             'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.04},
                       {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312, 'sspVal': 0.25*(2**2), 'nonsspVal': 0.25*(2**3), 'kvalue': 0.04},
                       {'name': 'SSP-ARK-4-2-3', 'exe': SSP_ARK_423,             'sspVal': 0.25*(2**0), 'nonsspVal': 0.25*(2**1), 'kvalue': 0.04} ]

bisection_midval(solvernames_fixedK0, "fixed", paramList = fixed_params)
bisection_midval(solvernames_fixedK2, "fixed", paramList = fixed_params)
bisection_midval(solvernames_fixedK4, "fixed", paramList = fixed_params)


## ----------------------------------------------------------------------------------------------------
# This section generates the data for each method, diffusion coefficient with different fixed step
# sizes and rtols
## ----------------------------------------------------------------------------------------------------
sorted_adaptive_params = sorted(adaptive_params) ## relative tolerances
sorted_fixed_params    = sorted(fixed_params) ## fixed time step sizes

## Diffusion coefficients
diff_coef = {'k0':0.0, 'k2':0.02, 'k4':0.04}

## Integrator types
solvertype = [{'name': 'SSP-ARK-2-1-2',       'exe': SSP_ARK_212},
              {'name': 'SSP-ARK-3-1-2',       'exe': SSP_ARK_312},
              {'name': 'SSP-LSPUM-ARK-3-1-2', 'exe': SSP_LSPUM_ARK_312},
              {'name': 'SSP-ARK-4-2-3',       'exe': SSP_ARK_423}]

# run tests and collect results as a pandas data frame
fname = 'population_density_imex_stats' 
RunStats = []
for k_name, k_val in diff_coef.items():
    for runV_val in sorted_adaptive_params:
        for solver_adapt in solvertype:
            adaptive_stat, _ = runtest(solver_adapt, "adaptive", runV_val, k_val, showcommand=True, sspcommand=True)
            RunStats.append(adaptive_stat)

    for runV_val in sorted_fixed_params:
        for solver_adapt in solvertype:
            fixed_stat, _ = runtest(solver_adapt, "fixed", runV_val, k_val, showcommand=True, sspcommand=True)
            RunStats.append(fixed_stat)
RunStatsDf = pd.DataFrame.from_records(RunStats)

# save dataframe as Excel file
print("RunStatsDf object:")
print(RunStatsDf)
print("Saving as Excel")
RunStatsDf.to_excel(fname + '.xlsx', index=False)



# -----------------------------------------------------------------------------------------
# This section generates accuracy, convergence and efficiency plots
# -----------------------------------------------------------------------------------------

df = pd.read_excel('population_density_imex_stats' + '.xlsx') # excel file

diff_coeff = {'k0':0.0,'k2':0.02, 'k4':0.04} #diffusion coefficients

adapt_accuracy         = True
adapt_efficiency_time  = True
adapt_efficiency_steps = True
fixed_convergence      = True
fixed_efficiency_work  = True
fixed_efficiency_time  = True

for kname, kval in diff_coeff.items():    
# ------------ adaptive run ---------------  
    data_adaptive = df[(df["diff_coef"] == kval) & (df["Runtype"] == "adaptive")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
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
        plt.savefig(f"popu_adaptive_accuracy_{kname}.pdf")
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
            plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig(f"popu_adaptive_efficiency_time_{kname}.pdf")
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
            plt.plot(sspness['Steps'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('number of steps')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig(f"popu_adaptive_efficiency_steps_{kname}.pdf")
        plt.show()

# --------------- fixed run ----------------            
    data_fixed = df[(df["diff_coef"] == kval) & (df["Runtype"] == "fixed")][["Runtype", "IMEX_method", "diff_coef", "runVal", "Nonlinear_Solves", "Explicit_RHS", 
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
            plt.plot(sspness['runVal'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('h')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig(f"popu_fixed_convergence_{kname}.pdf")
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
            plt.plot(sspness['runtime'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('runtime')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig(f"popu_fixed_efficiency_time_{kname}.pdf")
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
            plt.plot(sspness['Total Func Eval'], sspness['error'], marker='x', linewidth=2, linestyle='none', color=method_line_color)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Total Num of Func Evals')
        plt.ylabel('$L_{\\infty}$ error')
        plt.legend()
        plt.savefig(f"popu_fixed_efficiency_work_{kname}.pdf")
        plt.show()



