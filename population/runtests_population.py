#!/usr/bin/env python
#------------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ UMBC
#------------------------------------------------------------------------------------------------------------------------------------
# Description: Runs the population density model with a switch mode between problems.
#              It uses plot_population.py to compute D, D_ref and the error, and to save
#              the final-time plots.
#
#   MODE = "uncheckedSMOOTH"  (run first; smoothness is not checked)
#       For each method, diffusion coefficient and run type, the list values are run from the finest dt (tightest rtol)
#       upward until the first value with a negative solution at one or more time steps. That value and the last value
#       before it with a nonnegative solution at every time step form the initial bisection bracket. The bracket is
#       refined by bisection (test values rounded to four significant figures) to find the largest value with a
#       nonnegative solution at every time step and the smallest value with a negative solution.
#       Runs with a SUNDIALS failure are recorded but skipped, they never become an end of a bracket.
#       Output:
#         population_runs_with_bisection.xlsx  list values from the finest up to the upper end of the initial bracket,
#                                              plus the final bracket (midpoints that were discarded are not kept)
#         bisection_initial_brackets.xlsx      the initial bracket (two rows per case)
#         bisection_final_brackets.xlsx        the final bracket (two rows per case)
#         finalSoln_plots/<kname>/<runtype>/   final-time plot of every run in population_runs_with_bisection.xlsx,
#                                              used to choose TAU
#
#   MODE = "ssp"  (run second, after TAU is set)
#       Reads population_runs_with_bisection.xlsx and gives every run an ssp status: ssp if the solution is nonnegative
#       at every time step and |D_ref - D|/D_ref <= tau. For each case, the runs are checked from the finest value upward
#       until the first run that is not ssp; that run and the ssp run before it form the bracket, which is refined by
#       bisection. The final bracket (largest ssp value, smallest non-ssp value) is added to the file.
#       Output:
#         population_ssp_runs_with_bisection_fixed.xlsx     fixed runs with their ssp status, plus the final ssp bracket
#         population_ssp_runs_with_bisection_adaptive.xlsx  adaptive runs with their ssp status, plus the final ssp bracket
#         bisection_ssp_final_brackets.xlsx         the final ssp bracket (two rows per case)
#         <x>_error_population_<kname>.png      efficiency plots: for each case, the ssp runs from the finest value
#                                                   up to (not including) the first non-ssp run
#-------------------------------------------------------------------------------------------------------------------------------------

# imports
import pandas as pd
import subprocess
import time
import shlex
import shutil
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor

## ===================================================================================================================================
MODE = "uncheckedSMOOTH"        # "uncheckedSMOOTH" (run first to determine TAU values) or "ssp" (run second, after setting TAU)

# smoothness tolerance: the final profile is smooth if |D_ref - D| / D_ref <= tau
# set these after inspecting the final-time plots by running uncheckedSMOOTH
TAU = {0.0: 0.0015, 0.02: 0.015, 0.04: 0.015}

PLOT_DIR           = "finalSoln_plots"
PLOT_TMP_DIR       = os.path.join(PLOT_DIR, "_tmp_midpoints")   # plots of bisection midpoints until it is known which are kept
PLOT_SCRIPT        = "plot_population.py"
FINAL_TIME         = 10.0
MAX_SUNDIALS_FAILS = 20       # largest number of bisection test values with a SUNDIALS failure allowed per bisection bracket

## ===================================================================================================================================

# to label plots with the run value (rtol or fixed_h) in scientific notation with four significant figures
def fmt_val(v):
    """ value as it appears in the plot file names, e.g. 1.0000e-03 """
    return f"{v:.4e}"


# utility routine to run a test, storing the run options and solver statistics
def runtest(solver, modetype, runV, kVal, kName, plotfile=None):
    """
    Runs the population model using either fixed or adaptive time stepping and returns the stats.

    Input: solver:            imex scheme to run
           modetype (string): adaptive or fixed time stepping
           runV:              rtol (adaptive) or fixed_h (fixed)
           kVal:              diffusion coefficient
           kName:             diffusion coefficient name
           plotfile:          file name for the final-time plot (None: no plot)

    Output: returns the statistics
    """
    stats = {'Runtype': modetype, 'ReturnCode': 0, 'IMEX_method': solver['name'], 'diff_coef': kVal,
             'runVal': runV, 'FinalTime': 0.0,
             'Steps': 0, 'StepAttempts': 0, 'ErrTestFails': 0, 'Explicit_RHS': 0,
             'Implicit_RHS': 0, 'Implicit_solves': 0, 'Negative_model': 0, 'Negative_stages': 0,
             'lmax_1dev': 0.0, 'lmax_1dev_ref': 0.0, 'rel_diff_1dev': np.nan,
             'error': 0.0, 'runtime': 0.0, 'avg_dt': 0.0}

    if (modetype == "adaptive"):
        runcommand = " %s   --rtol %e   --k %e" % (solver['exe'], runV, kVal)
    elif (modetype == "fixed"):
        runcommand = " %s   --fixed_h %e   --k %e" % (solver['exe'], runV, kVal)

    # remove any old output so a failed run cannot reuse it
    if os.path.exists("population.txt"):
        os.remove("population.txt")

    start_time = time.time()
    result = subprocess.run(shlex.split(runcommand), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    end_time = time.time()
    stats['ReturnCode'] = result.returncode
    stats['runtime']    = end_time - start_time

    stdout_lines = result.stdout.decode('utf-8').splitlines()
    stderr_lines = result.stderr.decode('utf-8').splitlines()

    # SUNDIALS failure: "SUNDIALS_ERROR", "test failed repeatedly" or a nonzero return code
    sundials_failed = (result.returncode != 0)
    for line in stderr_lines:
        if ("SUNDIALS_ERROR" in line) or ("test failed repeatedly" in line) or ("Solver failure" in line):
            sundials_failed = True

    if sundials_failed:
        if (modetype == "adaptive"):
            print("SUNDIALS failed for  %s  --rtol %e --k %e" % (solver['exe'], runV, kVal))
        elif (modetype == "fixed"):
            print("SUNDIALS failed for %s  --fixed_h %e --k %e" % (solver['exe'], runV, kVal))
        stats['ReturnCode']   = 1
        stats['runtime']      = 0
        return stats

    print("Running: " + runcommand + " SUCCESS")
    for line in stdout_lines:
        txt = line.split()
        if ("Steps" in txt):
            stats['Steps'] = int(txt[2])
        elif (("Step" in txt) and ("attempts" in txt)):
            stats['StepAttempts'] = int(txt[3])
        elif (("Error" in txt) and ("fails" in txt)):
            stats['ErrTestFails'] = float(txt[4])
        elif (("Explicit" in txt) and ("RHS" in txt)):
            stats['Explicit_RHS'] = int(txt[5])
        elif (("Implicit" in txt) and ("RHS" in txt)):
            stats['Implicit_RHS'] = int(txt[5])

    # number of time steps at which the model has negative values
    sum_negLines = 0
    for line in stdout_lines:
        if ("Model has a negative" in line):
            sum_negLines += 1
    stats['Negative_model'] = sum_negLines
    if stats['StepAttempts'] > 0:
        stats['avg_dt'] = (FINAL_TIME - 0.0) / stats['StepAttempts']

    # number of stages at which the model has negative values
    sum_negStages = 0
    for line in stdout_lines:
        if ("Stage has a negative" in line):
            sum_negStages += 1
    stats['Negative_stages'] = sum_negStages

    # number of implicit solves for each method
    solves_per_step = {'SSP212': 2, 'SSP312': 3, 'SSPL312': 3, 'SSP423': 3, 'SSP923': 4}
    stats['Implicit_solves'] = solves_per_step.get(solver['name'], 0) * stats['StepAttempts']

    # D, D_ref, error and final-time plot
    if plotfile is not None:
        os.makedirs(os.path.dirname(plotfile), exist_ok=True)
    label = f"{solver['name']}  {kName}  {modetype}  {'rtol' if modetype == 'adaptive' else 'dt'} = {fmt_val(runV)}"
    plotcommand = [sys.executable, PLOT_SCRIPT, plotfile if plotfile else "none",
                   str(stats['Negative_model']), label]
    plot_result = subprocess.run(plotcommand, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    plot_lines = plot_result.stdout.decode('utf-8').splitlines()
    if plot_result.returncode != 0:
        stats['ReturnCode'] = 1
        return stats

    for line in plot_lines:
        if line.startswith("last time read"):
            stats['FinalTime'] = float(line.split('=')[-1].strip())
        elif line.startswith("Lmax of first derivative"):
            stats['lmax_1dev'] = float(line.split('=')[-1].strip())
        elif line.startswith("Lmax error using reference"):
            stats['error'] = float(line.split('=')[-1].strip())
        elif line.startswith("Reference first-derivative max"):
            stats['lmax_1dev_ref'] = float(line.split('=')[-1].strip())
        elif line.startswith("Relative difference of first derivative"):
            stats['rel_diff_1dev'] = float(line.split('=')[-1].strip())   # |D_ref - D| / D_ref

    # the run must reach the final time (in population.cpp the time loop only stops early if there is a SUNDIALS failure)
    if abs(stats['FinalTime'] - FINAL_TIME) > 1e-8:
        stats['ReturnCode']   = 1
        return stats

    # ignore errors greater than 10
    if stats['error'] > 10.0:
        stats['ReturnCode'] = 1

    return stats
## end of function


## -------------------------------------------------------------------------------------------------------------
## pass/fail tests: "pass", "fail" or "excluded" (SUNDIALS failure)
## -------------------------------------------------------------------------------------------------------------
def classify_positivity(stats):
    if abs(stats['FinalTime'] - FINAL_TIME) > 1e-8:     # run did not reach the final time (SUNDIALS failure)
        return "excluded"
    return "pass" if (stats['Negative_model'] == 0 and stats['Negative_stages'] == 0) else "fail"

def classify_ssp(stats):
    if abs(stats['FinalTime'] - FINAL_TIME) > 1e-8:     # run did not reach the final time (SUNDIALS failure)
        return "excluded"
    tau = TAU[stats['diff_coef']]
    return "pass" if (stats['Negative_model'] == 0 and stats['Negative_stages'] == 0 and stats['rel_diff_1dev'] <= tau) else "fail"


# shortcuts to executable/configuration of different embedded IMEX SSP methods
SSP212       = "./population   --dirk_table ARKODE_SSP_SDIRK_2_1_2        --erk_table ARKODE_SSP_ERK_2_1_2  "
SSP312       = "./population   --dirk_table ARKODE_SSP_DIRK_3_1_2         --erk_table ARKODE_SSP_ERK_3_1_2  "
SSPL312      = "./population   --dirk_table ARKODE_SSP_LSPUM_SDIRK_3_1_2  --erk_table ARKODE_SSP_LSPUM_ERK_3_1_2  "
SSP423       = "./population   --dirk_table ARKODE_SSP_ESDIRK_4_2_3       --erk_table ARKODE_SSP_ERK_4_2_3  "
SSP923       = "./population   --dirk_table ARKODE_SSP_ESDIRK_9_2_3       --erk_table ARKODE_SSP_ERK_9_2_3  "

# search lists (sorted from smallest to largest), same as runtests_population.py
adaptive_params = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0, 2.0, 3.0, 4.0, 5.0] #relative tolerances
fixed_params    = [0.25 * (2 ** i) for i in range(-11, 6)] #fixed step sizes
params          = {'fixed': fixed_params, 'adaptive': adaptive_params}

## Diffusion coefficients
diff_coef = {'diffk0': 0.0, 'diffk02': 0.02, 'diffk04': 0.04}

solvertype = [{'name': 'SSP212',       'exe': SSP212},
              {'name': 'SSP312',       'exe': SSP312},
              {'name': 'SSPL312',      'exe': SSPL312},
              {'name': 'SSP423',       'exe': SSP423},
              {'name': 'SSP923',       'exe': SSP923}]


solver_by_name = {s['name']: s for s in solvertype}
kname_of = {v: k for k, v in diff_coef.items()}

# Rounds a number to the given number of significant figures.
def round_to_sf(x, sf):
    """ rounds x to sf significant figures """
    return round(x, -int(floor(log10(abs(x)))) + (sf - 1))

# Builds the file path for the final-time plot of a run from the method, diffusion coefficient, run type and step size or tolerance.
def plot_name(kName, runtype, method, runV, folder=None):
    folder = folder if folder else os.path.join(PLOT_DIR, kName, runtype)
    return os.path.join(folder, f"finalSoln_{kName}_{runtype}_{method}_{fmt_val(runV)}.png")

# Picks the next bisection value as the midpoint, rounded to four significant figures, skipping values where SUNDIALS failed.
def next_test_value(lowVal, highVal, failed):
    """ midpoint (four significant figures) of the lowest gap between lowVal, the SUNDIALS-failed values and highVal """
    points = [lowVal] + sorted(v for v in failed if lowVal < v < highVal) + [highVal]
    for a, b in zip(points[:-1], points[1:]):
        mid = round_to_sf((a + b) / 2.0, 4)
        if a < mid < b and mid not in failed:
            return mid
    return None

# Bisects between a passing run and a failing run until no new four-significant-figure value is left, and returns the final passing and failing runs.
def bisection(low, high, classify, tmp_plots):
    """
    Bisection between the run 'low' (passes) and the run 'high' (fails). Test values are rounded to four significant
    figures. A test value with a SUNDIALS failure never becomes an end; the next test value is taken from the gap below it
    (then the gap above it). Returns the final lower and upper runs.
    tmp_plots: save midpoint plots to PLOT_TMP_DIR (True) or make no plot (False).
    """
    solver = solver_by_name[low['IMEX_method']]
    runtype, kVal = low['Runtype'], low['diff_coef']
    kName = kname_of[kVal]
    failed = []
    while True:
        midVal = next_test_value(low['runVal'], high['runVal'], failed)
        if midVal is None or len(failed) >= MAX_SUNDIALS_FAILS:
            break
        pfile = plot_name(kName, runtype, solver['name'], midVal, PLOT_TMP_DIR) if tmp_plots else None
        stats = runtest(solver, runtype, midVal, kVal, kName, plotfile=pfile)
        c = classify(stats)
        if c == "pass":
            low = stats
        elif c == "fail":
            high = stats
        else:
            failed.append(midVal)
    return low, high

# Labels the two ends of a bracket so they can be saved as rows in the bracket Excel files.
def bracket_rows(low, high, label_low, label_high):
    rows = []
    for r, lab in [(low, label_low), (high, label_high)]:
        if r is not None:
            r = dict(r)
            r['bracket_end'] = lab
            rows.append(r)
    return rows


## ===================================================================================================================================
## MODE = "uncheckedSMOOTH": this is the first run, to determine TAU values for each diffusion coefficient, it produces the final time plots
## ===================================================================================================================================
if MODE == "uncheckedSMOOTH":

    all_rows, initial_rows, final_rows = [], [], []
    for k_name, k_val in diff_coef.items():
        for runtype in ['fixed', 'adaptive']:
            for solver in solvertype:
                case_rows = []
                low, high = None, None

                # list values from the finest upward, stop at the first negative solution
                for runV in params[runtype]:
                    stats = runtest(solver, runtype, runV, k_val, k_name,
                                    plotfile=plot_name(k_name, runtype, solver['name'], runV))
                    case_rows.append(stats)
                    c = classify_positivity(stats)
                    if c == "pass":
                        low = stats
                    elif c == "fail":
                        high = stats
                        break

                initial_rows += bracket_rows(low, high, "largest_nonnegative", "smallest_negative")

                # bisection
                if low is not None and high is not None:
                    low, high = bisection(low, high, classify_positivity, tmp_plots=True)
                    # keep the final bracket ends that are new runs (bisection midpoints), with their plots
                    for r in (low, high):
                        if r['runVal'] not in params[runtype]:
                            tmp  = plot_name(k_name, runtype, solver['name'], r['runVal'], PLOT_TMP_DIR)
                            keep = plot_name(k_name, runtype, solver['name'], r['runVal'])
                            if os.path.exists(tmp):
                                os.makedirs(os.path.dirname(keep), exist_ok=True)
                                shutil.move(tmp, keep)
                            case_rows.append(r)

                final_rows += bracket_rows(low, high, "largest_nonnegative", "smallest_negative")
                all_rows += sorted(case_rows, key=lambda r: r['runVal'])

    pd.DataFrame.from_records(all_rows).to_excel("population_runs_with_bisection.xlsx", index=False)
    pd.DataFrame.from_records(initial_rows).to_excel("bisection_initial_brackets.xlsx", index=False)
    pd.DataFrame.from_records(final_rows).to_excel("bisection_final_brackets.xlsx", index=False)
    shutil.rmtree(PLOT_TMP_DIR, ignore_errors=True)       # plots of discarded midpoints


## ===================================================================================================================================
## MODE = "ssp"
## ===================================================================================================================================
elif MODE == "ssp":

    runs = pd.read_excel("population_runs_with_bisection.xlsx").to_dict('records')
    status_name = {"pass": "ssp", "fail": "not ssp", "excluded": "sundials_failed"}
    for r in runs:
        r['ssp_status'] = status_name[classify_ssp(r)]

    out_rows, final_rows = [], []
    for k_name, k_val in diff_coef.items():
        for runtype in ['fixed', 'adaptive']:
            for solver in solvertype:
                case_rows = sorted([r for r in runs if r['IMEX_method'] == solver['name'] and r['diff_coef'] == k_val
                                    and r['Runtype'] == runtype], key=lambda r: r['runVal'])
                low, high = None, None
                for r in case_rows:
                    if r['ssp_status'] == "ssp":
                        low = r
                    elif r['ssp_status'] == "not ssp":
                        high = r
                        break
                if low is not None and high is not None:
                    low, high = bisection(low, high, classify_ssp, tmp_plots=False)
                    for r in (low, high):
                        if r['runVal'] not in [c['runVal'] for c in case_rows]:
                            r['ssp_status'] = status_name[classify_ssp(r)]
                            case_rows.append(r)

                final_rows += bracket_rows(low, high, "largest_ssp", "smallest_not_ssp")
                out_rows += sorted(case_rows, key=lambda r: r['runVal'])

    out_df = pd.DataFrame.from_records(out_rows)
    out_df[out_df['Runtype'] == 'fixed'].to_excel("population_ssp_runs_with_bisection_fixed.xlsx", index=False)
    out_df[out_df['Runtype'] == 'adaptive'].to_excel("population_ssp_runs_with_bisection_adaptive.xlsx", index=False)
    pd.DataFrame.from_records(final_rows).to_excel("bisection_ssp_final_brackets.xlsx", index=False)

    # ===============================================================================================================================
    #  Efficiency plots: for each method, the ssp runs from the finest value up to (not including) the first non-ssp run
    # ===============================================================================================================================
    df_fixed    = pd.read_excel('population_ssp_runs_with_bisection_fixed.xlsx')
    df_adaptive = pd.read_excel('population_ssp_runs_with_bisection_adaptive.xlsx')

    def ssp_runs_to_plot(method_data):
        rows = []
        for _, r in method_data.sort_values('runVal').iterrows():
            if r['ssp_status'] == "sundials_failed":
                continue
            if r['ssp_status'] != "ssp":
                break
            rows.append(r)
        return pd.DataFrame(rows)

    colors    = ['red', 'black', 'blue', 'green', 'orange']
    x_metrics = [('StepAttempts', 'step-attempts', 'step_attempts'),
                 ('Implicit_solves', 'implicit-solves', 'implicit_solves'),
                 ('runtime', 'runtime', 'runtime')]
    y_metrics = [('error', 'error', 'error')]

    for k_name, k_val in diff_coef.items():
        for x_metric, x_label, x_filename in x_metrics:
            for y_metric, y_label, y_filename in y_metrics:
                fig, ax = plt.subplots(figsize=(7, 6))

                data_fixed    = df_fixed[(df_fixed["diff_coef"] == k_val)]
                data_adaptive = df_adaptive[(df_adaptive["diff_coef"] == k_val)]

                # fixed run
                for i, SSPmethodFix in enumerate(data_fixed['IMEX_method'].unique()):
                    valid_data = ssp_runs_to_plot(data_fixed[data_fixed['IMEX_method'] == SSPmethodFix])
                    if len(valid_data) == 0:
                        continue
                    valid_data = valid_data[valid_data['ReturnCode'] != 1]
                    ax.plot(valid_data[x_metric], valid_data[y_metric], color=colors[i], marker='o', markersize=5,
                            linestyle='-', linewidth=2, label=f"{SSPmethodFix}-h")

                # adaptive run
                for i, SSPmethodAdapt in enumerate(data_adaptive['IMEX_method'].unique()):
                    valid_data = ssp_runs_to_plot(data_adaptive[data_adaptive['IMEX_method'] == SSPmethodAdapt])
                    if len(valid_data) == 0:
                        continue
                    valid_data = valid_data[valid_data['ReturnCode'] != 1]
                    ax.plot(valid_data[x_metric], valid_data[y_metric], color=colors[i], marker='*', markersize=5,
                            linestyle='-.', linewidth=2, label=f"{SSPmethodAdapt}-rtol")

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.tick_params(axis='both', labelsize=20)

                # remove duplicates
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), borderaxespad=0., loc='upper left', fontsize=20)

                ax.set_xlabel(f'{x_label}', fontsize=20)
                ax.set_ylabel(f'{y_label} (d={k_val})', fontsize=20)
                if k_val == 0.0:
                    ax.set_ylim(bottom=5e-10)
                plt.savefig(f"{x_filename}_{y_filename}_population_{k_name}.png", bbox_inches="tight")
                plt.close(fig)

else:
    sys.exit("Error: MODE must be \"uncheckedSMOOTH\" or \"ssp\".")
