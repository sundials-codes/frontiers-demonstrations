#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Programmer(s): Sylvia Amihere @ SMU
# ------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2024, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
# ------------------------------------------------------------------------------
# matplotlib-based plotting script for the serial linear advection example
# ------------------------------------------------------------------------------

# imports
import shutil
import subprocess
import shlex
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# data file name
datafile = "hyperbolic_relaxation.out"

# return with an error if the file does not exist
if not os.path.isfile(datafile):
    msg = "Error: file " + datafile + " does not exist"
    sys.exit(msg)

gamma = 7.0/5.0
# read solution file, storing each line as a string in a list
with open(datafile, "r") as file:
    lines = file.readlines()

    # extract header information
    title = lines.pop(0)
    nvar = int((lines.pop(0).split())[2])
    varnames = lines.pop(0)
    nt = int((lines.pop(0).split())[2])
    nx = int((lines.pop(0).split())[2])
    xl = float((lines.pop(0).split())[2])
    xr = float((lines.pop(0).split())[2])
    lastline  = (lines[-1])
    num_steps = lastline.split(':')
    nsteps    = int(num_steps[1].strip()) # total number of steps taken
    lines.pop()   # remove "Number of Time Steps Taken: 2604"

    ndata = nsteps+1
    # store solution at final time step in numpy arrays
    rho = np.zeros((nx), dtype=float)
    mx = np.zeros((nx), dtype=float)
    my = np.zeros((nx), dtype=float)
    mz = np.zeros((nx), dtype=float)
    et = np.zeros((nx), dtype=float)
    x = np.linspace(xl, xr, nx)
    dx = (xr - xl)/nx

    last_line_run = ""
    for line in lines:
        if line.strip():
            last_line_run = line

    last_line_run_data = last_line_run.split()
    t = float(last_line_run_data.pop(0))

    
    # store remaining data into numpy arrays, the first element in each array is the time step
    for ix in range(nx):
        rho[ix] = last_line_run_data.pop(0)
        mx[ix] = last_line_run_data.pop(0)
        my[ix] = last_line_run_data.pop(0)
        mz[ix] = last_line_run_data.pop(0)
        et[ix] = last_line_run_data.pop(0)

print("last time read =", t, " tf should be 0.08")
# largeDev_xgrid = [] #contains grid values were largest derivative occurs
# largeDev_time  = [] #contains the time step corresponding to the largest derivative value
# for it in range(ndata):
#     largeDev      = 0.0 #largest derivative value
#     largeDev_xloc = 0   # spatial grid location of the largest derivative
#     for ix in range(nx-1):
#         max_derv = abs(rho[it, ix+1] - rho[it, ix])/dx
#         if (max_derv > largeDev):
#             largeDev      = max_derv
#             largeDev_xloc = ix
#         timeV = it
#         #end
#     #end
#     largeDev_xgrid.append(float(x[largeDev_xloc]))
#     largeDev_time.append(float(t[timeV]))
# #end

# solution at the final time step
przdata = np.zeros((nx), dtype=float) #pressure
rhodata = np.zeros((nx), dtype=float) #density
veldata = np.zeros((nx), dtype=float) #velocity
e_eq    = 25.0 * np.ones((nx),  dtype=float) #e_{0}
etdiff  = np.zeros((nx), dtype=float) #E - E_{0}
for i in range(nx):
    przdata[i] = (gamma-1.0) * (et[i] - (mx[i] * mx[i] + my[i] * my[i] + mz[i] * mz[i]) * 0.5 / rho[i])
    rhodata[i] = rho[i]
    veldata[i] = mx[i]/rho[i]
    etdiff[i] = ( (et[i]/rho[i]) - 0.5 * ((mx[i]/rho[i])**2) ) - e_eq[i] 
# end


## ------------------ Extract Reference Solution at Final Time Step -----------------------
def read_ref_solution(filename):
    """
    This script extract the solution at the final time step of the reference solution 
    required to compute the error norm at the final time step.

    Input: filename: reference solution filename

    Output: returns the solution vector at the final time step
    """
    if not os.path.isfile(filename):
        msg = "Error: file " + filename + " does not exist"
        sys.exit(msg)
    
    # read solution file, storing each line as a string in a list
    with open(filename, "r") as file_ref:

        # extract header information
        title_ref     = file_ref.readline()
        nvar_ref      = int((file_ref.readline().split())[2])
        varnames_ref  = file_ref.readline()
        nt_ref        = int((file_ref.readline().split())[2])
        nx_ref        = int((file_ref.readline().split())[2])
        xl_ref        = float((file_ref.readline().split())[2])
        xr_ref        = float((file_ref.readline().split())[2])

        last_line = ""
        for line in file_ref:
            if "Number of Time Steps" in line:
                nsteps_ref = int(line.split(':')[1].strip()) # extract total number of steps taken
                break
            # track the last non-empty solution, every nonempty line overwrites the last line
            if line.strip():
                last_line = line

    # store only solution at the final step
    last_data = last_line.split()
    t_ref_final = float(last_data.pop(0)) #final time step

    rho_ref = np.zeros((nx_ref), dtype=float)
    mx_ref  = np.zeros((nx_ref), dtype=float)
    my_ref  = np.zeros((nx_ref), dtype=float)
    mz_ref  = np.zeros((nx_ref), dtype=float)
    et_ref  = np.zeros((nx_ref), dtype=float)
        
    for ix in range(nx_ref):
        rho_ref[ix] = float(last_data.pop(0))
        mx_ref[ix]  = float(last_data.pop(0))
        my_ref[ix]  = float(last_data.pop(0))
        mz_ref[ix]  = float(last_data.pop(0))
        et_ref[ix]  = float(last_data.pop(0))

    rhoRefFinal = np.zeros((nx_ref), dtype=float) #density
    velRefFinal = np.zeros((nx_ref), dtype=float) #velocity
    etRefFinal  = np.zeros((nx_ref), dtype=float) #energy
    e_eq        = 25.0 * np.ones((nx_ref),  dtype=float) #e_{0}
    przRefFinal = np.zeros((nx_ref), dtype=float) #pressure

    gamma = 7.0/5.0
    for i in range(nx_ref):
        rhoRefFinal[i] = rho_ref[i] 
        velRefFinal[i] = mx_ref[i]/rho_ref[i] 
        etRefFinal[i]  = ( (et_ref[i]/rho_ref[i]) - 0.5 * ((mx_ref[i]/rho_ref[i])**2) ) - e_eq[i] 
        przRefFinal[i] = (gamma-1.0) * (et_ref[i] - (mx_ref[i] * mx_ref[i] + my_ref[i] * my_ref[i] + mz_ref[i] * mz_ref[i]) * 0.5 / rho_ref[i])
    
    return rhoRefFinal, velRefFinal, etRefFinal, przRefFinal, t_ref_final


## -------------------- Compute L-infinty norm using the reference solution -----------------------
elmax_rho = 0.0 
elmax_vel = 0.0 
elmax_et = 0.0 
elmax_prz = 0.0 

refLastSoln_rho, refLastSoln_vel, refLastSoln_et, refLastSoln_prz, refLastSoln_t = read_ref_solution("hyperbolic_relaxation_reference_solution.out")
if np.abs(refLastSoln_t - 0.08) > 1e-10:
    sys.exit(f"ERROR: reference solution is at t = {refLastSoln_t}, not 0.08")
else:
    elmax_rho = np.max(np.abs(refLastSoln_rho - rhodata))
    elmax_vel = np.max(np.abs(refLastSoln_vel - veldata))
    elmax_et = np.max(np.abs(refLastSoln_et - etdiff))
    elmax_prz = np.max(np.abs(refLastSoln_prz - przdata))
    print("Lmax rho error using reference solution = %e" %elmax_rho)
    print("Lmax velocity error using reference solution = %e" %elmax_vel)
    print("Lmax energy error using reference solution = %e" %elmax_et)
    print("Lmax pressure error using reference solution = %e" %elmax_prz)


##### end of script #####


# ## -------------------- plot solution overlaid with reference solution: solution at final time step -----------------------
# x_ref = np.linspace(xl, xr, len(refLastSoln_rho))
# fig, ax = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
# # plot density
# ax[0, 0].plot(x, rhodata, linestyle='-', label='computed', color='blue', linewidth=2)
# ax[0, 0].plot(x_ref, refLastSoln_rho, linestyle='--',label='reference', color='red', linewidth=2)
# ax[0, 0].set_xlabel('x', fontsize=10)
# ax[0, 0].set_ylabel(r'Density ($\rho$)', fontsize=10)
# ax[0, 0].legend()

# # plot velocity
# ax[0, 1].plot(x, veldata, linestyle='-', label='computed', color='blue', linewidth=2)
# ax[0, 1].plot(x_ref, refLastSoln_vel, linestyle='--',label='reference', color='red', linewidth=2)
# ax[0, 1].set_xlabel('x', fontsize=10)
# ax[0, 1].set_ylabel(r'velocity ($\mu$)', fontsize=10)
# ax[0, 1].legend()

# # plot pressure
# ax[1, 0].plot(x, przdata, linestyle='-', label='computed', color='blue', linewidth=2)
# ax[1, 0].plot(x_ref, refLastSoln_prz, linestyle='--',label='reference', color='red', linewidth=2)
# ax[1, 0].set_xlabel('x', fontsize=10)
# ax[1, 0].set_ylabel(r'pressure ($p$)', fontsize=10)
# ax[1, 0].legend()

# # plot internal energy difference
# ax[1, 1].plot(x, etdiff, linestyle='-', label='computed', color='blue', linewidth=2)
# ax[1, 1].plot(x_ref, refLastSoln_et, linestyle='--',label='reference', color='red', linewidth=2)
# ax[1, 1].set_xlabel('x', fontsize=10)
# ax[1, 1].set_ylabel(r'energy ($e - e_{eq}$)', fontsize=10)
# ax[1, 1].legend()

# plt.show()
# # plt.savefig("hyperbolic_relaxation_final_time_solution.png")
# # plt.close()
