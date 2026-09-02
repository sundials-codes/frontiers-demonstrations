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
    # allocate solution data as 2D Python arrays
    t = np.zeros((ndata), dtype=float)
    rho = np.zeros((ndata, nx), dtype=float)
    mx = np.zeros((ndata, nx), dtype=float)
    my = np.zeros((ndata, nx), dtype=float)
    mz = np.zeros((ndata, nx), dtype=float)
    et = np.zeros((ndata, nx), dtype=float)
    x = np.linspace(xl, xr, nx)
    dx = (xr - xl)/nx

    # lines.pop(0) #remove the initial solution
    
    # store remaining data into numpy arrays, the first element in each array is the time step
    for it in range(ndata):
        line = (lines.pop(0)).split()
        t[it] = line.pop(0)
        for ix in range(nx):
            rho[it, ix] = line.pop(0)
            mx[it, ix] = line.pop(0)
            my[it, ix] = line.pop(0)
            mz[it, ix] = line.pop(0)
            et[it, ix] = line.pop(0)

print("last time read =", t[ndata-1], " tf should be 0.3")
largeDev_xgrid = [] #contains grid values were largest derivative occurs
largeDev_time  = [] #contains the time step corresponding to the largest derivative value
for it in range(ndata):
    largeDev      = 0.0 #largest derivative value
    largeDev_xloc = 0   # spatial grid location of the largest derivative
    for ix in range(nx-1):
        max_derv = abs(rho[it, ix+1] - rho[it, ix])/dx
        if (max_derv > largeDev):
            largeDev      = max_derv
            largeDev_xloc = ix
        timeV = it
        #end
    #end
    largeDev_xgrid.append(float(x[largeDev_xloc]))
    largeDev_time.append(float(t[timeV]))
#end

# solution at the final time step
przdata = np.zeros((nx), dtype=float) #pressure
rhodata = np.zeros((nx), dtype=float) #density
veldata = np.zeros((nx), dtype=float) #velocity
e_eq    = 25.0 * np.ones((nx),  dtype=float) #e_{0}
etdiff  = np.zeros((nx), dtype=float) #E - E_{0}
for i in range(nx):
    przdata[i] = (gamma-1.0) * (et[-1, i] - (mx[-1, i] * mx[-1, i] + my[-1, i] * my[-1, i] + mz[-1, i] * mz[-1, i]) * 0.5 / rho[-1, i])
    rhodata[i] = rho[-1, i]
    veldata[i] = mx[-1, i]/rho[-1, i]
    etdiff[i] = ( (et[-1, i]/rho[-1, i]) - 0.5 * ((mx[-1, i]/rho[-1, i])**2) ) - e_eq[i] 
# end

# determine interface location
iloc = 0
for i in range(nx):
    if x[i]>=0.5:
        iloc = i
        break

# energy error in the stiff region (E - E_{0})
etdiff_stiff  = np.zeros((len(x[iloc:])), dtype=float) 
etdiff_stiff = etdiff[iloc:]

# Lmax error for between the energy at the final time step and the equilibrium energy
energy_errMax = 0.0
for i in range(len(etdiff_stiff)):
    energy_err = np.abs(etdiff_stiff[i])
    if (energy_err > energy_errMax) :
       energy_errMax = energy_err
    # end
#end
print("Maximum energy error = %.4e" %energy_errMax)

## plot defaults: increase default font size, increase plot width, enable LaTeX rendering
# ## subplots with time snapshots of the density, x-velocity, and pressure
# fig = plt.figure(figsize=(10, 5))
# gs = GridSpec(1, 3, figure=fig)
# ax00 = fig.add_subplot(gs[0, 0])  # 1st column - initial time step
# ax01 = fig.add_subplot(gs[0, 1])  # 2nd column
# ax02 = fig.add_subplot(gs[0, 2])  # 3rd column

# it = 0
# tval = repr(float(t[it])).zfill(3)
# ax00.plot(x, rho[it, :], "-b",)
# ax00.set_title(r"$t =$ " + tval)
# ax00.set_ylabel(r"$P(t,x)$")
# ax00.set_xlabel(r"$x$")

# middleval = int(np.ceil(nsteps/2))
# it = middleval
# tval = repr(float(t[it])).zfill(3)
# ax01.plot(x, rho[it, :], "-b")
# ax01.set_title(r"$t =$ " + tval)
# ax01.set_ylabel(r"$P(t,x)$")
# ax01.set_xlabel(r"$x$")

# it = -1
# tval = repr(float(t[it])).zfill(3)
# ax02.plot(x, rho[it, :], "-b")
# ax02.set_title(r"$t =$ " + tval)
# ax02.set_ylabel(r"$P(t,x)$")
# ax02.set_xlabel(r"$x$")

# plt.rc("font", size=15)
# plt.rcParams["figure.figsize"] = [7.2, 4.8]
# plt.rcParams["text.usetex"] = True
# plt.rcParams["figure.constrained_layout.use"] = True
# # plt.savefig("hyperbolic_relaxation_frames.pdf")
# plt.close()


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
    przRefFinal = np.zeros((nx_ref), dtype=float) #pressure

    gamma = 7.0/5.0
    for i in range(nx_ref):
        rhoRefFinal[i] = rho_ref[i] 
        velRefFinal[i] = mx_ref[i]/rho_ref[i] 
        etRefFinal[i]  = et_ref[i] 
        przRefFinal[i] = (gamma-1.0) * (et_ref[i] - (mx_ref[i] * mx_ref[i] + my_ref[i] * my_ref[i] + mz_ref[i] * mz_ref[i]) * 0.5 / rho_ref[i])
    
    return rhoRefFinal, t_ref_final


## -------------------- Compute L-infinty norm using the reference solution -----------------------
elmax = 0.0 
refLastSoln_rho, refLastSoln_t = read_ref_solution("hyperbolic_relaxation_reference_solution.out")
if np.abs(refLastSoln_t - 0.3) > 1e-10:
    sys.exit(f"ERROR: reference solution is at t = {refLastSoln_t}, not 0.3")
else:
    elmax = np.max(np.abs(refLastSoln_rho - rhodata))
    print("Lmax error using reference solution = %e" %elmax)


##### end of script #####
