#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere and Daniel R. Reynolds @ SMU
# --------------------------------------------------------------------------------------------------------------------------------
# SUNDIALS Copyright Start
# Copyright (c) 2002-2024, Lawrence Livermore National Security
# and Southern Methodist University.
# All rights reserved.
#
# See the top-level LICENSE and NOTICE files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
# SUNDIALS Copyright End
#----------------------------------------------------------------------------------------------------------------------------------
# ReadME: This script generates the L-infinity norm and plots at 3 time steps for a particular method.
#         Run the "runtest_referenceSolution.py" script to generate the reference solutions to be used in this script.
#-----------------------------------------------------------------------------------------------------------------------------------

# imports
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# data file name
datafile = "population.txt"

# return with an error if the file does not exist
if not os.path.isfile(datafile):
    msg = "Error: file " + datafile + " does not exist"
    sys.exit(msg)

# read solution file, storing each line as a string in a list
with open(datafile, "r") as file:
    lines = file.readlines()

    # extract header information
    title  = lines.pop(0)  # Title
    T0     = float(lines.pop(0).split()[2])   # initial time
    Tf     = float(lines.pop(0).split()[2])   # final time
    N      = int(lines.pop(0).split()[2])     # spatial dimension
    xl     = float(lines.pop(0).split()[2])   # left endpoint on spatial grid
    xr     = float(lines.pop(0).split()[2])   # right endpoint on spatial grid 
    diff_k = float(lines.pop(0).split()[2])   # diffusion coefficient 
    x      = np.linspace(xl, xr, N)

    lastline  = (lines[-1])
    num_steps = lastline.split(':')
    nsteps    = int(num_steps[1].strip()) # total number of steps taken

    dt = (Tf-T0)/nsteps                   # temporal step size
    dx = (xr - xl)/N                      # spatial step size
    
    # allocate solution data as 2D Python arrays
    t    = np.zeros((nsteps), dtype=float)
    pSol = np.zeros((nsteps, N), dtype=float)

    # store remaining data into numpy arrays
    it  = 0
    for i in range(0, len(lines)):
        if "Time step" in lines[i]:
            get_t  = lines[i].split(':')
            time_t = get_t[1].strip()
            i = i + 1
            pSol[it,:] = np.array(list(map(float, lines[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t[it] = time_t #(it + 1) * dt
            it = it + 1

    ## Extract vector solution at the last time step
    pSol_lastStep = np.zeros((N), dtype=float)
    for i in range(len(pSol[nsteps-1, :])):
        pSol_lastStep[i] = pSol[nsteps-1, i] 

    ## Use First Derivative to Determine Smoothness of Graph 
    pSol_x = np.zeros((N), dtype=float)
    for i in range(len(pSol[nsteps-1,:])):
        pSol_x[i] = pSol[nsteps-1,i]

    lmax_pSol_xi = 0.0
    for i in range(len(pSol_x-1)-1):
        pSol_xi = (pSol_x[i+1] - pSol_x[i])/dx
        if pSol_xi > lmax_pSol_xi:
            lmax_pSol_xi = pSol_xi
    print("lmax of first derivative at final time step: %.6f" %lmax_pSol_xi)
   

    ## plot defaults: increase default font size, increase plot width, enable LaTeX rendering
    plt.rc("font", size=15)
    plt.rcParams["figure.figsize"] = [7.2, 4.8]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.constrained_layout.use"] = True

    ## subplots with time snapshots of the density, x-velocity, and pressure
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 3, figure=fig)
    ax00 = fig.add_subplot(gs[0, 0])  # left column
    ax01 = fig.add_subplot(gs[0, 1])  # middle column
    ax02 = fig.add_subplot(gs[0, 2])  # right column
    it = 0
    tval = repr(float(t[it])).zfill(3)
    ax00.plot(x, pSol[it, :], "-b",)
    ax00.set_title(r"$t =$ " + tval)
    ax00.set_ylabel(r"$P(t,x)$")
    ax00.set_xlabel(r"$x$")
    ax00.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    middleval = int(np.ceil(nsteps/2))
    it = middleval
    tval = repr(float(t[it])).zfill(3)
    ax01.plot(x, pSol[it, :], "-b")
    ax01.set_title(r"$t =$ " + tval)
    ax01.set_xlabel(r"$x$")
    ax01.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    it = nsteps - 1
    tval = repr(float(t[it])).zfill(3)
    ax02.plot(x, pSol[it, :], "-b")
    ax02.set_title(r"$t =$ " + tval)
    ax02.set_xlabel(r"$x$")
    ax02.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.savefig("populationModel_frames.png")

    plt.rc("font", size=15)
    plt.rcParams["figure.figsize"] = [7.2, 4.8]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.constrained_layout.use"] = True

    # plt.close()
    plt.show()


## ------------------ Extract Reference Solution at Final Time Step -----------------------
def read_ref_solution(filename, N):
    """
    This script extract the solution at the final time step of the reference solution 
    required to compute the error norm at the final time step.

    Input: filename: reference solution filename
           N : the length of the solution at the final time step

    Output: returns the solution vector at the final time step
    """
    if not os.path.isfile(filename):
        msg = "Error: file " + filename + " does not exist"
        sys.exit(msg)
    
    # read solution file, storing each line as a string in a list
    with open(filename, "r") as file_ref:
        lines_ref = file_ref.readlines()

        lastline_ref  = (lines_ref[-1])
        num_steps_ref = lastline_ref.split(':')
        nsteps_ref    = int(num_steps_ref[1].strip()) # total number of steps taken
        
        ## allocate solution data as 2D Python arrays
        t_ref    = np.zeros((nsteps_ref), dtype=float)
        pSol_ref = np.zeros((nsteps_ref, N), dtype=float)

        ## store remaining data into numpy arrays
        it  = 0
        for i in range(0, len(lines_ref)):
            if "Time step" in lines_ref[i]:
                get_t_ref  = lines_ref[i].split(':')
                time_t_ref = get_t_ref[1].strip()
                i = i + 1
                pSol_ref[it,:] = np.array(list(map(float, lines_ref[i].split()))) #to remove single quotes around the vectors since each vector is a line
                t_ref[it] = time_t_ref #(it + 1) * dt
                it = it + 1

        pSol_ref_lastStep = np.zeros((N), dtype=float)
        for i in range(len(pSol_ref[nsteps_ref-1, :])):
            pSol_ref_lastStep[i] = pSol_ref[nsteps_ref-1, i]
    return pSol_ref_lastStep

# fixed runs
fixedk0_refSoln_lastStep = read_ref_solution("fixed_referenceSoln_k0.txt", N)
fixedk2_refSoln_lastStep = read_ref_solution("fixed_referenceSoln_k2.txt", N)
fixedk4_refSoln_lastStep = read_ref_solution("fixed_referenceSoln_k4.txt", N)
# adaptive runs
adaptk0_refSoln_lastStep = read_ref_solution("adaptive_referenceSoln_k0.txt", N)
adaptk2_refSoln_lastStep = read_ref_solution("adaptive_referenceSoln_k2.txt", N)
adaptk4_refSoln_lastStep = read_ref_solution("adaptive_referenceSoln_k4.txt", N)


## -------------------- Compute L-infinty norm using the reference solution -----------------------
AdaptiveRun = True
FixedRun    = True
elmax       = 0.0 #l-infinity error
if (FixedRun):
    if (diff_k==0.0):
        for i in range(N):
            errV = np.abs(fixedk0_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.02):
        for i in range(N):
            errV = np.abs(fixedk2_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.04):
        for i in range(N):
            errV = np.abs(fixedk4_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
if (AdaptiveRun):
    if (diff_k==0.0):
        for i in range(N):
            errV = np.abs(adaptk0_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.02):
        for i in range(N):
            errV = np.abs(adaptk2_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.04):
        for i in range(N):
            errV = np.abs(adaptk4_refSoln_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end

print("Lmax error using reference solution: %.6e" %elmax)
# end if statement

##### end of script #####
