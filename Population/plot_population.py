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
# ReadME: If running fixed step sizes, ensure that fixedRun = True (1 location) and fixhRun = True (1 location) 
#         in the script,runtests_population_density_imex.py Also, ensure that FixedRun = True (1 location) in this script,
#         plot_population.py . This means that adaptiveRun = False (1 location) and adaptRun = False (1 location) in 
#         the script, runtests_population_density_imex.py and, AdaptiveRun = False (1 location) in this script plot_population.py 
#
#         Similarly, if running adaptive step sizes, ensure that adaptiveRun = True (1 location) and
#         adaptRun = True (1 location) in the script, runtests_population_density_imex.py 
#         Also, ensure that AdaptiveRun = True (1 location) in the script, plot_population.py . 
#         This means that fixedRun = False (1 location) and fixhRun = False (1 location) in the script, 
#         runtests_population_density_imex.py and, FixedRun = False (1 location) in this script plot_population.py 
#
#         Ensure that the reference solutions are generated and stored in the textfiles:
#         refSoln_k0pt02_h0pt01_ssp423.txt and refSoln_k0pt04_h0pt01_ssp423.txt for fixed temporal step sizes for k=0.02 and k=0.04,
#         respectively and, refSoln_k0pt02_rtol1en8_ssp423.txt and refSoln_k0pt04_rtol1en8_ssp423.txt for adaptive step sizes, for 
#         k=0.02 and k=0.04, respectively.
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

    plt.close()


## ------------------ Fixed Step Size Reference Solution for k = 0.02 using SSP 423 -----------------------
# data file name for reference solution 
datafile_refkpt02 = "refSoln_k0pt02_h0pt01_ssp423.txt"
if not os.path.isfile(datafile_refkpt02):
    msg = "Error: file " + datafile_refkpt02 + " does not exist"
    sys.exit(msg)

# read solution file, storing each line as a string in a list
with open(datafile_refkpt02, "r") as file_refkpt02:
    lines_refkpt02 = file_refkpt02.readlines()

    lastline_refkpt02  = (lines_refkpt02[-1])
    num_steps_refkpt02 = lastline_refkpt02.split(':')
    nsteps_refkpt02    = int(num_steps_refkpt02[1].strip()) # total number of steps taken
    
    ## allocate solution data as 2D Python arrays
    t_refkpt02    = np.zeros((nsteps_refkpt02), dtype=float)
    pSol_refkpt02 = np.zeros((nsteps_refkpt02, N), dtype=float)

    ## store remaining data into numpy arrays
    it  = 0
    for i in range(0, len(lines_refkpt02)):
        if "Time step" in lines_refkpt02[i]:
            get_t_refkpt02  = lines_refkpt02[i].split(':')
            time_t_refkpt02 = get_t_refkpt02[1].strip()
            i = i + 1
            pSol_refkpt02[it,:] = np.array(list(map(float, lines_refkpt02[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t_refkpt02[it] = time_t_refkpt02 #(it + 1) * dt
            it = it + 1

    pSol_refkpt02_lastStep = np.zeros((N), dtype=float)
    for i in range(len(pSol_refkpt02[nsteps_refkpt02-1, :])):
        pSol_refkpt02_lastStep[i] = pSol_refkpt02[nsteps_refkpt02-1, i]



## ----------------------- Fixed Step Size Reference Solution for k = 0.04 using SSP 423 ------------------------
# data file name for reference solution 
datafile_refkpt04 = "refSoln_k0pt04_h0pt01_ssp423.txt"
if not os.path.isfile(datafile_refkpt04):
    msg = "Error: file " + datafile_refkpt04 + " does not exist"
    sys.exit(msg)

# read solution file, storing each line as a string in a list
with open(datafile_refkpt04, "r") as file_refkpt04:
    lines_refkpt04 = file_refkpt04.readlines() 

    lastline_refkpt04  = (lines_refkpt04[-1])
    num_steps_refkpt04 = lastline_refkpt04.split(':')
    nsteps_refkpt04    = int(num_steps_refkpt04[1].strip()) # total number of steps taken
    
    ## allocate solution data as 2D Python arrays
    t_refkpt04    = np.zeros((nsteps_refkpt04), dtype=float)
    pSol_refkpt04 = np.zeros((nsteps_refkpt04, N), dtype=float)

    ## store remaining data into numpy arrays
    it  = 0
    for i in range(0, len(lines_refkpt04)):
        if "Time step" in lines_refkpt04[i]:
            get_t_refkpt04  = lines_refkpt04[i].split(':')
            time_t_refkpt04 = get_t_refkpt04[1].strip()
            i = i + 1
            pSol_refkpt04[it,:] = np.array(list(map(float, lines_refkpt04[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t_refkpt04[it] = time_t_refkpt04 #(it + 1) * dt
            it = it + 1

    pSol_refkpt04_lastStep = np.zeros((N), dtype=float)
    for i in range(len(pSol_refkpt04[nsteps_refkpt04-1, :])):
        pSol_refkpt04_lastStep[i] = pSol_refkpt04[nsteps_refkpt04-1, i]


## ------------------ Adaptive Step Size Reference Solution for k = 0.02 using SSP 423 -----------------------
# data file name for reference solution 
datafile_adtkpt02 = "refSoln_k0pt02_rtol1en8_ssp423.txt"
if not os.path.isfile(datafile_adtkpt02):
    msg = "Error: file " + datafile_adtkpt02 + " does not exist"
    sys.exit(msg)

# read solution file, storing each line as a string in a list
with open(datafile_adtkpt02, "r") as file_adtkpt02:
    lines_adtkpt02 = file_adtkpt02.readlines()

    lastline_adtkpt02  = (lines_adtkpt02[-1])
    num_steps_adtkpt02 = lastline_adtkpt02.split(':')
    nsteps_adtkpt02    = int(num_steps_adtkpt02[1].strip()) # total number of steps taken
    
    ## allocate solution data as 2D Python arrays
    t_adtkpt02    = np.zeros((nsteps_adtkpt02), dtype=float)
    pSol_adtkpt02 = np.zeros((nsteps_adtkpt02, N), dtype=float)

    ## store remaining data into numpy arrays
    it  = 0
    for i in range(0, len(lines_adtkpt02)):
        if "Time step" in lines_adtkpt02[i]:
            get_t_adtkpt02  = lines_adtkpt02[i].split(':')
            time_t_adtkpt02 = get_t_adtkpt02[1].strip()
            i = i + 1
            pSol_adtkpt02[it,:] = np.array(list(map(float, lines_adtkpt02[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t_adtkpt02[it] = time_t_adtkpt02 #(it + 1) * dt
            it = it + 1

    pSol_adtkpt02_lastStep = np.zeros((N), dtype=float)
    for i in range(len(pSol_adtkpt02[nsteps_adtkpt02-1, :])):
        pSol_adtkpt02_lastStep[i] = pSol_adtkpt02[nsteps_adtkpt02-1, i]


## ------------- Adaptive Step Size Reference Solution for k = 0.04 using SSP 423 ------------
# data file name for reference solution 
datafile_adtkpt04 = "refSoln_k0pt04_rtol1en8_ssp423.txt"
if not os.path.isfile(datafile_adtkpt04):
    msg = "Error: file " + datafile_adtkpt04 + " does not exist"
    sys.exit(msg)

# read solution file, storing each line as a string in a list
with open(datafile_adtkpt04, "r") as file_adtkpt04:
    lines_adtkpt04 = file_adtkpt04.readlines() 

    lastline_adtkpt04  = (lines_adtkpt04[-1])
    num_steps_adtkpt04 = lastline_adtkpt04.split(':')
    nsteps_adtkpt04    = int(num_steps_adtkpt04[1].strip()) # total number of steps taken
    
    ## allocate solution data as 2D Python arrays
    t_adtkpt04    = np.zeros((nsteps_adtkpt04), dtype=float)
    pSol_adtkpt04 = np.zeros((nsteps_adtkpt04, N), dtype=float)

    ## store remaining data into numpy arrays
    it  = 0
    for i in range(0, len(lines_adtkpt04)):
        if "Time step" in lines_adtkpt04[i]:
            get_t_adtkpt04  = lines_adtkpt04[i].split(':')
            time_t_adtkpt04 = get_t_adtkpt04[1].strip()
            i = i + 1
            pSol_adtkpt04[it,:] = np.array(list(map(float, lines_adtkpt04[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t_adtkpt04[it] = time_t_adtkpt04 #(it + 1) * dt
            it = it + 1

    pSol_adtkpt04_lastStep = np.zeros((N), dtype=float)
    for i in range(len(pSol_adtkpt04[nsteps_adtkpt04-1, :])):
        pSol_adtkpt04_lastStep[i] = pSol_adtkpt04[nsteps_adtkpt04-1, i]


## find the lmax error if k = 0.02 or k = 0.04 using the reference solution
AdaptiveRun = True
FixedRun    = True
elmax       = 0.0 #l-infinity error
if (FixedRun):
    if (diff_k==0.02):
        for i in range(N):
            errV = np.abs(pSol_refkpt02_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.04):
        for i in range(N):
            errV = np.abs(pSol_refkpt04_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
if (AdaptiveRun):
    if (diff_k==0.02):
        for i in range(N):
            errV = np.abs(pSol_adtkpt02_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end
    elif (diff_k==0.04):
        for i in range(N):
            errV = np.abs(pSol_adtkpt04_lastStep[i] - pSol_lastStep[i])
            if (errV > elmax):
                elmax = errV
            # end
        # end

print("Lmax error using reference solution: %.6e" %elmax)
# end if statement

##### end of script #####
