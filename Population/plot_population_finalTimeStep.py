#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere @ SMU
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
# ReadME: This script extracts the rtol and fixed_h for which a method is SSP and stores then a list.
#         The data is then used to plot the numerical solution at the final time step for each method.
#-----------------------------------------------------------------------------------------------------------------------------------

# imports
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

## ------------------------ Extract data from excel file -------------------
fixed_ssp_params    = []
adaptive_ssp_params = []
if os.path.exists("population_density_imex_stats.xlsx"):
    print("Extracting data from file.")
    #extract the fixed values and save into a list


else:
    print("Warning: populationModel_frames.png not found.")






##------------------------------- Generate Plots --------------------------
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
   
    it = nsteps - 1
    tval = repr(float(t[it])).zfill(3)
    plt.plot(x, pSol[it, :], "-b")
    plt.title(r"$t =$ " + tval) #add runtype and value so you know what you are saving
    plt.xlabel(r"$x$")
    plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.show()

##### end of script #####
