#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Programmer(s):  Sylvia Amihere and Daniel R. Reynolds @ SMU
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
    # print(len(lines))
    # print(lines[6])


    # extract header information
    title = lines.pop(0)  # Title
    Nt = int(lines.pop(0).split()[4]) # number of time steps
    # print(Nt)
    T0 = float(lines.pop(0).split()[2])  # initial time
    # print(T0)
    Tf = float(lines.pop(0).split()[2]) # final time
    # print(Tf)
    N = int(lines.pop(0).split()[2]) # spatial dimension
    # print(N)
    xl = float(lines.pop(0).split()[2]) # spatial dimension
    # print(xl)
    xr = float(lines.pop(0).split()[2]) # spatial dimension
    # print(xr)
    x = np.linspace(xl, xr, N)


    # allocate solution data as 2D Python arrays
    t = np.zeros((Nt), dtype=float)
    pSol = np.zeros((Nt, N), dtype=float)

    # store remaining data into numpy arrays
    dt = (Tf-T0)/Nt
    # print(dt)
    
    it  = 0
    for i in range(0, len(lines)):
        if "Time step" in lines[i]:
            i=i+1
            pSol[it,:] = np.array(list(map(float, lines[i].split()))) #to remove single quotes around the vectors since each vector is a line
            t[it] = (it + 1) * dt
            it = it +1
    # print(len(pSol))
    # print(pSol)

    ##Alternatively
    # for it in range(Nt): #1 is added in order to include the last time step
    #     t[it] = (it + 1) * dt
    # # print(t)
    # all_pSol = []
    # for i in range(0, len(lines)):
    #     if "Time step" in lines[i]:
    #         # print(lines[i])
    #         i=i+1
    #         pSol = list(map(float, lines[i].split()))
    #         # print(pSol)
    #         all_pSol.append(pSol)
    # # print(len(all_pSol))
    # # print(len(all_pSol[0]))
    
    #   plot defaults: increase default font size, increase plot width, enable LaTeX rendering
    plt.rc("font", size=15)
    plt.rcParams["figure.figsize"] = [7.2, 4.8]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.constrained_layout.use"] = True

    #   subplots with time snapshots of the density, x-velocity, and pressure
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
    it = 2
    tval = repr(float(t[it])).zfill(3)
    ax01.plot(x, pSol[it, :], "-b")
    ax01.set_title(r"$t =$ " + tval)
    ax01.set_xlabel(r"$x$")
    it = Nt - 1
    tval = repr(float(t[it])).zfill(3)
    ax02.plot(x, pSol[it, :], "-b")
    ax02.set_title(r"$t =$ " + tval)
    ax02.set_xlabel(r"$x$")
    plt.savefig("populationModel_frames.png")

    plt.rc("font", size=15)
    plt.rcParams["figure.figsize"] = [7.2, 4.8]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.constrained_layout.use"] = True

    plt.show()

##### end of script #####
