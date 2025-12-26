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
    # print(nsteps)


    # allocate solution data as 2D Python arrays
    t = np.zeros((nsteps), dtype=float)
    rho = np.zeros((nsteps, nx), dtype=float)
    mx = np.zeros((nsteps, nx), dtype=float)
    my = np.zeros((nsteps, nx), dtype=float)
    mz = np.zeros((nsteps, nx), dtype=float)
    et = np.zeros((nsteps, nx), dtype=float)
    x = np.linspace(xl, xr, nx)

    # store remaining data into numpy arrays
    for it in range(nsteps):
        lines.pop(0)
        line = (lines.pop(0)).split()
        t[it] = line.pop(0)
        for ix in range(nx):
            rho[it, ix] = line.pop(0)
            mx[it, ix] = line.pop(0)
            my[it, ix] = line.pop(0)
            mz[it, ix] = line.pop(0)
            et[it, ix] = line.pop(0)

# print(t[-1])
gamma   = 7.0/5.0
przdata = np.zeros((nx), dtype=float) #pressure
rhodata = np.zeros((nx), dtype=float) #density
veldata = np.zeros((nx), dtype=float) #velocity
eknot   = np.ones((nx),  dtype=float) #e_{0}
etdiff  = np.zeros((nx), dtype=float) #E - E_{0}
for i in range(nx):
    przdata[i] = (gamma-1.0) * (et[nsteps-1, i] - (mx[nsteps-1, i] * mx[nsteps-1, i] + my[nsteps-1, i] * my[nsteps-1, i] + mz[nsteps-1, i] * mz[nsteps-1, i]) * 0.5 / rho[nsteps-1, i])
    rhodata[i] = rho[nsteps-1, i]
    veldata[i] = mx[nsteps-1, i]/rho[nsteps-1, i]
    etdiff[i] = ( (et[nsteps-1, i]/rho[nsteps-1, i]) - 0.5 * ((mx[nsteps-1, i]/rho[nsteps-1, i])**2) ) - eknot[i] 
# end

# confirm the L2 error norm for (p - rho)
# sumerror = 0.0
# for i in range(len(rhodata)):
#     sumerror = sumerror + (rhodata[i]-przdata[i])*(rhodata[i]-przdata[i])
# l2error = np.sqrt(sumerror/nx)
# print("L2 error norm: %.6e" %l2error)

fig = plt.figure(figsize=(10, 5))
gs  = GridSpec(2, 2, figure=fig)

## density 
ax00 = fig.add_subplot(gs[0, 0])  
ax00.plot(x, rhodata, marker='x', markersize=6, linestyle='-',  color='blue', label="density",  linewidth=0.8)
ax00.set_ylabel(r"density")
ax00.set_xlabel(r"x")
# ax00.set_xticks(np.linspace(0,1,11))
# ax00.set_yticks(np.linspace(0.1,1.1,11))
plt.legend()

## pressure
ax01 = fig.add_subplot(gs[0, 1]) 
ax01.plot(x, przdata, marker='o', markersize=6, linestyle='-.', color='red',  label="pressure", linewidth=0.8)
ax01.set_ylabel(r"pressure")
ax01.set_xlabel(r"x")
# ax01.set_xticks(np.linspace(0,1,11))
# ax01.set_yticks(np.linspace(0.05,0.45,9))
plt.legend()

## velocity
ax03 = fig.add_subplot(gs[1, 0]) 
ax03.plot(x, veldata, marker='s', markersize=6, linestyle='--', color='black',  label="velocity", linewidth=0.8)
ax03.set_ylabel(r"velocity")
ax03.set_xlabel(r"x")
# ax03.set_xticks(np.linspace(0,1,11))
# ax03.set_yticks(np.linspace(-0.1,0.6,8))
plt.legend()

## E - E_{0}
ax04 = fig.add_subplot(gs[1, 1]) 
ax04.plot(x, etdiff, marker='+', markersize=6, linestyle=':', color='green',  label="$E - E_{0}$", linewidth=0.8)
ax04.set_ylabel(r"$E - E_{0}$")
ax04.set_xlabel(r"x")
# ax04.set_xticks(np.linspace(0,1,11))
plt.legend()

plt.savefig("hyperbolic_relaxation_frames.png")
plt.show()

##### end of script #####
