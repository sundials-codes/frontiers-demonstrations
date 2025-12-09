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
    # lastline  = (lines[-1])
    # num_steps = lastline.split()
    # nsteps    = int(num_steps[2]) # total number of steps taken

    # allocate solution data as 2D Python arrays
    t = np.zeros((nt), dtype=float)
    rho = np.zeros((nt, nx), dtype=float)
    mx = np.zeros((nt, nx), dtype=float)
    my = np.zeros((nt, nx), dtype=float)
    mz = np.zeros((nt, nx), dtype=float)
    et = np.zeros((nt, nx), dtype=float)
    x = np.linspace(xl, xr, nx)

    # store remaining data into numpy arrays
    for it in range(nt):
        line = (lines.pop(0)).split()
        t[it] = line.pop(0)
        for ix in range(nx):
            rho[it, ix] = line.pop(0)
            mx[it, ix] = line.pop(0)
            my[it, ix] = line.pop(0)
            mz[it, ix] = line.pop(0)
            et[it, ix] = line.pop(0)

gamma = 7.0/5.0
przdata = np.zeros((nx), dtype=float)
rhodata = np.zeros((nx), dtype=float)
for i in range(nx):
    przdata[i] = (gamma-1.0) * (et[nt-1, i] - (mx[nt-1, i] * mx[nt-1, i] + my[nt-1, i] * my[nt-1, i] + mz[nt-1, i] * mz[nt-1, i]) * 0.5 / rho[nt-1, i])
#end
for i in range(nx):
    rhodata[i] = rho[nt-1, i]
#end

# confirm the L2 error norm for (p - rho)
# sumerror = 0.0
# for i in range(len(rhodata)):
#     sumerror = sumerror + (rhodata[i]-przdata[i])*(rhodata[i]-przdata[i])
# l2error = np.sqrt(sumerror/nx)
# print("L2 error norm: %.6e" %l2error)

plt.plot(x, przdata, marker='o', markersize=6, linestyle='-.', color='red',  label="pressure", linewidth=0.8)
plt.plot(x, rhodata, marker='x', markersize=6, linestyle='-',  color='blue', label="density",  linewidth=0.8)

plt.xlabel('x')
plt.ylabel('pressure / density')
plt.legend()
plt.savefig("hyperbolic_relaxation_frames.png")
# plt.show()

##### end of script #####
