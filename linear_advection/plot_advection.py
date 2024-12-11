#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
# Modified by Sylvia Amihere @ SMU
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
datafile = "advection.out"

#parameters
rho0 = 0.01
rho1 = 1.0

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


# utility routines for computing the analytical solution
def advection_solution(t, x):
    if (x >= 0.25+t) and (x<=0.5+t):
        return rho1
    else:
        return rho0


# generate analytical solutions over same mesh and times as loaded from data file
rhotrue = np.zeros((nt, nx), dtype=float)
for it in range(nt):
    for ix in range(nx):
        rhotrue[it, ix] = advection_solution(t[it], x[ix])

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
ax00.plot(x, rho[it, :], "-b", x, rhotrue[it, :], ":k")
ax00.set_title(r"$t =$ " + tval)
ax00.set_ylabel(r"$\rho$")
# ax10.set_ylabel(r"$v_x$")
# ax20.set_ylabel(r"$p$")
ax00.set_xlabel(r"$x$")
it = nt // 2
tval = repr(float(t[it])).zfill(3)
ax01.plot(x, rho[it, :], "-b", x, rhotrue[it, :], ":k")
ax01.set_title(r"$t =$ " + tval)
ax01.set_xlabel(r"$x$")
it = nt - 1
tval = repr(float(t[it])).zfill(3)
ax02.plot(x, rho[it, :], "-b", x, rhotrue[it, :], ":k")
ax02.set_title(r"$t =$ " + tval)
ax02.set_xlabel(r"$x$")
plt.savefig("advection_frames.png")

plt.show()

##### end of script #####
