#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds @ SMU
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
# matplotlib-based plotting script for the serial ark_sod_lsrk example
# ------------------------------------------------------------------------------

# imports
import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# data file name
datafile = "euler_reaction.out"

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
    u = np.zeros((nt, nx), dtype=float)
    v = np.zeros((nt, nx), dtype=float)
    w = np.zeros((nt, nx), dtype=float)
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
            u[it, ix] = line.pop(0)
            v[it, ix] = line.pop(0)
            w[it, ix] = line.pop(0)

    gamma = 1.4
    ux = mx / rho
    p = (gamma - 1.0) * (et - (mx * mx + my * my + mz * mz) / (2.0 * rho))

#   plot defaults: increase default font size, increase plot width, enable LaTeX rendering
plt.rc("font", size=15)
plt.rcParams["figure.figsize"] = [10.0, 12.0]
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.constrained_layout.use"] = True

#   subplots with time snapshots of the density, x-velocity, pressure, and reactants (u,v,w)
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(6, 5, figure=fig)
ax00 = fig.add_subplot(gs[0, 0])  # first column
ax10 = fig.add_subplot(gs[1, 0])
ax20 = fig.add_subplot(gs[2, 0])
ax30 = fig.add_subplot(gs[3, 0])
ax40 = fig.add_subplot(gs[4, 0])
ax50 = fig.add_subplot(gs[5, 0])
ax01 = fig.add_subplot(gs[0, 1])  # second column
ax11 = fig.add_subplot(gs[1, 1])
ax21 = fig.add_subplot(gs[2, 1])
ax31 = fig.add_subplot(gs[3, 1])
ax41 = fig.add_subplot(gs[4, 1])
ax51 = fig.add_subplot(gs[5, 1])
ax02 = fig.add_subplot(gs[0, 2])  # third column
ax12 = fig.add_subplot(gs[1, 2])
ax22 = fig.add_subplot(gs[2, 2])
ax32 = fig.add_subplot(gs[3, 2])
ax42 = fig.add_subplot(gs[4, 2])
ax52 = fig.add_subplot(gs[5, 2])
ax03 = fig.add_subplot(gs[0, 3])  # fourth column
ax13 = fig.add_subplot(gs[1, 3])
ax23 = fig.add_subplot(gs[2, 3])
ax33 = fig.add_subplot(gs[3, 3])
ax43 = fig.add_subplot(gs[4, 3])
ax53 = fig.add_subplot(gs[5, 3])
ax04 = fig.add_subplot(gs[0, 4])  # fifth column
ax14 = fig.add_subplot(gs[1, 4])
ax24 = fig.add_subplot(gs[2, 4])
ax34 = fig.add_subplot(gs[3, 4])
ax44 = fig.add_subplot(gs[4, 4])
ax54 = fig.add_subplot(gs[5, 4])
it = 0
tval = repr(float(t[it])).zfill(3)
ax00.plot(x, rho[it, :], "-b")
ax10.plot(x, ux[it, :], "-b")
ax20.plot(x, p[it, :], "-b")
ax30.plot(x, u[it, :], "-b")
ax40.plot(x, v[it, :], "-b")
ax50.plot(x, w[it, :], "-b")
ax00.set_title(r"$t =$ " + tval)
ax00.set_ylabel(r"$\rho$")
ax10.set_ylabel(r"$v_x$")
ax20.set_ylabel(r"$p$")
ax30.set_ylabel(r"$u$")
ax40.set_ylabel(r"$v$")
ax50.set_ylabel(r"$w$")
ax50.set_xlabel(r"$x$")
it = nt // 4
tval = repr(float(t[it])).zfill(3)
ax01.plot(x, rho[it, :], "-b")
ax11.plot(x, ux[it, :], "-b")
ax21.plot(x, p[it, :], "-b")
ax31.plot(x, u[it, :], "-b")
ax41.plot(x, v[it, :], "-b")
ax51.plot(x, w[it, :], "-b")
ax01.set_title(r"$t =$ " + tval)
ax51.set_xlabel(r"$x$")
it = nt // 2
tval = repr(float(t[it])).zfill(3)
ax02.plot(x, rho[it, :], "-b")
ax12.plot(x, ux[it, :], "-b")
ax22.plot(x, p[it, :], "-b")
ax32.plot(x, u[it, :], "-b")
ax42.plot(x, v[it, :], "-b")
ax52.plot(x, w[it, :], "-b")
ax02.set_title(r"$t =$ " + tval)
ax52.set_xlabel(r"$x$")
it = nt * 3 // 4
tval = repr(float(t[it])).zfill(3)
ax03.plot(x, rho[it, :], "-b")
ax13.plot(x, ux[it, :], "-b")
ax23.plot(x, p[it, :], "-b")
ax33.plot(x, u[it, :], "-b")
ax43.plot(x, v[it, :], "-b")
ax53.plot(x, w[it, :], "-b")
ax03.set_title(r"$t =$ " + tval)
ax53.set_xlabel(r"$x$")
it = nt - 1
tval = repr(float(t[it])).zfill(3)
ax04.plot(x, rho[it, :], "-b")
ax14.plot(x, ux[it, :], "-b")
ax24.plot(x, p[it, :], "-b")
ax34.plot(x, u[it, :], "-b")
ax44.plot(x, v[it, :], "-b")
ax54.plot(x, w[it, :], "-b")
ax04.set_title(r"$t =$ " + tval)
ax54.set_xlabel(r"$x$")
plt.savefig("euler_reaction_frames.png")

#   time histories of reactants (from center of domain)
fig = plt.figure(figsize=(5, 4))
plt.plot(t, rho[:, nx // 2], "-k", label=r"$\rho$")
plt.plot(t, u[:, nx // 2], "-b", label=r"$u$")
plt.plot(t, v[:, nx // 2], "-r", label=r"$v$")
plt.plot(t, w[:, nx // 2], "-g", label=r"$w$")
plt.title(r"Reactant histories at %x =$ " + repr(float(x[nx // 2])).zfill(3))
plt.xlabel(r"$t$")
plt.savefig("euler_reaction_histories.png")

plt.show()

##### end of script #####
