#!/usr/bin/env python3
# ------------------------------------------------------------------------------
# Programmer(s):  Daniel R. Reynolds and Sylvia Amihere @ SMU
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

# print(t)


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

# print(rho)


### ********************************************************************************************* ###
# #Amihere: rough work
# itx = 10
# print("the rho array is:", rho[itx]) #Amihere
# print("the rhoTRUE array is:", rhotrue[itx]) #Amihere

# result_rho = 0.0
# for ix in range(len(rho[itx])):
#     result_rho = result_rho + abs(rho[itx][ix] - rhotrue[itx][ix])**2
#     # result_rho = result_rho + abs(rho[it][ix])**2
# result_rho = np.sqrt(result_rho/nx)
# print("rs", result_rho)

PrintNorms = False
DoPlots = True

## Amihere (Task2): Plotting the L2 error norm :: ||rho_{xx}(t,x)|| - ||rho_{xx}(0,x)||
itx = 0
itx2 = 1
t2 = t[itx2:] 

dx = ((xr - xl)/nx) * 1.0

FDL2Enorm = []


## Amihere (Task3): plot (||rho_{xx}(t,x)||_{x} - ||rho_{xx}(0,x)||_{x}) -- want this to be strictly negative at each t, but we'd be okay with it small and positive
##        (Task3b): compute max_t(||rho_{xx}(t,x)||_{x} - ||rho_{xx}(0,x)||_{x}) -- want this to be strictly negative.
rhoPrevSum = 0.0
for ii in range(1, len(rho[0]) - 1):
    rhoPrevSum = rhoPrevSum + abs((1.0/(dx**2))*(rho[itx][ii-1] - 2.0*rho[itx][ii] + rho[itx][ii+1]))**2
rhoPrevSum = np.sqrt(rhoPrevSum/nx)
print ("Initial L2 of second derivative = %.14f \n" %rhoPrevSum)

rhoTDiffMax = 0.0
for it in range(itx2, len(rho)):
    rhoTSum = 0.0
    for ii in range(1, len(rho[1]) - 1):
        rhoTSum = rhoTSum + abs((1.0/(dx**2))*(rho[it][ii-1] - 2.0*rho[it][ii] + rho[it][ii+1]))**2
    rhoTSum = np.sqrt(rhoTSum/nx)
    rhoT0Diff = (rhoTSum - rhoPrevSum)/rhoPrevSum
    FDL2Enorm.append(rhoT0Diff)
    # print("FD error at time step %f is %.14f." %(t[it] ,rhoT0Diff))

    rhoTDiffMax = max(rhoTDiffMax, rhoT0Diff)
    rhoPrevSum = rhoTSum

rhoTDiffTol = 0.001
if (rhoTDiffMax <= rhoTDiffTol):
    print("Satisfies SSP condition")
else:
    print("Fails SSP condition. Max relative increase = %.14f (tolerance = %.14f) \n" %(rhoTDiffMax, rhoTDiffTol))
    
# print(FDL2Enorm)


## Amihere (Task4): Plotting the L2 error norm :: ||rho(t,x) - rhotrue(t,x)||_{x}
L2Enorm = []

## Amihere (Task5): Computing the L2 error at each time step :: ||rho(t,x) - rhotrue(t,x)||_{(t,x)}
for it in range(len(rho)):
    L2_error = 0.0
    for ix in range(len(rho[1])):
        L2_error = L2_error + abs(rho[it][ix] - rhotrue[it][ix])**2
        # L2_error = L2_error + abs(rho[it][ix])**2 #Amihere: check computed rhorms output on screen
    L2_error = np.sqrt(L2_error/nx)
    L2Enorm.append(L2_error)
    if (PrintNorms):
        print("L2 error norm at time step %f is %.14f." %(t[it] ,L2_error))

# print(L2Enorm)
### ********************************************************************************************* ###


if (DoPlots):
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
    it = 1
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

    plt.rc("font", size=15)
    plt.rcParams["figure.figsize"] = [7.2, 4.8]
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.constrained_layout.use"] = True


    # ## Amihere Task2: Plotting the second order L2 error norm as a function of time
    # fig2 = plt.figure()
    # gs = GridSpec(1, 1, figure=fig2)
    # ax00 = fig2.add_subplot(gs[0, 0])
    # ax00.plot(t2, FDL2Enorm, "-*k")
    # ax00.set_ylabel(r"$||\rho_{xx}(t)|| - ||\rho_{xx}(0)||$")
    # ax00.set_xlabel(r"$t$")
    # plt.savefig("advection_frames_FDL2norm.png")


    # ## Amihere Task4: Plotting the L2 error norm as a function of time
    # fig3 = plt.figure()
    # gs = GridSpec(1, 1, figure=fig3)
    # ax00 = fig3.add_subplot(gs[0, 0])  # left column
    # ax00.plot(t, L2Enorm, "-or")
    # ax00.set_ylabel(r"$||\rho - \rho_{true}||$")
    # ax00.set_xlabel(r"$t$")
    # plt.savefig("advection_frames_L2norm.png")


    plt.show()

##### end of script #####
