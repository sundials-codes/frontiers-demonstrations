#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------------------------------------
# Programmer(s): Sylvia Amihere and Daniel R. Reynolds @ SMU
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
datafile = "linear_adv_rec.txt"

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
    dx     = (xr - xl)/(N)                      # spatial step size
    x      = np.linspace(xl + dx, xr, N)

    lastline  = (lines[-1])
    num_steps = lastline.split(':')
    nsteps    = int(num_steps[1].strip()) # total number of steps taken

    dt = (Tf-T0)/nsteps                   # temporal step size
   
    
    # allocate solution data as 2D Python arrays
    t    = np.zeros((nsteps), dtype=float)
    uSol = np.zeros((nsteps, N), dtype=float)
    vSol = np.zeros((nsteps, N), dtype=float)

    # store remaining data into numpy arrays
    it = 0
    for i in range(len(lines)):
        if "Time step" in lines[i]:
            # Extract the time value
            get_t = lines[i].split(':')
            t[it] = float(get_t[1].strip())
            
            # the solution at each time step contains the 200 numbers (stored as [u0,v0,u1,v1,u2,v2,...,u99,v99])
            sols = lines[i+1].split()
            full_sol = np.array(sols, dtype=float)
            
            # extract only the solution u and v (stored as [u0,v0,u1,v1,u2,v2,...,u99,v99])
            u_only = full_sol[0::2]
            v_only = full_sol[1::2]
            
            uSol[it, :] = u_only
            vSol[it, :] = v_only
            it = it + 1

    ## Extract solution u at the last time step
    uSol_lastStep = np.zeros((N), dtype=float)
    for i in range(len(uSol[nsteps-1, :])):
        uSol_lastStep[i] = uSol[nsteps-1, i] 

    ## Extract solution v at the last time step
    vSol_lastStep = np.zeros((N), dtype=float)
    for i in range(len(vSol[nsteps-1, :])):
        vSol_lastStep[i] = vSol[nsteps-1, i]
   

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(x, uSol[0, :], color = 'r', linestyle = '--', label = "initial solution")
    ax1.plot(x, uSol[-1, :], color = 'b', linestyle = '-', label = "final solution")
    ax1.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax1.set_title("u-component")
    ax1.grid(True)

    ax2.plot(x, vSol[0, :], color = 'r', linestyle = '--', label = "initial solution")
    ax2.plot(x, vSol[-1, :], color = 'b', linestyle = '-', label = "final solution")
    ax2.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax2.set_title("v-component")
    ax2.grid(True)

    # ## plot defaults: increase default font size, increase plot width, enable LaTeX rendering
    # plt.rc("font", size=15)
    # plt.rcParams["figure.figsize"] = [7.2, 4.8]
    # plt.rcParams["text.usetex"] = True
    # plt.rcParams["figure.constrained_layout.use"] = True

    # ## subplots with time snapshots of the density, x-velocity, and pressure
    # fig = plt.figure(figsize=(10, 5))
    # gs = GridSpec(1, 3, figure=fig)
    # ax00 = fig.add_subplot(gs[0, 0])  # left column
    # ax01 = fig.add_subplot(gs[0, 1])  # middle column
    # ax02 = fig.add_subplot(gs[0, 2])  # right column
    # it = 0
    # tval = repr(float(t[it])).zfill(3)
    # ax00.plot(x, vSol[it, :], "-b",)
    # ax00.set_title(r"$t =$ " + tval)
    # ax00.set_ylabel(r"$v(t,x)$")
    # ax00.set_xlabel(r"$x$")
    # # ax00.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # middleval = int(np.ceil(nsteps/2))
    # it = middleval
    # tval = repr(float(t[it])).zfill(3)
    # ax01.plot(x, vSol[it, :], "-b")
    # ax01.set_title(r"$t =$ " + tval)
    # ax01.set_xlabel(r"$x$")
    # # ax01.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # it = -1
    # tval = repr(float(t[it])).zfill(3)
    # ax02.plot(x, vSol[it, :], "-b")
    # ax02.set_title(r"$t =$ " + tval)
    # ax02.set_xlabel(r"$x$")
    # # ax02.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    # plt.savefig("linear_adv_rec_frames.png")

    # plt.rc("font", size=15)
    # plt.rcParams["figure.figsize"] = [7.2, 4.8]
    # plt.rcParams["text.usetex"] = True
    # plt.rcParams["figure.constrained_layout.use"] = True

    plt.legend()
    # plt.show()
    plt.close()


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
        title_ref = file_ref.readline()
        T0_ref    = float((file_ref.readline().split())[2])
        Tf_ref    = float((file_ref.readline().split())[2])
        N_ref     = int((file_ref.readline().split())[2])
        xl_ref    = float((file_ref.readline().split())[2])
        xr_ref    = float((file_ref.readline().split())[2])

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

    uSolRefFinal = np.zeros((N_ref), dtype=float)
    vSolRefFinal = np.zeros((N_ref), dtype=float)
        
    uSolRefFinal = np.array(last_data[0::2], dtype=float)
    vSolRefFinal = np.array(last_data[1::2], dtype=float)
    
    return uSolRefFinal, vSolRefFinal

## -------------------- Compute L-infinty norm using the reference solution -----------------------
k1Val1 = False #only one type of stiffness parameter option cna be true at a time (keep as only "1" space before and after =)
k1Val1e6 = True
# k1Val1e8 = False

AdaptiveRun = True #only one type of run can be true at a time (keep as only "1" space before and after =)
FixedRun = False

# l-infinity error
elmax = 0.0 
if (FixedRun):
    if(k1Val1):
        fixed_k1Val1_uSol_ref, fixed_k1Val1_vSol_ref  = read_ref_solution("refSoln_linear_adv_rec_k1Val1.txt")
        elmax = np.max(np.abs(fixed_k1Val1_vSol_ref - vSol_lastStep))

    elif(k1Val1e6):
        fixed_k1Val1e6_uSol_ref, fixed_k1Val1e6_vSol_ref  = read_ref_solution("refSoln_linear_adv_rec_k1Val1e6.txt")
        elmax = np.max(np.abs(fixed_k1Val1e6_vSol_ref - vSol_lastStep))

    # elif(k1Val1e8):
    #     fixed_k1Val1e8_uSol_ref, fixed_k1Val1e8_vSol_ref = read_ref_solution("refSoln_linear_adv_rec_k1Val1e8.txt")
    #     elmax = np.max(np.abs(fixed_k1Val1e8_vSol_ref - vSol_lastStep))

elif (AdaptiveRun):
    if (k1Val1): 
        adaptive_k1Val1_uSol_ref, adaptive_k1Val1_vSol_ref = read_ref_solution("refSoln_linear_adv_rec_k1Val1.txt")
        elmax = np.max(np.abs(adaptive_k1Val1_vSol_ref - vSol_lastStep))

    elif (k1Val1e6): 
        adaptive_k1Val1e6_uSol_ref, adaptive_k1Val1e6_vSol_ref = read_ref_solution("refSoln_linear_adv_rec_k1Val1e6.txt")
        # print("level %d", len(adaptive_k1Val1e6_vSol_ref))
        # plt.plot(x, adaptive_k1Val1e6_vSol_ref)
        # plt.plot(x, adaptive_k1Val1e6_uSol_ref)
        # plt.show()
        elmax = np.max(np.abs(adaptive_k1Val1e6_vSol_ref - vSol_lastStep))

    # elif (k1Val1e8): 
    #     adaptive_k1Val1e8_uSol_ref, adaptive_k1Val1e8_vSol_ref = read_ref_solution("refSoln_linear_adv_rec_k1Val1e8.txt")
    #     elmax = np.max(np.abs(adaptive_k1Val1e8_vSol_ref - vSol_lastStep))
# end

print("Lmax error using reference solution = %.4e" %elmax)

##### end of script #####
