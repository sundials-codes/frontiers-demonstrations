#!/usr/bin/env python3
# --------------------------------------------------------------------------------------------------------------------------------
# Programmer(s): Sylvia Amihere and Daniel R. Reynolds @ UMBC
# --------------------------------------------------------------------------------------------------------------------------------
# Used by runtests_population.py. Computes D, D_ref, |D_ref - D|/D_ref and the error at the final time, and saves the plot of the 
# solution at the final time. The plot title shows D, D_ref, |D_ref - D|/D_ref and the number of time steps with negative values.
#-----------------------------------------------------------------------------------------------------------------------------------

# imports
import sys, os
import numpy as np          
import matplotlib.pyplot as plt

# optional arguments (passed by runtests_population.py)
outfile  = sys.argv[1] if len(sys.argv) > 1 else "populationModel_frames.png"
negsteps = sys.argv[2] if len(sys.argv) > 2 else "not counted"
label    = sys.argv[3] if len(sys.argv) > 3 else ""

# data file name
datafile = "population.txt"

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

dx = (xr - xl)/N                      # spatial step size (same as in plot_population.py)

# allocate solution data as 2D Python arrays
t    = np.zeros((nsteps), dtype=float)
pSol = np.zeros((nsteps, N), dtype=float)

# store remaining data into numpy arrays
it = 0
for i in range(len(lines)):
    if "Time step" in lines[i]:
        t[it] = float(lines[i].split(':')[1])
        pSol[it, :] = np.array(lines[i+1].split(), dtype=float)
        it += 1
        if it == nsteps:
            break

# solution at the last time step
pSol_lastStep = np.array(pSol[nsteps-1, :], dtype=float)
print("last time read = %.16e" % t[nsteps-1])

def max_first_derivative(sol, dx):
    """ max |p_{i+1} - p_i| / dx over the N-1 neighbouring pairs """
    return np.max(np.abs(np.diff(sol))) / dx

# first derivative of the numerical solution (same formula as plot_population.py)
lmax_pSol_xi = max_first_derivative(pSol_lastStep, dx)
print("Lmax of first derivative at final time step = %.6f" % lmax_pSol_xi)

## ------------------ Extract Reference Solution at Final Time Step -----------------------
def read_ref_solution(filename):
    """ returns the solution vector at the final time step of the reference solution """
    with open(filename, "r") as file_ref:
        title_ref = file_ref.readline()
        T0_ref    = float((file_ref.readline().split())[2])
        Tf_ref    = float((file_ref.readline().split())[2])
        N_ref     = int((file_ref.readline().split())[2])
        last_line = ""
        for line in file_ref:
            if "Number of Time Steps" in line:
                break
            if line.strip():
                last_line = line
    return np.array(last_line.split(), dtype=float)

# reference solution for the diffusion coefficient in population.txt
ref_files = {0.0: "referenceSoln_population_diffk0.txt",
             0.02: "referenceSoln_population_diffk02.txt",
             0.04: "referenceSoln_population_diffk04.txt"}
pSol_ref = read_ref_solution(ref_files[diff_k])
elmax = np.max(np.abs(pSol_ref - pSol_lastStep))
dref  = max_first_derivative(pSol_ref, dx)
rel_diff = abs(dref - lmax_pSol_xi) / dref

print("Lmax error using reference solution = %.6e" % elmax)
print("Reference first-derivative max = %.6f" % dref)
print("Relative difference of first derivative = %.10e" % rel_diff)

## ------------------ Plot the solution at the final time step -----------------------
if outfile.lower() != "none":
    fig, ax = plt.subplots()
    ax.plot(x, pSol_lastStep)
    ax.set_title(f"{label}\n"
                 f"D = {lmax_pSol_xi:.4f},  D_ref = {dref:.4f},  |D_ref - D|/D_ref = {100*rel_diff:.3f}%,  "
                 f"negative steps = {negsteps}", fontsize=9)
    fig.savefig(outfile)
    plt.close(fig)

##### end of script #####
