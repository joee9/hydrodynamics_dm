# Joe Nyhan, 30 June 2021
# Global parameters

# globally used packages
import sys
import os
import numpy as np; np.set_printoptions(threshold=sys.maxsize)
from numba import njit
from time import process_time

# for recording data

record_data = 0
record_ringdown = 1
output_number = 1   

continue_run = 0

path = "data/"

# ========== NUMERICAL INTEGRATION METHOD

# int_modified_euler = 1
int_rk3 = 1

# if int_modified_euler:  int_method = "modified euler"
if int_rk3:
    int_method = "RK3"

# ========== EQUATION OF STATE

darkmatter = False 

# different EOS
eos_UR        = 0
eos_polytrope = 0

eos_SLy       = 1
eos_FPS       = 0

eos_BSk19     = 0
eos_BSk20     = 0
eos_BSk21     = 0


if eos_UR:          eos = "UR"
if eos_polytrope:   eos = "polytrope"

if eos_SLy:         eos = "SLy"
if eos_FPS:         eos = "FPS"

if eos_BSk19:       eos = "BSk19"
if eos_BSk20:       eos = "BSk20"
if eos_BSk21:       eos = "BSk21"

# initialize necessary parameters given an eos
if eos_UR:
    Gamma = 1.3

if eos_polytrope:
    K = 100
    Gamma = 2


# general parameters for TOV solutions
p_val = 0.00305762
vc_val = 10**(-1.5)

# ========== INITIAL CONDITIONS

PRIM_IC_GAUSSIAN = 0
PRIM_IC_TOV = 1

SF_IC_GAUSSIAN = 0
SF_IC_TOV = 1

# primitive variables
if  PRIM_IC_GAUSSIAN:
    # gives a gaussian for rho, then used for finding P
    A = 1E-5     # amplitude
    r0 = 5        # center
    d = 2        # exponential argument scale
    PRIM_IC = "Gaussian"

if PRIM_IC_TOV:
    if eos_polytrope:
        PRIM_IC = "TOV solution polytrope"
    else:
        PRIM_IC = "TOV solution fit"
        
# TODO: change these parameter names to be different than for prim gaussian (ie A1, etc.)
if SF_IC_GAUSSIAN:
	# parameters for first scalar field
	A_1 = 1e-5
	r0_1 = 5
	d_1 = 2

	# parameters for second scalar field
	A_2 = 1e-5
	r0_2 = 10
	d_2 = 2

	SF_IC = "Gaussian"

if SF_IC_TOV:
    if eos_polytrope:
        SF_IC = "TOV solution polytrope"
    else:
        SF_IC = "TOV solution fit"


# ========== GRID PARAMETERS

dr      = 0.02
rmin    = 0
rmax    = 350

gamma   = 0.5
tmin    = 0
tmax    = 5000

dt = gamma * dr

# number of virtual points to the *left* of zero
NUM_VPOINTS = 1
# number of spatial points kept in grid
NUM_SPOINTS = int((rmax - rmin)/dr)
# number of temporal points kept in the grid
NUM_TPOINTS = int((tmax - tmin)/dt)

# number of staved steps before wrapping around; to save memory
T_STEPS_ARRAY = 5

# number of spatial points to write to file
R_POINTS_FILE = 1000
# step between spatial points in file
R_STEP_FILE = NUM_SPOINTS // R_POINTS_FILE

RECORD_INTERVAL = 500 # temporary interval to write to file

RING_INTERVAL = 5

PRINT_INTERVAL = 25
# TODO: add number of spatial points to write to file, number of time steps to record to file, number of time steps to display to screen; these should all be determined here and available to all other files

# TODO: initialize all ringdown parameters

# ========== MISC OTHER PARAMETERS

floor = 1E-15

# Newton Rhapson rootfinding parameters
NR_MAX_ITERATIONS = 50
NR_TOL = 1E-7

mu = 1.22089
Lambda = 0
G = 1

# for indexing
P_i = 0
rho_i = 1
v_i = 2

Pi_i = 0
Phi_i = 1

phi_i = 0
X_i = 1
Y_i = 2

# MISC WARNINGS
# if darkmatter and int_modified_euler:
#     print("Using modified Euler with darkmatter; will not be stable!")
#     exit()

# ensure not overwriting data
while os.path.exists(f"{path}/{output_number}-0params.txt"):
    output_number += 1
    print(f"Updating output number; now {output_number}.")