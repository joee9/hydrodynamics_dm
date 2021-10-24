# Joe Nyhan, 30 June 2021
# Global parameters

# globally used packages
import sys
import numpy as np; np.set_printoptions(threshold=sys.maxsize)
from numba import njit
from time import process_time
# NUMBA_DISABLE_JIT = 1

# for recording data

record_data = 1
record_ringdown = 1
output_number = 2

path = "data/"

# ========== NUMERICAL INTEGRATION METHOD

int_modified_euler = 1
int_rk3 = 0

if int_modified_euler:  int_method = "modified euler"
if int_rk3:             int_method = "RK3"

# ========== EQUATION OF STATE

darkmatter = False 
charge     = False

eos_UR = 0
eos_polytrope = 0
eos_SLy = 1


if eos_UR:          eos = "ultra relativistic"
if eos_polytrope:   eos = "polytrope"
if eos_SLy:         eos = "SLy"

# initialize necessary parameters given an eos
if eos_UR:
    Gamma = 1.3

if eos_polytrope:
    K = 100
    Gamma = 2

if eos_SLy:
    pass

# general parameters for TOV solutions
p_val = 1e-4

# ========== INITIAL CONDITIONS

PRIM_IC_GAUSSIAN = 0
PRIM_IC_TOV = 1

# primitive variables
if  PRIM_IC_GAUSSIAN:
    # gives a gaussian for rho, then used for finding P
    A = 1E-5     # amplitude
    r0 = 5        # center
    d = 2        # exponential argument scale
    PRIM_IC = "Gaussian"

if PRIM_IC_TOV:
    if eos_polytrope:
        PRIM_IC = "TOV solution polytrope"    # from Ben
    elif eos_SLy:
        PRIM_IC = "TOV solution SLy"            # from Ben


# ========== GRID PARAMETERS

dr      = 0.02
rmin    = 0
rmax    = 100

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

floor = 1E-12

# Newton Rhapson rootfinding parameters
NR_MAX_ITERATIONS = 50
NR_TOL = 1E-7

mu = 1
Lambda = 0
G = 1
# if charge:
#   g = .65*np.sqrt(8*np.pi)
# elif not charge:
#   g = 1e-10

# for indexing
P_i = 0
rho_i = 1
v_i = 2

Pi_i = 0
Phi_i = 1

phi_i = 0
X_i = 1
Y_i = 2
