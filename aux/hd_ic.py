# Joe Nyhan, 30 June
# Initial conditions for the hydrodynamical system.

from hd_params import *
from aux.hd_equations import R
from aux.hd_eos import P

# ========== CONSERVATIVE FROM PRIMITIVE

@njit
def Pi(prim):
    """
    prim: the primitive values at a given spatial and temporal point
    """
    P, rho, v = prim
    return (rho + P) / (1 - v) - P

@njit
def Phi(prim):
    """
    prim: the primitive values at a given spatial and temporal point
    """
    P, rho, v = prim
    return (rho + P) / (1 + v) - P

# ========== FOR GETTING VALUES FROM FILE

def getvals(input):
    vals = []

    for i in range(NUM_VPOINTS):
        vals.append(0)

    with open(input, "r") as f:
        for i in range(NUM_SPOINTS-NUM_VPOINTS):
            val = f.readline()
            if (val != ""):    vals.append(val)
            else: vals.append(0)

    vals = np.array(vals)
    print(f'{input} successfully read.\n')

    return vals

# ========== PRIMITIVES


if PRIM_IC == "TOV solution polytrope":

    vals_path = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}"

    rho_file = f"{vals_path}_rho.txt"
    p_file = f"{vals_path}_P.txt"

    initial_rho_vals = getvals(rho_file)
    initial_p_vals = getvals(p_file)

if PRIM_IC == "TOV solution fit":

    vals_path = f"input/{eos}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}"

    rho_file = f"{vals_path}_rho.txt"
    p_file = f"{vals_path}_P.txt"

    initial_rho_vals = getvals(rho_file)
    initial_p_vals = getvals(p_file)


@njit
def rho_gaussian(r):
    return A*np.exp(-(r-r0)**2/d**2)+1E-13

if PRIM_IC == "Gaussian":
    rho_vals = []
    p_vals = []

    for i in range(NUM_SPOINTS):
        r = R(i+1/2)    # stored at a staggered grid; i + 1/2 -> i
        rho_ = rho_gaussian(r)
        rho_vals.append(rho_)
        p_vals.append(P(rho_))
        
    initial_rho_vals = np.array(rho_vals)
    initial_p_vals = np.array(p_vals)

# ========== SCALAR FIELD

@njit
def phi1_gaussian(r):
    return A_1*np.exp(-(r-r0_1)**2/d_1**2)

@njit
def X1_gaussian(r):
    return -(2*A_1/d_1**2)*(r-r0_1)*np.exp(-(r-r0_1)**2/d_1**2)

@njit
def Y1_gaussian(r):
    return 0

@njit
def phi2_gaussian(r):
    return A_2*np.exp(-(r-r0_2)**2/d_2**2)

@njit
def X2_gaussian(r):
    return -(2*A_2/d_2**2)*(r-r0_2)*np.exp(-(r-r0_2)**2/d_2**2)

@njit
def Y2_gaussian(r):
    return 0


if SF_IC == "Gaussian":

    phi1_vals = []
    phi2_vals = []
    X1_vals = []
    X2_vals = []
    Y1_vals = []
    Y2_vals = []

    for i in range(NUM_SPOINTS):
        r = R(i)
        phi1_vals.append(phi1_gaussian(r))
        phi2_vals.append(phi2_gaussian(r))
        X1_vals.append(X1_gaussian(r))
        X2_vals.append(X2_gaussian(r))
        Y1_vals.append(Y1_gaussian(r))
        Y2_vals.append(Y2_gaussian(r))
    
    initial_phi1_vals = np.array(phi1_vals)
    initial_phi2_vals = np.array(phi2_vals)
    initial_X1_vals = np.array(X1_vals)
    initial_X2_vals = np.array(X2_vals)
    initial_Y1_vals = np.array(Y1_vals)
    initial_Y2_vals = np.array(Y2_vals)

# for both cases of SF_IC

if not SF_IC == "Gaussian":

    # this is unnecessary

    # phi2_vals = []
    # X2_vals = []
    # Y1_vals = []

    # for i in range(NUM_SPOINTS):
    #     phi2_vals.append(0)
    #     X2_vals.append(0)
    #     Y1_vals.append(0)

    if SF_IC == "TOV solution polytrope":

        vals_path = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}"

        varphi_file = f"{vals_path}_varphi.txt"
        X_file = f"{vals_path}_X.txt"
        Y_file = f"{vals_path}_Y.txt"

    if SF_IC == "TOV solution fit":

        vals_path = f"input/{eos}_p{p_val:.8f}_dr{dr:.3f}"

        varphi_file = f"{vals_path}_varphi.txt"
        X_file = f"{vals_path}_X.txt"
        Y_file = f"{vals_path}_Y.txt"

    initial_phi1_vals = getvals(varphi_file)
    initial_phi2_vals = np.zeros(NUM_SPOINTS)
    initial_X1_vals = getvals(X_file)
    initial_X2_vals = np.zeros(NUM_SPOINTS)
    initial_Y1_vals = np.zeros(NUM_SPOINTS)
    initial_Y2_vals = getvals(Y_file)
