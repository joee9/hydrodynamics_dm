# Joe Nyhan, 30 June
# Initial conditions for the hydrodynamical system.

from hd_params import *
from aux_dm.hd_equations import R
from aux_dm.hd_eos import P


def getvals(input):
	vals = []
	with open(input, "r") as f:
		for i in range(NUM_SPOINTS):
			vals.append(f.readline())
		
	print(f'{input} successfully read.\n')
	return np.array(vals)

# ========== PRIMITIVES

if PRIM_IC == "TOV solution polytrope":

	if charge:
		rho_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_rho.txt"
		p_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_P.txt"
	if not charge:
		rho_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vphi{vphi_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_rho.txt"
		p_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vphi{vphi_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_P.txt"

	initial_rho_vals = getvals(rho_file)
	initial_p_vals = getvals(p_file)

if PRIM_IC == "TOV solution SLy":
	rho_file = f"input/SLy_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_rho.txt"
	p_file = f"input/SLy_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_P.txt"

	initial_rho_vals = getvals(rho_file)
	initial_p_vals = getvals(p_file)

@njit
def rho_gaussian(r):
	return A*np.exp(-(r-r0)**2/d**2)+1E-13

if PRIM_IC == "Gaussian":
	rho_vals = []
	p_vals = []

	for i in range(NUM_SPOINTS):
		r = R(i+1/2)	# stored at a staggered grid; i + 1/2 -> i
		rho_ = rho_gaussian(r)
		rho_vals.append(rho_)
		p_vals.append(P(rho_))
		
	initial_rho_vals = np.array(rho_vals)
	initial_p_vals = np.array(p_vals)

# ========== SCALAR FIELDS

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

@njit
def A_r_gaussian(r):
	return 0

@njit
def z_gaussian(r):
	return 0

@njit
def Omega_gaussian(r):
	return 0

if SF_IC == "Gaussian":

	phi1_vals = []
	phi2_vals = []
	X1_vals = []
	X2_vals = []
	Y1_vals = []
	Y2_vals = []
	A_r_vals = []
	z_vals = []
	Omega_vals = []

	for i in range(NUM_SPOINTS):
		r = R(i)
		phi1_vals.append(phi1_gaussian(r))
		phi2_vals.append(phi2_gaussian(r))
		X1_vals.append(X1_gaussian(r))
		X2_vals.append(X2_gaussian(r))
		Y1_vals.append(Y1_gaussian(r))
		Y2_vals.append(Y2_gaussian(r))
		A_r_vals.append(A_r_gaussian(r))
		z_vals.append(z_gaussian(r))
		Omega_vals.append(Omega_gaussian(r))
	
	initial_phi1_vals = np.array(phi1_vals)
	initial_phi2_vals = np.array(phi2_vals)
	initial_X1_vals = np.array(X1_vals)
	initial_X2_vals = np.array(X2_vals)
	initial_Y1_vals = np.array(Y1_vals)
	initial_Y2_vals = np.array(Y2_vals)
	initial_A_r_vals = np.array(A_r_vals)
	initial_z_vals = np.array(z_vals)
	initial_Omega_vals = np.array(Omega_vals)

if SF_IC == "TOV solution polytrope":

	phi2_vals = []
	X2_vals = []
	Y1_vals = []

	for i in range(NUM_SPOINTS):
		phi2_vals.append(0)
		X2_vals.append(0)
		Y1_vals.append(0)

	if charge:
		varphi_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_varphi.txt"
		X_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_X.txt"
		Y_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_Y.txt"
		Z_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_Z.txt"
		Omega_file = f"input/polytrope_g{g_val:.3f}_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_Omega.txt"

		initial_phi1_vals = getvals(varphi_file)
		initial_phi2_vals = np.zeros(NUM_SPOINTS)
		initial_X1_vals = getvals(X_file)
		initial_X2_vals = np.zeros(NUM_SPOINTS)
		initial_Y1_vals = np.zeros(NUM_SPOINTS)
		initial_Y2_vals = getvals(Y_file)
		initial_z_vals = getvals(Z_file)
		initial_Omega_vals = getvals(Omega_file)
		initial_A_r_vals = np.zeros(NUM_SPOINTS)
	
	if not charge:
		varphi_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vphi{vphi_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_varphi.txt"
		X_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vphi{vphi_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_X.txt"
		Y_file = f"input/polytrope_K{K:.1f}_gamma{Gamma:.1f}_p{p_val:.8f}_vphi{vphi_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_Y.txt"

		initial_phi1_vals = getvals(varphi_file)
		initial_phi2_vals = np.zeros(NUM_SPOINTS)
		initial_X1_vals = getvals(X_file)
		initial_X2_vals = np.zeros(NUM_SPOINTS)
		initial_Y1_vals = np.zeros(NUM_SPOINTS)
		initial_Y2_vals = getvals(Y_file)
		initial_z_vals = np.zeros(NUM_SPOINTS)
		initial_Omega_vals = np.zeros(NUM_SPOINTS)
		initial_A_r_vals = np.zeros(NUM_SPOINTS)

if SF_IC == "TOV solution SLy":

	phi2_vals = []
	X2_vals = []
	Y1_vals = []

	for i in range(NUM_SPOINTS):
		phi2_vals.append(0)
		X2_vals.append(0)
		Y1_vals.append(0)

	varphi_file = f"input/SLy_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_varphi.txt"
	X_file = f"input/SLy_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_X.txt"
	Y_file = f"input/SLy_p{p_val:.8f}_vc{vc_val:.8f}_lam{Lambda:.3f}_dr{dr:.3f}_Y.txt"

	initial_phi1_vals = getvals(varphi_file)
	initial_phi2_vals = np.zeros(NUM_SPOINTS)
	initial_X1_vals = getvals(X_file)
	initial_X2_vals = np.zeros(NUM_SPOINTS)
	initial_Y1_vals = np.zeros(NUM_SPOINTS)
	initial_Y2_vals = getvals(Y_file)




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

# ========== GAUSSIAN