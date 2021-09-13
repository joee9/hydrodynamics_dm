# Joe Nyhan, 19 July 2021
# File for creation of static solutions for a fermion/boson star.

#%%
from numba.core.errors import NumbaDeprecationWarning
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numba import jit, njit

# ========== PARAMS

eos_UR = 0
eos_polytrope = 1
eos_SLy = 0

rmin = 1e-5
rmax = 100
dr = 0.02

mu = 1.122089
Lambda = 1

# shooting method
omega_upper = 2
omega_lower = 1
omega_init = 1.5
its = 100

# initial data
p0 = 1e-3
phi0 = 1e-3

dphi0 = 0
m0 = 0 
sigma0 = 1

# tolerances
dphi_tol_upper = 5
phi_tol_upper = 2
phi_tol_lower = 0
p_tol_lower = 1e-10

# ========== EOS

if eos_UR:
	Gamma = 1.3

	@njit
	def rho_from_p(p):
		if p == 0:
			return 0
		return p / (Gamma-1)

elif eos_polytrope:
	Gamma = 2
	K = 100

	@njit
	def rho_from_p(p):
		if p == 0: 
			return 0
		return (p/K)**(1/Gamma)

elif eos_SLy:

	df = pd.read_csv("0-SLy_vals.vals")
	SLy_rhos = df.iloc[:,0].to_numpy()
	SLy_ps = df.iloc[:,1].to_numpy()

	def rho_from_p(p):
		if p == 0:
			return 0
		f = interp1d(SLy_rhos,SLy_ps)
		return f(p)

# ========== ARRAYS

NUM_SPOINTS = int((rmax - rmin)/dr)

p		= np.zeros(NUM_SPOINTS); p[0] = p0
phi		= np.zeros(NUM_SPOINTS); phi[0] = phi0
dphi	= np.zeros(NUM_SPOINTS); dphi[0] = dphi0
sigma	= np.zeros(NUM_SPOINTS); sigma[0] = sigma0
m		= np.zeros(NUM_SPOINTS); m[0] = m0

break_phi_upper = 0
break_phi_lower = 0
break_dphi_upper = 0
break_p_lower = 0


# ========== FUNCTIONS

# # NO p
# @njit
# def Tb_tt(r, y, omega_hat):

# 	phi, dphi, sigma_hat, m = y
# 	N = 1 - 2*m/r
# 	return - (omega_hat**2/(N*sigma_hat**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


# @njit
# def Tb_rr(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	N = 1 - 2*m/r
# 	return + (omega_hat**2/(N*sigma_hat**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


# @njit
# def alpha(r, y, omega_hat):
# 	"""
# 	alpha'/alpha
# 	"""
# 	phi, dphi, sigma_hat, m = y
# 	N = 1 - 2*m/r
# 	T = Tb_rr(r, y, omega_hat)

# 	return + (4*np.pi*r/N) * T + (m/(r**2*N))

# @njit
# def a(r, y, omega_hat):
# 	"""
# 	a'/a
# 	"""
# 	phi, dphi, sigma_hat, m = y
# 	N = 1 - 2*m/r
# 	T = Tb_tt(r, y, omega_hat)

# 	return - (4*np.pi*r/N) * T - (m/(r**2*N))

# @njit
# def dm_dr(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	T = Tb_tt(r, y, omega_hat)
# 	return -4*np.pi*r**2*T

# @njit
# def dsigma_dr(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	alpha_ = alpha(r, y, omega_hat)
# 	a_ = a(r, y, omega_hat)
# 	return sigma_hat*(a_+alpha_)

# @njit
# def dphi_dr(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	return dphi

# @njit
# def ddphi_dr(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	alpha_ = alpha(r, y, omega_hat)
# 	a_ = a(r, y, omega_hat)
# 	N = 1 - 2*m/r
	
# 	first_term = dphi*(a_ - alpha_-2/r)
# 	second_term = - omega_hat**2/(N**2*sigma_hat**2) * phi
# 	third_term = (1/N) * (mu**2*phi + 2*Lambda*phi**3)

# 	return first_term + second_term + third_term

# with p

# @njit
def Tb_tt(r, y, omega_hat):

	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	return - (omega_hat**2/(N*sigma_hat**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


# @njit
def Tb_rr(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	return + (omega_hat**2/(N*sigma_hat**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

def Tf_tt(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return -rho_from_p(p)

def Tf_rr(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return p


# @njit
def alpha(r, y, omega_hat):
	"""
	alpha'/alpha
	"""
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_rr(r, y, omega_hat) + Tf_rr(r,y,omega_hat)

	return + (4*np.pi*r/N) * T + (m/(r**2*N))

# @njit
def a(r, y, omega_hat):
	"""
	a'/a
	"""
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_tt(r, y, omega_hat) + Tf_tt(r,y,omega_hat)

	return - (4*np.pi*r/N) * T - (m/(r**2*N))

# @njit
def dm_dr(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	T = Tb_tt(r, y, omega_hat) + Tf_tt(r,y,omega_hat)
	return -4*np.pi*r**2*T

# @njit
def dsigma_dr(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat)
	a_ = a(r, y, omega_hat)
	return sigma_hat*(a_+alpha_)

# @njit
def dphi_dr(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return dphi

# @njit
def ddphi_dr(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat)
	a_ = a(r, y, omega_hat)
	N = 1 - 2*m/r
	
	first_term = dphi*(a_ - alpha_-2/r)
	second_term = - omega_hat**2/(N**2*sigma_hat**2) * phi
	third_term = (1/N) * (mu**2*phi + 2*Lambda*phi**3)

	return first_term + second_term + third_term

def dp_dr(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha(r,y,omega_hat)
	rho = rho_from_p(p)
	return - alpha_ * (rho + p)

# @njit
def rk4(r, i, p, phi, dphi, sigma, m, omega):

	h = dr

	break_phi_upper = 0
	break_phi_lower = 0
	break_dphi_upper = 0
	break_p_lower = 0

	y = phi[i], dphi[i], p[i], sigma[i], m[i]
	# print(y)

	k1dphi = h * ddphi_dr(r, y, omega)
	k1phi = h * dphi_dr(r, y, omega)
	k1p = h * dp_dr(r, y, omega)
	k1sigma = h * dsigma_dr(r, y, omega)
	k1m = h * dm_dr(r, y, omega)

	yk1 = phi[i] + k1phi/2, dphi[i] + k1dphi/2, p[i] + k1p/2, sigma[i] + k1sigma/2, m[i] + k1m/2
	r_p1h = r + h/2

	k2dphi = h * ddphi_dr(r_p1h, yk1, omega)
	k2phi = h * dphi_dr(r_p1h, yk1, omega)
	k2p = h * dp_dr(r_p1h, yk1, omega)
	k2sigma = h * dsigma_dr(r_p1h, yk1, omega)
	k2m = h * dm_dr(r_p1h, yk1, omega)

	yk2 = phi[i] + k2phi/2, dphi[i] + k2dphi/2, p[i] + k2p/2, sigma[i] + k2sigma/2, m[i] + k2m/2

	k3dphi = h * ddphi_dr(r_p1h, yk2, omega)
	k3phi = h * dphi_dr(r_p1h, yk2, omega)
	k3p = h * dp_dr(r_p1h, yk2, omega)
	k3sigma = h * dsigma_dr(r_p1h, yk2, omega)
	k3m = h * dm_dr(r_p1h, yk2, omega)

	yk3 = phi[i] + k3phi, dphi[i] + k3dphi, p[i] + k3p, sigma[i] + k3sigma, m[i] + k3m
	r_p1 = r + h

	k4dphi = h * ddphi_dr(r_p1, yk3, omega)
	k4phi = h * dphi_dr(r_p1, yk3, omega)
	k4p = h * dp_dr(r_p1, yk3, omega)
	k4sigma = h * dsigma_dr(r_p1, yk3, omega)
	k4m = h * dm_dr(r_p1, yk3, omega)

	phi[i+1] = phi[i] + (k1phi + 2*k2phi + 2*k3phi + k4phi)/6
	dphi[i+1] = dphi[i] + (k1dphi + 2*k2dphi + 2*k3dphi + k4dphi)/6
	p[i+1] = p[i] + (k1p + 2*k2p + 2*k3p + k4p)/6
	sigma[i+1] = sigma[i] + (k1sigma + 2*k2sigma + 2*k3sigma + k4sigma)/6
	m[i+1] = m[i] + (k1m + 2*k2m + 2*k3m + k4m)/6

	if phi[i+1] == np.nan or dphi[i+1] == np.nan or p[i+1] == np.nan or sigma[i+1] == np.nan or m[i+1] == np.nan:
		return -1
	elif phi[i+1] > phi_tol_upper:
		return "phi upper"
	elif phi[i+1] < phi_tol_lower:
		return "phi lower"
	elif dphi[i+1] > dphi_tol_upper:
		return "dphi upper"
	elif p[i+1] < p_tol_lower:
		return "p"
	else: return 0


for ct in range(its):
	# r = rmin
	omega = (omega_upper + omega_lower)/2
	print(omega)

	for i in range(NUM_SPOINTS-1):
		r = rmin + i*dr
		sol = rk4(r,i,p,phi,dphi,sigma,m,omega)
		print(f"{sol=}, {i=}")
		if sol == -1:
			print("nan occured")
			break
		elif sol == "phi upper" or sol == "dphi upper":
			omega_lower = omega
			print("upper")
			break
		elif sol == "phi lower":
			omega_upper = omega
			print("upper")
			break
		elif sol == "p":
			print("broken at p")
			break
			ps = p
			p = np.zeros(NUM_SPOINTS)






# %%
