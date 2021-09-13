
#%%
import sys
import numpy as np; np.set_printoptions(threshold=sys.maxsize)
import pandas as pd
from scipy import integrate
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit

# ========== PARAMETERS

eos_UR = 0
eos_polytrope = 1
eos_SLy = 0

rmin = 0.000001
rmax = 100
dr = 0.02

# initial conditions
phi0 = 1e-3
p0 = 1e-3

m0 = 0
sigma0 = 1
dphi0 = 0

# omega_hat0 = .80262
omega = 1.12
omega_upper = 2
omega_lower = 1 
its = 100

mu = 1.122089
Lambda = 0

phi_tol_lower = 0
phi_tol_upper = 2
dphi_tol_upper = 5
p_tol_lower = 1e-10

# ========== EOS

if eos_UR:
	Gamma = 1.3
if eos_polytrope:
	Gamma = 2
	K = 100
if eos_SLy:
	pass

if eos_SLy:
	df = pd.read_csv("0-SLy_vals.vals")
	SLy_rhos = df.iloc[:,0].to_numpy()
	SLy_ps = df.iloc[:,1].to_numpy()

def rho_from_P(p):
	# print(f"{p=}")
	# if p <= 0: return 0
	if eos_UR:
		return p/(Gamma-1)
	elif eos_polytrope:
		return (p/K)**(1/Gamma)
	elif eos_SLy:
		f = interp1d(SLy_rhos, SLy_ps)
		return f(p)

# ========== EQUATIONS

def Tf_tt(p):
	# if p == 0: return 0
	rho = rho_from_P(p)
	return -rho

def Tf_rr(p):
	return p

def Tb_tt(r, phi, dphi, sigma, m, omega):
	N = 1 - 2*m/r
	return - (omega**2/(N*sigma**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

def Tb_rr(r, phi, dphi, sigma, m, omega):
	N = 1 - 2*m/r
	return + (omega**2/(N*sigma**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

def alpha(r, phi, dphi, sigma, m, p, omega, integrate_p):
	"""
	alpha'/alpha; factor used extensively throughout RHS of ODEs
	"""
	N = 1-2*m/r
	if integrate_p:
		Ttot_rr = Tf_rr(p) + Tb_rr(r, phi, dphi, sigma, m, omega)
	else:
		Ttot_rr = Tb_rr(r, phi,dphi,sigma,m,omega)
	return + (4*np.pi*r/N) * Ttot_rr + (m/(r**2*N))

def a(r, phi, dphi, sigma, m, p, omega, integrate_p):
	"""
	a'/a; factor used extensively throughout RHS of ODEs
	"""
	N = 1-2*m/r
	if integrate_p:
		Ttot_tt = Tf_tt(p) + Tb_tt(r, phi, dphi, sigma, m, omega)
	else:
		Ttot_tt = Tb_tt(r, phi,dphi,sigma,m,omega)
	return - (4*np.pi*r/N) * Ttot_tt - (m/(r**2*N))

def f_phi(r, phi, dphi, sigma, m, p, omega, integrate_p):
	return dphi

def f_dphi(r, phi, dphi, sigma, m, p, omega, integrate_p):
	a_ = a(r, phi, dphi, sigma, m, p, omega, integrate_p)
	alpha_ = alpha(r, phi, dphi, sigma, m, p, omega, integrate_p)
	N = 1-2*m/r

	first = + dphi*(a_ - alpha_ - 2/r)
	second = - (omega**2/(N**2*sigma**2)) * phi
	third = + (1/N) * (mu**2*phi + 2*Lambda*phi**3)

	return first + second + third

def f_sigma(r, phi, dphi, sigma, m, p, omega, integrate_p):
	a_ = a(r, phi, dphi, sigma, m, p, omega, integrate_p)
	alpha_ = alpha(r, phi, dphi, sigma, m, p, omega, integrate_p)
	return sigma*(a_ + alpha_)

def f_m(r, phi, dphi, sigma, m, p, omega, integrate_p):
	if integrate_p:
		Ttot_tt = Tf_tt(p) + Tb_tt(r, phi, dphi, sigma, m, omega)
	else: Ttot_tt = Tb_tt(r, phi, dphi, sigma, m, omega)
	return - 4*np.pi*r**2*Ttot_tt

def f_p(r, phi, dphi, sigma, m, p, omega, integrate_p):
	if not integrate_p: return 0
	alpha_ = alpha(r, phi, dphi, sigma, m, p, omega, integrate_p)
	rho = rho_from_P(p)
	return -alpha_*(rho+p)

# ========== EVOLUTION

NUM_SPOINTS = int((rmax - rmin)/dr)

# arrays
phi = np.zeros(NUM_SPOINTS)
dphi = np.zeros(NUM_SPOINTS)
sigma = np.zeros(NUM_SPOINTS)
m = np.zeros(NUM_SPOINTS)
p = np.zeros(NUM_SPOINTS)
rs = np.zeros(NUM_SPOINTS)

for i in range(NUM_SPOINTS):
	rs[i] = rmin + i*dr

# initial data

def rk4(i, phi, dphi, sigma, m, p, omega, integrate_p):

	h = dr
	r = rmin + i*dr

	this_round_integrate_p = integrate_p

	k1phi	= h * f_phi  (r, phi[i], dphi[i], sigma[i], m[i], p[i], omega, integrate_p)
	k1dphi 	= h * f_dphi (r, phi[i], dphi[i], sigma[i], m[i], p[i], omega, integrate_p)
	k1sigma = h * f_sigma(r, phi[i], dphi[i], sigma[i], m[i], p[i], omega, integrate_p)
	k1m 	= h * f_m    (r, phi[i], dphi[i], sigma[i], m[i], p[i], omega, integrate_p)
	k1p = 0
	if integrate_p:
		k1p 	= h * f_p	 (r, phi[i], dphi[i], sigma[i], m[i], p[i], omega, integrate_p)
	
	if integrate_p and p[i] + k1p/2 < 0: this_round_integrate_p = False

	k2phi 	= h * f_phi  (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, p[i] + k1p/2, omega, integrate_p)
	k2dphi 	= h * f_dphi (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, p[i] + k1p/2, omega, integrate_p)
	k2sigma = h * f_sigma(r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, p[i] + k1p/2, omega, integrate_p)
	k2m 	= h * f_m    (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, p[i] + k1p/2, omega, integrate_p)
	k2p = 0
	if integrate_p:
		k2p		= h * f_p    (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, p[i] + k1p/2, omega, integrate_p)

	if integrate_p and p[i] + k2p/2 < 0: this_round_integrate_p = False
	
	k3phi	= h * f_phi  (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, p[i] + k2p/2, omega, integrate_p)
	k3dphi  = h * f_dphi (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, p[i] + k2p/2, omega, integrate_p)
	k3sigma = h * f_sigma(r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, p[i] + k2p/2, omega, integrate_p)
	k3m 	= h * f_m    (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, p[i] + k2p/2, omega, integrate_p)
	k3p = 0
	if integrate_p:
		k3p 	= h * f_p    (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, p[i] + k2p/2, omega, integrate_p)

	if integrate_p and p[i] + k3p < 0: this_round_integrate_p = False

	k4phi 	= h * f_phi  (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, p[i] + k3p, omega, integrate_p)
	k4dphi  = h * f_dphi (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, p[i] + k3p, omega, integrate_p)
	k4sigma = h * f_sigma(r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, p[i] + k3p, omega, integrate_p)
	k4m 	= h * f_m    (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, p[i] + k3p, omega, integrate_p)
	k4p = 0
	if integrate_p:
		k4p 	= h * f_p    (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, p[i] + k3p, omega, integrate_p)

	phi[i+1]   = phi[i]   + (k1phi   + 2*k2phi   + 2*k3phi   + k4phi)/6
	dphi[i+1]  = dphi[i]  + (k1dphi  + 2*k2dphi  + 2*k3dphi  + k4dphi)/6
	sigma[i+1] = sigma[i] + (k1sigma + 2*k2sigma + 2*k3sigma + k4sigma)/6
	m[i+1]     = m[i]     + (k1m     + 2*k2m     + 2*k3m     + k4m)/6
	if integrate_p:
		p[i+1] = p[i]     + (k1p     + 2*k2p     + 2*k3p     + k4p)/6
	if not this_round_integrate_p and integrate_p: # started off integrating p and had to stop
		p[i+1] = -1

# def rk4(i, phi, dphi, sigma, m, omega):

# 	h = dr
# 	r = rmin + i*dr

# 	k1phi	= h * f_phi  (r, phi[i], dphi[i], sigma[i], m[i], 0, omega)
# 	k1dphi 	= h * f_dphi (r, phi[i], dphi[i], sigma[i], m[i], 0, omega)
# 	k1sigma = h * f_sigma(r, phi[i], dphi[i], sigma[i], m[i], 0, omega)
# 	k1m 	= h * f_m    (r, phi[i], dphi[i], sigma[i], m[i], 0, omega)

# 	k2phi 	= h * f_phi  (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, 0, omega)
# 	k2dphi 	= h * f_dphi (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, 0, omega)
# 	k2sigma = h * f_sigma(r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, 0, omega)
# 	k2m 	= h * f_m    (r + h/2, phi[i] + k1phi/2, dphi[i] + k1dphi/2, sigma[i] + k1sigma/2, m[i] + k1m/2, 0, omega)

# 	k3phi	= h * f_phi  (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, 0, omega)
# 	k3dphi  = h * f_dphi (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, 0, omega)
# 	k3sigma = h * f_sigma(r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, 0, omega)
# 	k3m 	= h * f_m    (r + h/2, phi[i] + k2phi/2, dphi[i] + k2dphi/2, sigma[i] + k2sigma/2, m[i] + k2m/2, 0, omega)
	
# 	k4phi 	= h * f_phi  (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, 0, omega)
# 	k4dphi  = h * f_dphi (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, 0, omega)
# 	k4sigma = h * f_sigma(r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, 0, omega)
# 	k4m 	= h * f_m    (r + h, phi[i] + k3phi, dphi[i] + k3dphi, sigma[i] + k3sigma, m[i] + k3m, 0, omega)

# 	phi[i+1]   = phi[i]   + (k1phi   + 2*k2phi   + 2*k3phi   + k4phi)/6
# 	dphi[i+1]  = dphi[i]  + (k1dphi  + 2*k2dphi  + 2*k3dphi  + k4dphi)/6
# 	sigma[i+1] = sigma[i] + (k1sigma + 2*k2sigma + 2*k3sigma + k4sigma)/6
# 	m[i+1]     = m[i]     + (k1m     + 2*k2m     + 2*k3m     + k4m)/6

phi[0], dphi[0], sigma[0], m[0], p[0] = phi0, dphi0, sigma0, m0, p0

for i in range(1):
	# omega = (omega_upper + omega_lower)/2
	omega = 1.12
	integrate_p = True
	print(f"omega = {omega}")

	for i in range(NUM_SPOINTS-1):

		rk4(i, phi, dphi, sigma, m, p, omega, integrate_p)
		# if integrate_p:
		# 	rk4_p(i, phi, dphi, sigma, m, p, omega)
		# else:
		# 	rk4(i, phi, dphi, sigma, m, omega)

		if phi[i+1] > phi_tol_upper: 
			omega_lower = omega
			print("Phi diverged.")
			break
		elif phi[i+1] < phi_tol_lower: 
			omega_upper = omega
			print("Phi went below 0.")
			break
		elif dphi[i+1] > dphi_tol_upper: 
			omega_lower = omega
			print("dphi diverged")
			break
		elif integrate_p and p[i+1] < p_tol_lower:
			print(i)
			p[i+1] = 0

			print(f"P finished integrating.")
			integrate_p = False
			rk4(i, phi, dphi, sigma, m, p, omega, integrate_p)
			break
			# continue

plt.figure(1)
plt.plot(rs,dphi)
plt.title("$\\varphi'$")
plt.figure(2)
plt.plot(rs,phi)
plt.title("$\\varphi$")
# plt.figure(3)
plt.plot(rs,p)
plt.yscale("log")
plt.title("$P$")
# %%
# plt.xlim([5,20])

# %%
