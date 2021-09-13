# Joe Nyhan, 15 July 2021
# static solutions for bosonic stars, a complex scalar field

#%%

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit

# ========== PARAMETERS

rmin = 0.000001
rmax = 1000
dr = 0.1

eos_UR = 0
eos_polytrope = 1
eos_SLy = 0

to_file = True 
dr_file = 0.02
rmax_file = 5000
rmin_file = 1e-6


# initial conditions (constant ones listed below)
phi0 = 1e-3
p0 = 1e-3
# omega_hat0 = .80262
omega_hat0 = .81
omega_upper = 2
omega_lower = .5
its = 75

mu = 1
Lambda = 0

phi_tol_lower = 0
phi_tol_upper = 2
dphi_tol_upper = 2
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
	if eos_UR:
		return p/(Gamma-1)
	elif eos_polytrope:
		return (p/K)**(1/Gamma)
	elif eos_SLy:
		f = interp1d(SLy_rhos, SLy_ps)
		return f(p)
# ========== EQUATIONS

# @njit
def Tf_tt(r,y,omega_hat, integrate_p):
	phi, dphi, sigma_hat, m, p = y
	return -rho_from_P(p)

# @njit
def Tf_rr(r,y,omega_hat, integrate_p):
	phi, dphi, sigma_hat, m, p = y
	return p

# @njit
def Tb_tt(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	return - (omega_hat**2/(N*sigma_hat**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


# @njit
def Tb_rr(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	return + (omega_hat**2/(N*sigma_hat**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

# @njit
def alpha(r, y, omega_hat, integrate_p):
	"""
	alpha'/alpha
	"""
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
		T = Tb_rr(r, y, omega_hat, integrate_p) + Tf_rr(r, y, omega_hat, integrate_p)
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
		T = Tb_rr(r, y, omega_hat, integrate_p)
	N = 1 - 2*m/r

	return + (4*np.pi*r/N) * T + (m/(r**2*N))

# @njit
def a(r, y, omega_hat, integrate_p):
	"""
	a'/a
	"""
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
		T = Tb_tt(r, y, omega_hat, integrate_p) + Tf_tt(r, y, omega_hat, integrate_p)
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
		T = Tb_tt(r, y, omega_hat, integrate_p)
	N = 1 - 2*m/r

	return - (4*np.pi*r/N) * T - (m/(r**2*N))

# @njit
def dm_dr(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
		T = Tb_tt(r, y, omega_hat, integrate_p) + Tf_tt(r, y, omega_hat, integrate_p)
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
		T = Tb_tt(r, y, omega_hat, integrate_p)
	return -4*np.pi*r**2*T

# @njit
def dsigma_dr(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat, integrate_p)
	a_ = a(r, y, omega_hat, integrate_p)
	return sigma_hat*(a_+alpha_)

# @njit
def dphi_dr(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	return dphi

# @njit
def ddphi_dr(r, y, omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat, integrate_p)
	a_ = a(r, y, omega_hat, integrate_p)
	N = 1 - 2*m/r
	
	first_term = dphi*(a_ - alpha_-2/r)
	second_term = - omega_hat**2/(N**2*sigma_hat**2) * phi
	third_term = (1/N) * (mu**2*phi + 2*Lambda*phi**3)

	return first_term + second_term + third_term

# @njit
def dp_dr(r,y,omega_hat, integrate_p):
	if integrate_p:
		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y
	alpha_ = alpha(r,y,omega_hat, integrate_p)
	rho = rho_from_P(p)
	return - alpha_ * (rho + p)

# ========== EVOLUTION

# constant initial conditons
dphi0 = 0
m0 = 0
sigma_hat0 = 1


# @njit
def fs(r,y,omega_hat, integrate_p):
	if integrate_p:

		phi, dphi, sigma_hat, m, p = y
	elif not integrate_p:
		phi, dphi, sigma_hat, m = y

	fphi = dphi_dr(r,y,omega_hat, integrate_p)
	fdphi = ddphi_dr(r,y,omega_hat, integrate_p)
	fsigma_hat = dsigma_dr(r,y,omega_hat, integrate_p)
	fm = dm_dr(r,y,omega_hat, integrate_p)
	if integrate_p:
		fp = dp_dr(r, y, omega_hat, integrate_p)
	
	if integrate_p:
		return fphi, fdphi, fsigma_hat, fm, fp
	elif not integrate_p:
		return fphi, fdphi, fsigma_hat, fm

NUM_STEPS = int((rmax - rmin)/dr)
NUM_STORED = 100


# phi = np.zeros(NUM_STORED)
# dphi = np.zeros(NUM_STORED)
# sigma = np.zeros(NUM_STORED)
# m = np.zeros(NUM_STORED)


omega_last = 0
omega = 0
# for ct in range(1):
for ct in range(its):
	omega_last = omega
	omega = (omega_upper + omega_lower)/2

	if omega == omega_last: break
	print(f"{omega}")

	new_omega = False
	integrate_p = True
	skip = False

	phi = [phi0]
	dphi = [dphi0]
	sigma = [sigma_hat0]
	m = [m0]
	p = [p0]
	rs = [rmin]

	for i in range(NUM_STEPS):
		# print(i)
		r = rmin + i*dr
		
		if integrate_p:
			ics = [phi[i], dphi[i], sigma[i], m[i], p[i]]
		elif not integrate_p:
			i = i-1
			ics = [phi[i], dphi[i], sigma[i], m[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[omega, integrate_p])

		if integrate_p:
			phi_, dphi_, sigma_, m_, p_ = sol.y
		if not integrate_p:
			phi_, dphi_, sigma_, m_ = sol.y
		for k in range(len(phi_)):
			if integrate_p and p_[k] < p_tol_lower:
				integrate_p = False
				skip = True
				break
			elif phi_[k] > phi_tol_upper:
				# print("Phi upper bound.")
				omega_lower = omega
				new_omega = 1
				break
			elif phi_[k] < phi_tol_lower:
				# print("Phi lower bound.")
				omega_upper = omega
				new_omega = 1
				break
			elif np.abs(dphi_[k]) > dphi_tol_upper:
				# print("dphi upper bound.")
				omega_lower = omega
				new_omega = 1
				break
		
		if skip:
			skip = False
			continue
		elif new_omega:
			# print("new omega")
			break
		else:
			if integrate_p:
				p.append(p_[-1])
			else:
				p.append(0)
			phi.append(phi_[-1])
			dphi.append(dphi_[-1])
			sigma.append(sigma_[-1])
			m.append(m_[-1])
			rs.append(r+dr)
		
		if i == NUM_STEPS-1 and phi[-1] > 0: omega_lower = omega
		# if i == NUM_STEPS-1 and phi[-1] < 0: omega_lower -= .001

plt.plot(rs[:len(p)],p)
plt.plot(rs, phi)
plt.yscale("log")

rs = np.array(rs)
phi = np.array(phi)
dphi = np.array(dphi)
sigma = np.array(sigma)
m = np.array(m)

if to_file:
	if eos_polytrope: eos = f"polytrope_K{K:.1f}_gamma{Gamma:.1f}"
	if eos_SLy: eos = "SLy"
	s = f"../input/{eos}_p{p0:.8f}_vphi{phi0:.8f}_lam{Lambda:.3f}_dr{dr_file:.3f}"

	p_out = open(f"{s}_P.txt", "w")
	rho_out = open(f"{s}_rho.txt", "w")
	varphi_out = open(f"{s}_varphi.txt", "w")
	X_out = open(f"{s}_X.txt", "w")
	Y_out = open(f"{s}_Y.txt", "w")


	def Y(r, phi, sigma, m, omega): 
		N = 1 - 2*m/r
		return (omega*phi)/(N*sigma)

	ys = []
	for i in range(len(rs)):
		ys.append(Y(rs[i], phi[i], sigma[i], m[i], omega))

	fp = interp1d(rs, p)
	fvarphi = interp1d(rs, phi)
	fX = interp1d(rs, dphi)
	fY = interp1d(rs, ys)

	min_idx = np.argmin(phi)
	min_write = rs[min_idx]

	NUM_FILE_POINTS = int((rmax_file - rmin_file)/dr_file)

	for i in range(NUM_FILE_POINTS-1):
		r = rmin_file + i*dr_file
		if r < min_write:
			p_out.write(f"{fp(r):.16e}\n")
			rho_out.write(f"{rho_from_P(fp(r)):.16e}\n")
			varphi_out.write(f"{fvarphi(r):.16e}\n")
			X_out.write(f"{fX(r):.16e}\n")
			Y_out.write(f"{fY(r):.16e}\n")
		if r > min_write:
			p_out.write(f"{0:.16e}\n")
			rho_out.write(f"{0:.16e}\n")
			varphi_out.write(f"{0:.16e}\n")
			X_out.write(f"{0:.16e}\n")
			Y_out.write(f"{0:.16e}\n")
		
	
	p_out.close()
	rho_out.close()
	varphi_out.close()
	X_out.close()
	Y_out.close()

	


# %%
idx = np.argmin(phi)
plt.plot(rs[:idx], phi[:idx])
# %%
