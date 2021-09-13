# Joe Nyhan, 15 July 2021
# static solutions for bosonic stars, a complex scalar field

#%%

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit

# ========== PARAMETERS

rmin = 0.000001
rmax = 1000
dr = 0.1


# initial conditions (constant ones listed below)
phi0 = 1e-4
# omega_hat0 = .80262
omega_hat0 = .81
omega_upper = 2
omega_lower = .5
its = 40

mu = 1
Lambda = 0

phi_tol_lower = 0
phi_tol_upper = 2
dphi_tol_upper = 2

# ========== EQUATIONS

@njit
def Tb_tt(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	return - (omega_hat**2/(N*sigma_hat**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


@njit
def Tb_rr(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	return + (omega_hat**2/(N*sigma_hat**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

@njit
def alpha(r, y, omega_hat):
	"""
	alpha'/alpha
	"""
	phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_rr(r, y, omega_hat)

	return + (4*np.pi*r/N) * T + (m/(r**2*N))

@njit
def a(r, y, omega_hat):
	"""
	a'/a
	"""
	phi, dphi, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_tt(r, y, omega_hat)

	return - (4*np.pi*r/N) * T - (m/(r**2*N))

@njit
def dm_dr(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	T = Tb_tt(r, y, omega_hat)
	return -4*np.pi*r**2*T

@njit
def dsigma_dr(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat)
	a_ = a(r, y, omega_hat)
	return sigma_hat*(a_+alpha_)

@njit
def dphi_dr(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	return dphi

@njit
def ddphi_dr(r, y, omega_hat):
	phi, dphi, sigma_hat, m = y
	alpha_ = alpha(r, y, omega_hat)
	a_ = a(r, y, omega_hat)
	N = 1 - 2*m/r
	
	first_term = dphi*(a_ - alpha_-2/r)
	second_term = - omega_hat**2/(N**2*sigma_hat**2) * phi
	third_term = (1/N) * (mu**2*phi + 2*Lambda*phi**3)

	return first_term + second_term + third_term

# ========== EVOLUTION

# constant initial conditons
dphi0 = 0
m0 = 0
sigma_hat0 = 1


@njit
def fs(r,y,omega_hat):
	phi, dphi, sigma_hat, m = y

	fphi = dphi_dr(r,y,omega_hat)
	fdphi = ddphi_dr(r,y,omega_hat)
	fsigma_hat = dsigma_dr(r,y,omega_hat)
	fm = dm_dr(r,y,omega_hat)

	return fphi, fdphi, fsigma_hat, fm

NUM_STEPS = int((rmax - rmin)/dr)
NUM_STORED = 100


# phi = np.zeros(NUM_STORED)
# dphi = np.zeros(NUM_STORED)
# sigma = np.zeros(NUM_STORED)
# m = np.zeros(NUM_STORED)


N = 10
omega_last = 0
omega = 0
# for ct in range(1):
for ct in range(its):
	omega_last = omega
	omega = (omega_upper + omega_lower)/2

	# if omega == omega_last: break
	# omega = 1
	print(f"{omega}")

	new_omega = False

	phi = [phi0]
	dphi = [dphi0]
	sigma = [sigma_hat0]
	m = [m0]
	rs = [rmin]

	for i in range(NUM_STEPS):
		# print(i)
		r = rmin + i*dr

		sol = solve_ivp(fs, [r, r+dr], [phi[i], dphi[i], sigma[i], m[i]], 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[omega])

		phi_, dphi_, sigma_, m_ = sol.y
		for k in range(len(phi_)):
			if phi_[k] > phi_tol_upper:
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
		
		if new_omega:
			# print("new omega")
			break
		else:
			phi.append(phi_[-1])
			dphi.append(dphi_[-1])
			sigma.append(sigma_[-1])
			m.append(m_[-1])
			rs.append(r+dr)
		
		if i == NUM_STEPS-1 and phi[-1] > 0: omega_lower = omega
		# if i == NUM_STEPS-1 and phi[-1] < 0: omega_lower -= .001

plt.plot(rs,phi)
# plt.xlim(0,150)
# plt.ylim(0,.00015)

	

# %%
plt.plot(rs,phi)
plt.title(f"$\\varphi$, $\\omega = 1.1$")
plt.savefig(f"vphi,negative.pdf")
# plt.yscale("log")

# %%
