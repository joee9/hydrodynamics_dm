# Joe Nyhan, 15 July 2021
# static solutions for bosonic stars, a complex scalar field

#%%

import sys
import numpy as np; np.set_printoptions(threshold=sys.maxsize)
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numba import njit

# ========== PARAMETERS

rmin = 0.000001
rmax = 250
dr = 0.01


# initial conditions (constant ones listed below)
phi0 = 1e-3
# omega_hat0 = .80262
omega_hat0 = .81
omega_hat0_min = .5
omega_hat0_max = 1.5
its = 100

mu = 1
Lambda = 0

phi_tol_lower = 0
phi_tol_upper = 2
dphi_tol_upper = 1

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

	return np.array([fphi, fdphi, fsigma_hat, fm])

# # termination
# def term_phi_upper(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	return phi - phi_tol_upper

# term_phi_upper.terminal = True
# term_phi_upper.direction = 0

# def term_phi_lower(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	return phi - phi_tol_lower

# term_phi_lower.terminal = True
# term_phi_lower.direction = 0

# def term_dphi_upper(r, y, omega_hat):
# 	phi, dphi, sigma_hat, m = y
# 	return dphi - dphi_tol_upper

# term_dphi_upper.terminal = True
# term_dphi_upper.direction = 0

# # rs = np.arange(rmin,rmax,dr)
# omega_lower = omega_hat0_min
# omega_upper = omega_hat0_max
# omega0 = 0

# NUM_SPOINTS = int((rmax - rmin)/dr)
# phis = [phi0]
# dphis = [dphi0]
# ms = [m0]
# sigmas = [sigma_hat0]
# rs = [rmin]

# NUM_STORED = 10

# phi = np.zeros(NUM_STORED)
# dphi = np.zeros(NUM_STORED)
# sigma = np.zeros(NUM_STORED)
# m = np.zeros(NUM_STORED)

# for i in range(its):
for i in range(1):
	# omega0 = (omega_upper + omega_lower)/2
	# omega0 = 1.0033
	# omega0 = 1.1
	omega0 = 1.0
	# omega0 = 1.1
	print(f"{omega0}")

	# sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-6, rtol = 1e-6, args=[omega0])

	# sol_phi, sol_dphi, sol_sigma_hat, sol_m = sol.y
	# diff = sol_phi[-1] - sol_phi[-2]
	# RK4
	h = dr
	# for i in range(NUM_SPOINTS):
	# 	r = rmin + i*h

	# 	curr = i % NUM_STORED
	# 	next = (curr + 1) % NUM_STORED

	# 	y = np.array([phi[curr], dphi[curr], sigma[curr], m[curr]])

	# 	k1 = h * fs(r, y, omega0)
	# 	k2 = h * fs(r+h/2, y + k1/2, omega0)
	# 	k3 = h * fs(r+h/2, y + k2/2, omega0)
	# 	k4 = h * fs(r+h, y + k3, omega0)

	# 	phi_, dphi_, sigma_, m_ = y + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

	# 	phi[next], dphi[next], sigma[next], m[next] = phi_, 

	# 	if i % 1000 == 0:
	# 		phi.append(phi_)
	# 		dphi.append(dphi_)
	# 		sigma.append(sigma_)
	# 		m.append(m_)
	# 		rs.append(r+h)

		if phi[i+1] > phi_tol_upper: print(f"phi upper"); omega_lower = omega0; break
		elif phi[i+1] < phi_tol_lower: print(f"phi lower"); omega_upper = omega0; break
		elif dphi[i+1] > dphi_tol_upper: print(f"dphi upper"); omega_lower = omega0; break

	# # < 0: crossed zero;
	# if diff < 0: omega_upper = omega0
	# # > 0: diverged 
	# elif diff > 0: omega_lower = omega0


# sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-6, rtol = 1e-6, args=[omega_hat0])

# rs = np.arange(rmin, rmax, dr)
# sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', t_eval = rs, events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-10, rtol = 1e-10, args=[omega0])

# plt.ylim(-2,2)
# plt.xlim(0,175)
print(f"{omega0=}")

plt.scatter(rs, dphi)
# plt.yscale("log")
	
# %%
# plt.scatter(sol.t,sol.y[0])
phis = [phi[i*100] for i in range(NUM_SPOINTS//1000)
]
plt.plot(phis)
# plt.yscale("log")
# plt.ylim(1e-10,1e-3)
# %%
