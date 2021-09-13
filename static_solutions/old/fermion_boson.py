# Joe Nyhan, 15 July 2021
# static solutions for bosonic stars, a complex scalar field

#%%

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import njit


# ========== EOS

eos_UR = 0
eos_polytrope = 1
eos_SLy = 0

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

# ========== PARAMETERS

rmin = 0.000001
rmax = 100
dr = 0.02


# initial conditions (constant ones listed below)
phi0 = 1e-3
p0 = 1e-3

# omega_hat0 = .80262
omega_hat0 = .81
omega_hat0_min = .5
omega_hat0_max = 2.5
its = 100

mu = 1.122089
Lambda = 0

phi_tol_lower = 0
phi_tol_upper = 2
dphi_tol_upper = 5
p_tol_lower = 1e-10


# ========== EQUATIONS, no P

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

@njit
def fs(r,y,omega_hat):
	phi, dphi, sigma_hat, m = y

	fphi = dphi_dr(r,y,omega_hat)
	fdphi = ddphi_dr(r,y,omega_hat)
	fsigma_hat = dsigma_dr(r,y,omega_hat)
	fm = dm_dr(r,y,omega_hat)

	return fphi, fdphi, fsigma_hat, fm

# ========== EQUATIONS, with P

# @njit
def Tb_tt_p(r, y, omega_hat):

	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	return - (omega_hat**2/(N*sigma_hat**2))*phi**2 - N*dphi**2 - mu**2*phi**2 - Lambda*phi**4


# @njit
def Tb_rr_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	return + (omega_hat**2/(N*sigma_hat**2))*phi**2 + N*dphi**2 - mu**2*phi**2 - Lambda*phi**4

def Tf_tt_p(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return -rho_from_P(p)

def Tf_rr_p(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return p


# @njit
def alpha_p(r, y, omega_hat):
	"""
	alpha'/alpha
	"""
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_rr_p(r, y, omega_hat) + Tf_rr_p(r,y,omega_hat)

	return + (4*np.pi*r/N) * T + (m/(r**2*N))

# @njit
def a_p(r, y, omega_hat):
	"""
	a'/a
	"""
	phi, dphi, p, sigma_hat, m = y
	N = 1 - 2*m/r
	T = Tb_tt_p(r, y, omega_hat) + Tf_tt_p(r,y,omega_hat)

	return - (4*np.pi*r/N) * T - (m/(r**2*N))

# @njit
def dm_dr_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	T = Tb_tt_p(r, y, omega_hat) + Tf_tt_p(r,y,omega_hat)
	return -4*np.pi*r**2*T

# @njit
def dsigma_dr_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha_p(r, y, omega_hat)
	a_ = a_p(r, y, omega_hat)
	return sigma_hat*(a_+alpha_)

# @njit
def dphi_dr_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	return dphi

# @njit
def ddphi_dr_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha_p(r, y, omega_hat)
	a_ = a_p(r, y, omega_hat)
	N = 1 - 2*m/r
	
	first_term = dphi*(a_ - alpha_-2/r)
	second_term = - omega_hat**2/(N**2*sigma_hat**2) * phi
	third_term = (1/N) * (mu**2*phi + 2*Lambda*phi**3)

	return first_term + second_term + third_term

def dp_dr_p(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	alpha_ = alpha_p(r,y,omega_hat)
	rho = rho_from_P(p)
	return - alpha_ * (rho + p)

# @njit
def fs_p(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y

	fphi = dphi_dr_p(r,y,omega_hat)
	fdphi = ddphi_dr_p(r,y,omega_hat)
	fp = dp_dr_p(r,y,omega_hat)
	fsigma_hat = dsigma_dr_p(r,y,omega_hat)
	fm = dm_dr_p(r,y,omega_hat)

	return fphi, fdphi, fp, fsigma_hat, fm

# ========== EVOLUTION

term_phi_low = 0
term_phi_up = 0
term_dphi = 0
term_p = 0

# constant initial conditons
dphi0 = 0
m0 = 0
sigma_hat0 = .8

# termination with p
def term_phi_upper_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	diff = phi_tol_upper - phi
	if diff < 0: term_phi_up = 1
	return diff

term_phi_upper_p.terminal = True
term_phi_upper_p.direction = 0

def term_phi_lower_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	diff = phi - phi_tol_lower 
	if diff < 0: term_phi_low = 1
	return diff

term_phi_lower_p.terminal = True
term_phi_lower_p.direction = 0

def term_dphi_upper_p(r, y, omega_hat):
	phi, dphi, p, sigma_hat, m = y
	diff = dphi_tol_upper - dphi
	if diff < 0: term_dphi = 1
	return diff

term_dphi_upper_p.terminal = True
term_dphi_upper_p.direction = 0

def term_p_lower_p(r,y,omega_hat):
	phi, dphi, p, sigma_hat, m = y
	diff = p - p_tol_lower
	if diff < 0: term_p = True
	return diff

term_p_lower_p.terminal = True
term_p_lower_p.direction = 0

# termination without p

def term_phi_upper(r, y, omega_hat):
	phi, dphi,sigma_hat, m = y
	diff = phi_tol_upper - phi
	if diff < 0: term_phi_up = 1
	return diff

term_phi_upper.terminal = True
term_phi_upper.direction = 0

def term_phi_lower(r, y, omega_hat):
	# print("lower phi")
	phi, dphi,sigma_hat, m = y
	return phi - phi_tol_lower

term_phi_lower.terminal = True
term_phi_lower.direction = 0

def term_dphi_upper(r, y, omega_hat):
	# print("upper dphi")
	phi, dphi,sigma_hat, m = y
	return dphi - dphi_tol_upper

term_dphi_upper.terminal = True
term_dphi_upper.direction = 0
	

# ========== FINDING OMEGA

# rs = np.arange(rmin,rmax,dr)
omega_lower = omega_hat0_min
omega_upper = omega_hat0_max
# omega0 = 0
last_omega = 0

sol = 1

# with p
for i in range(its):
	omega0 = (omega_upper + omega_lower)/2
	if omega0 == last_omega: break #machine precision reached
	print(f"{omega0}")

	sol = solve_ivp(fs_p, [rmin,rmax], [phi0,dphi0,p0,sigma_hat0,m0], method = 'RK45', events=[term_phi_lower_p, term_phi_upper_p, term_dphi_upper_p, term_p_lower_p], atol = 1e-10, rtol = 1e-10, args=[omega0])

	sol_phi, sol_dphi, sol_p, sol_sigma_hat, sol_m = sol.y
	ps = sol_p
	ts = sol.t

	if term_p:
		print("entered")
	# if int(np.log10(sol_p[-1])) == int(np.log10(p_tol_lower)): # this means that it terminated on pressure
		term_p = False
		phi0_n, dphi0_n, p_not, sigma_hat0_n, m0_n = sol.y[:,-2]
		rmin = sol.t[-2]
		sol = solve_ivp(fs, [rmin,rmax], [phi0_n,dphi0_n,sigma_hat0_n,m0_n], method = 'RK45', events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-10, rtol = 1e-10, args=[omega0])
	
	diff = sol_phi[-2] - sol_phi[-3]

	# < 0: crossed zero;
	if diff < 0: omega_upper = omega0
	# > 0: diverged 
	elif diff > 0: omega_lower = omega0
	last_omega = omega0


rmin = 0.000001
rs = np.arange(rmin, rmax, dr)
sol1 = solve_ivp(fs_p, [rmin,rmax], [phi0,dphi0,p0,sigma_hat0,m0], method = 'RK45', t_eval = np.arange(rmin,rmax,dr), events=[term_phi_lower_p, term_phi_upper_p, term_dphi_upper_p, term_p_lower_p], atol = 1e-10, rtol = 1e-10, args=[omega0])
ps = sol1.y[2]
phi0, dphi0, p_not, sigma_hat0, m0 = sol1.y[:,-2]
rmin = sol1.t[-2]
sol2 = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', t_eval = np.arange(rmin,rmax,dr), events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-10, rtol = 1e-10, args=[omega0])

plt.plot(sol1.t,sol1.y[2])
plt.plot(sol1.t,sol1.y[0])
plt.plot(sol2.t,sol2.y[0])

# ps = sol.y[2]
# # plt.figure(1)
# plt.plot(sol.t,ps)
# # plt.figure(2)
# rmin = sol.t[-1]
# ts = sol.t
# phi0, dphi0, p_not, sigma_hat0, m0 = sol.y[:,-1]
# s_phi, s_dphi, s_p, s_sigma_hat, s_m = sol.y

# omega0 = 1.592919555

# omega_lower = 1
# omega_upper = 2

# for i in range(its):
# 	omega0 = (omega_upper + omega_lower)/2
# 	print(f"{omega0}")

# 	sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-10, rtol = 1e-10, args=[omega0])

# 	sol_phi, sol_dphi, sol_sigma_hat, sol_m = sol.y
# 	diff = sol_phi[-1] - sol_phi[-2]

# 	# < 0: crossed zero;
# 	if diff < 0: omega_upper = omega0
# 	# > 0: diverged 
# 	elif diff > 0: omega_lower = omega0

# rs = np.arange(rmin, rmax, dr)
# sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', events=[term_phi_lower, term_phi_upper, term_dphi_upper], t_eval=rs, atol = 1e-10, rtol = 1e-10, args=[omega0])

# rs = np.arange(rmin, rmax, dr)
# sol = solve_ivp(fs, [rmin,rmax], [phi0,dphi0,sigma_hat0,m0], method = 'RK45', t_eval = rs, events=[term_phi_lower, term_phi_upper, term_dphi_upper], atol = 1e-10, rtol = 1e-10, args=[omega0])

# plt.ylim(-2,2)
# plt.xlim(0,175)
# print(f"{omega0=}")
plt.yscale("log")
plt.xlim([0,10])
# plt.plot(sol.t,sol.y[0])
# plt.plot(ts,s_phi)
# plt.xlim([0,10])
	
 # %%
# plt.scatter(sol.t,sol.y[0])
plt.plot(sol1.t,sol1.y[0])

# plt.yscale("log")
# plt.ylim(1e-10,1e-3)
# %%
plt.plot(ps)
# %%
