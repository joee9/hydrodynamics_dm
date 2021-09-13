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

to_file = False
dr_file = 0.02
rmax_file = 5000
rmin_file = 1e-6


# initial conditions (constant ones listed below)
phi0 = 10**(-3.5)
p0 = 1e-4
At_lower = 0
At_upper = 1
A_tol = 1e-7

# omega_hat0 = .80262
its = 75

mu = 1.122089
Lambda = 0
gs = .65
g = gs * np.sqrt(8*np.pi)

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

# if integrate_fermions and integrate_bosons:
# 	sigma, m, phi, dphi, A_t, dA_t, p = y
# elif integrate_fermions and not integrate_bosons:
# 	sigma, m, p = y
# elif not integrate_fermions and integrate_bosons:
# 	sigma, m, phi, dphi, A_t, dA_t = y

def Tf_tt(r, y, integrate_fermions, integrate_bosons):
	if not integrate_fermions and integrate_bosons:
		print("Invalid: Tf_tt: pressure not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif integrate_fermions and not integrate_bosons:
		sigma, m, p = y

	return -rho_from_P(p)

def Tf_rr(r, y, integrate_fermions, integrate_bosons):
	if not integrate_fermions and integrate_bosons:
		print("Invalid: Tf_rr: pressure not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif integrate_fermions and not integrate_bosons:
		sigma, m, p = y

	return p

def Tb_tt(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and not integrate_bosons:
		print("Invalid: Tb_tt: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y

	N = 1 - 2*m/r

	first = -N*dphi**2
	second = -(mu**2 * g**2 * A_t**2 * phi**2)/(N*sigma**2)
	third = -mu**2*phi**2
	fourth = -Lambda*phi**4
	fifth = -dA_t**2/(2*sigma**2)

	return first + second + third + fourth + fifth

def Tb_rr(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and not integrate_bosons:
		print("Invalid: Tb_tt: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y

	N = 1 - 2*m/r

	first = +N*dphi**2
	second = +(mu**2 * g**2 * A_t**2 * phi**2)/(N*sigma**2)
	third = -mu**2*phi**2
	fourth = -Lambda*phi**4
	fifth = -dA_t**2/(2*sigma**2)

	return first + second + third + fourth + fifth

def f_sigma(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons) + Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons) + Tb_tt(r, y, integrate_fermions, integrate_bosons)

	elif integrate_fermions and not integrate_bosons:
		sigma, m, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons)

	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
		Ttot_rr = Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tb_tt(r, y, integrate_fermions, integrate_bosons)

	N = 1 - 2*m/r

	return 4*np.pi*r*sigma/N * (Ttot_rr - Ttot_tt)

def f_m(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons) + Tb_tt(r, y, integrate_fermions, integrate_bosons)

	elif integrate_fermions and not integrate_bosons:
		sigma, m, p = y
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons)

	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
		Ttot_tt = Tb_tt(r, y, integrate_fermions, integrate_bosons)

	return -4*np.pi*r**2*Ttot_tt

def f_phi(r, y, integrate_fermions, integrate_bosons):
	if integrate_fermions and not integrate_bosons:
		print("Invalid: f_phi: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
	
	return dphi

def f_dphi(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and not integrate_bosons:
		print("Invalid: f_dphi: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons) + Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons) + Tb_tt(r, y, integrate_fermions, integrate_bosons)
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
		Ttot_rr = Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tb_tt(r, y, integrate_fermions, integrate_bosons)
	
	N = 1 - 2*m/r

	braces = -(2/r * ((4*np.pi*r)/N) * (Ttot_rr + Ttot_tt) + (2*m)/(N*r**2)) * dphi
	second = - (1/N)*((mu**2*g**2*A_t**2)/(N*sigma**2) - mu**2 - 2*Lambda*phi**2)*phi

	return braces + second

def f_A_t(r, y, integrate_fermions, integrate_bosons):
	if integrate_fermions and not integrate_bosons:
		print("Invalid: f_A_t: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
	
	return dA_t

def f_dA_t(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and not integrate_bosons:
		print("Invalid: f_phi: boson sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons) + Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tf_tt(r, y, integrate_fermions, integrate_bosons) + Tb_tt(r, y, integrate_fermions, integrate_bosons)
	elif not integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t = y
		Ttot_rr = Tb_rr(r, y, integrate_fermions, integrate_bosons)
		Ttot_tt = Tb_tt(r, y, integrate_fermions, integrate_bosons)
	
	N = 1 - 2*m/r

	braces = ((4*np.pi*r/N)*(Ttot_rr - Ttot_tt) - 2/r) * dA_t
	second = (2*mu**2*g**2/N) * A_t * phi**2

	return braces + second

def f_p(r, y, integrate_fermions, integrate_bosons):

	if not integrate_fermions and integrate_bosons:
		print("Invalid: f_p: fermion sector not found")
		return
	elif integrate_fermions and integrate_bosons:
		sigma, m, phi, dphi, A_t, dA_t, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons) + Tb_rr(r, y, integrate_fermions, integrate_bosons)
	elif integrate_fermions and not integrate_bosons:
		sigma, m, p = y
		Ttot_rr = Tf_rr(r, y, integrate_fermions, integrate_bosons)

	N = 1 - 2*m/r

	alpha = (4*np.pi*r**3*Ttot_rr + m)/(r**2*N)
	rho = rho_from_P(p)

	return - alpha * (rho + p)


# ========== EVOLUTION

# constant initial conditons
dphi0 = 0
m0 = 0
sigma_hat0 = 1
dA_t0 = 0


def fs(r, y, integrate_fermions, integrate_bosons):

	if integrate_fermions and integrate_bosons:
		# sigma, m, phi, dphi, A_t, dA_t, p = y
		fsigma 	= f_sigma(r, y, integrate_fermions, integrate_bosons)
		fm 		= f_m(r, y, integrate_fermions, integrate_bosons)
		fphi 	= f_phi(r, y, integrate_fermions, integrate_bosons)
		fdphi 	= f_dphi(r, y, integrate_fermions, integrate_bosons)
		fA_t 	= f_A_t(r, y, integrate_fermions, integrate_bosons)
		fdA_t 	= f_dA_t(r, y, integrate_fermions, integrate_bosons)
		fp 		= f_p(r, y, integrate_fermions, integrate_bosons)
		return fsigma, fm, fphi, fdphi, fA_t, fdA_t, fp

	elif integrate_fermions and not integrate_bosons:
		# sigma, m, p = y
		fsigma 	= f_sigma(r, y, integrate_fermions, integrate_bosons)
		fm 		= f_m(r, y, integrate_fermions, integrate_bosons)
		fp 		= f_p(r, y, integrate_fermions, integrate_bosons)
		return fsigma, fm, fp

	elif not integrate_fermions and integrate_bosons:
		# sigma, m, phi, dphi, A_t, dA_t = y
		fsigma 	= f_sigma(r, y, integrate_fermions, integrate_bosons)
		fm 		= f_m(r, y, integrate_fermions, integrate_bosons)
		fphi 	= f_phi(r, y, integrate_fermions, integrate_bosons)
		fdphi 	= f_dphi(r, y, integrate_fermions, integrate_bosons)
		fA_t 	= f_A_t(r, y, integrate_fermions, integrate_bosons)
		fdA_t 	= f_dA_t(r, y, integrate_fermions, integrate_bosons)
		return fsigma, fm, fphi, fdphi, fA_t, fdA_t


NUM_STEPS = int((rmax - rmin)/dr)
NUM_STORED = 100


# phi = np.zeros(NUM_STORED)
# dphi = np.zeros(NUM_STORED)
# sigma = np.zeros(NUM_STORED)
# m = np.zeros(NUM_STORED)


At_last = 0
At0 = 0
# for ct in range(1):
for ct in range(its):
	At_last = At0
	# omega = (omega_upper + omega_lower)/2
	At0 = (At_upper + At_lower)/2
	# At0 = .325

	# if At0 == At_last: break
	if np.abs(At0 - At_last) < A_tol: break
	print(f"{At0 = }")

	new_At = False
	integrate_fermions = True
	integrate_bosons = True
	skip = False

	sigma = [sigma_hat0]
	m = [m0]
	phi = [phi0]
	dphi = [dphi0]
	A_t = [At0]
	dA_t = [dA_t0]
	p = [p0]
	rs = [rmin]

	for i in range(NUM_STEPS):
		# print(i)
		r = rmin + i*dr
		
		if integrate_fermions and integrate_bosons:
			# ics = [phi[i], dphi[i], sigma[i], m[i], p[i]]
			ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i], p[i]]
		elif integrate_fermions and not integrate_bosons:
			i = i-1
			ics = [sigma[i], m[i], p[i]]
		elif not integrate_fermions and integrate_bosons:
			i = i-1
			ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[integrate_fermions, integrate_bosons])

		if integrate_fermions and integrate_bosons:
			sigma_, m_, phi_, dphi_, A_t_, dA_t_, p_ = sol.y
		elif integrate_fermions and not integrate_bosons:
			sigma_, m_, p_ = sol.y
		elif not integrate_fermions and integrate_bosons:
			sigma_, m_, phi_, dphi_, A_t_, dA_t_ = sol.y

		for k in range(len(phi_)):
			if integrate_fermions and p_[k] < p_tol_lower:
				integrate_fermions = False
				fermion_idx = i-1
				skip = True
				break
			# elif integrate_bosons and phi_[k] < phi_tol_lower:
			# 	integrate_bosons = False
			# 	skip = True
			# 	break
			elif phi_[k] > phi_tol_upper:
				print("Phi upper bound.")
				At_lower = At0
				new_At = 1
				break
			elif phi_[k] < phi_tol_lower:
				print("Phi lower bound.")
				At_upper = At0
				new_At = 1
				break
			elif np.abs(dphi_[k]) > dphi_tol_upper:
				print("dphi upper bound.")
				At_lower = At0
				new_At = 1
				break
		
		if skip:
			skip = False
			continue
		elif new_At:
			# print("new omega")
			break
		else:
			if integrate_fermions and integrate_bosons:
				sigma.append(sigma_[-1])
				m.append(m_[-1])
				phi.append(phi_[-1])
				dphi.append(dphi_[-1])
				A_t.append(A_t_[-1])
				dA_t.append(dA_t_[-1])
				p.append(p_[-1])
				rs.append(r+dr)
			if integrate_fermions and not integrate_bosons:
				sigma.append(sigma_[-1])
				m.append(m_[-1])
				phi.append(0)
				dphi.append(0)
				A_t.append(0)
				dA_t.append(0)
				p.append(p_[-1])
				rs.append(r+dr)
			if not integrate_fermions and integrate_bosons:
				sigma.append(sigma_[-1])
				m.append(m_[-1])
				phi.append(phi_[-1])
				dphi.append(dphi_[-1])
				A_t.append(A_t_[-1])
				dA_t.append(dA_t_[-1])
				p.append(0)
				rs.append(r+dr)
		
		if i == NUM_STEPS-1 and phi[-1] > 0: At_lower = At0
		# if i == NUM_STEPS-1 and phi[-1] < 0: omega_lower -= .001
	
# actual integration for plotting and writing

boson_idx = np.argmin(np.array(phi))
if boson_idx > fermion_idx:
	boson_first = False
	fermion_first = True
else: 
	boson_first = True
	fermion_first = False

integrate_fermions = True
integrate_bosons = True

sigma = [sigma_hat0]
m = [m0]
phi = [phi0]
dphi = [dphi0]
A_t = [At0]
dA_t = [dA_t0]
p = [p0]
rs = [rmin]

if boson_first:
	for i in range(boson_idx):
		# print(i)
		r = rmin + i*dr
		
		ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i], p[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[integrate_fermions, integrate_bosons])

		sigma_, m_, phi_, dphi_, A_t_, dA_t_, p_ = sol.y

		sigma.append(sigma_[-1])
		m.append(m_[-1])
		phi.append(phi_[-1])
		dphi.append(dphi_[-1])
		A_t.append(A_t_[-1])
		dA_t.append(dA_t_[-1])
		p.append(p_[-1])
		rs.append(r+dr)
		
	integrate_bosons = False

	for i in range(boson_idx-1, fermion_idx):
		# print(i)
		r = rmin + i*dr
		
		ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[integrate_fermions, integrate_bosons])

		sigma_, m_, p_ = sol.y
			
		sigma.append(sigma_[-1])
		m.append(m_[-1])
		# phi.append(0)
		# dphi.append(0)
		# A_t.append(0)
		# dA_t.append(0)
		p.append(p_[-1])
		rs.append(r+dr)
		
if fermion_first:
	for i in range(fermion_idx):
		# print(i)
		r = rmin + i*dr
		
		ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i], p[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[integrate_fermions, integrate_bosons])

		sigma_, m_, phi_, dphi_, A_t_, dA_t_, p_ = sol.y

		sigma.append(sigma_[-1])
		m.append(m_[-1])
		phi.append(phi_[-1])
		dphi.append(dphi_[-1])
		A_t.append(A_t_[-1])
		dA_t.append(dA_t_[-1])
		p.append(p_[-1])
		rs.append(r+dr)
		
	integrate_fermions = False

	for i in range(fermion_idx-1, boson_idx):
		# print(i)
		r = rmin + i*dr
		
		ics = [sigma[i], m[i], phi[i], dphi[i], A_t[i], dA_t[i]]

		sol = solve_ivp(fs, [r, r+dr], ics, 
		method = 'RK45', atol = 1e-10, rtol = 1e-10, args=[integrate_fermions, integrate_bosons])

		sigma_, m_, phi_, dphi_, A_t_, dA_t_ = sol.y
			
		sigma.append(sigma_[-1])
		m.append(m_[-1])
		phi.append(phi_[-1])
		dphi.append(dphi_[-1])
		A_t.append(A_t_[-1])
		dA_t.append(dA_t_[-1])
		# p.append(0)
		rs.append(r+dr)


	# for k in range(len(phi_)):
	# 	if integrate_fermions and p_[k] < p_tol_lower:
	# 		integrate_fermions = False
	# 		skip = True
	# 		break
	# 	# elif integrate_bosons and phi_[k] < phi_tol_lower:
	# 	# 	integrate_bosons = False
	# 	# 	skip = True
	# 	# 	break
	# 	elif phi_[k] > phi_tol_upper:
	# 		print("Phi upper bound.")
	# 		# At_lower = At0
	# 		# new_At = 1
	# 		break
	# 	elif phi_[k] < phi_tol_lower:
	# 		print("Phi lower bound.")
	# 		# At_upper = At0
	# 		# new_At = 1
	# 		break
	# 	elif np.abs(dphi_[k]) > dphi_tol_upper:
	# 		print("dphi upper bound.")
	# 		# At_lower = At0
	# 		# new_At = 1
	# 		break


plt.plot(rs[:len(p)],p)
plt.plot(rs, phi)
plt.yscale("log")

rs = np.array(rs)
phi = np.array(phi)
dphi = np.array(dphi)
A_t = np.array(A_t)
dA_t = np.array(dA_t)
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


	ys = []
	# for i in range(len(rs)):
	# 	ys.append(Y(rs[i], phi[i], sigma[i], m[i]))

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
plt.plot(rs[:len(p)],p)
plt.plot(rs, phi)
plt.yscale("log")
plt.ylim([1e-13,1e-2])
