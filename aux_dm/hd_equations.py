# Joe Nyhan, 30 June 2021
# The equations used for simulation.

# from logging import debug
from numba.core.decorators import jit
from hd_params import *
from aux_dm.hd_eos import *
# NUMBA_DISABLE_JIT = 1

# ========== GRID

@njit
def R(i):
	"""
	gives the grid point for a given index i; here, each R(i) is center of a cell, C_i, with domain [R((i+1)/2),R((i-1)/2)]
	"""
	return (i - NUM_VPOINTS + 0) * dr

# ========== CALCULATE PRIMITIVES

@njit
def V(u, P):
	"""
	from eq. (30); calculates v at a given i, given P and u
	"""
	Pi, Phi = u
	# print(Pi,Phi,P)
	return (Pi - Phi)/(Pi + Phi + 2*P)

@njit
def calcPrims(u, rho0):
	"""
	calculates the primitive values, given the conservative values at a given spatial point
	"""
	rho_ = rho(u, rho0)
	p = P(rho_)
	v = V(u, p)

	return np.array([p,rho_,v])

# ========== TIME EVOLUTION

# fermiionic neutron star matter

@njit
def da_dt(i,alpha,a,cons,sc1,sc2):
	r = R(i)
	if i == NUM_SPOINTS-1: alpha = 1/a
	Ttot_tr = calcTtot_tr(cons, sc1, sc2, a, alpha)
	return -4 * np.pi * r * alpha * a**2 * Ttot_tr 


@njit
def du_dt(i, rho0, u, phi1, phi2, z, a, a_p1h, a_m1h, alpha, alpha_p1h, alpha_m1h, F1_p1h, F1_m1h, F2_p1h, F2_m1h):
	"""
	The time evolution of u
	F1, F2: the F value at a given spatial boundary (still a two component vector)
	"""

	# calculate average alpha value for cell center
	alpha = (alpha_p1h + alpha_m1h)/2

	# r values
	r = R(i)
	r_p1h = R(i+1/2)
	r_m1h = R(i-1/2)

	firstFrac = 3/(r_p1h**3 - r_m1h**3)
	def fbTerm(r,alpha,a,F1):
		return r**2 * alpha * F1 / a

	firstBracket = (fbTerm(r_p1h, alpha_p1h, a_p1h, F1_p1h) - fbTerm(r_m1h, alpha_m1h, a_m1h, F1_m1h))
	
	def sbTerm(alpha, a, F2):
		return alpha * F2 / a

	secondBracket = (sbTerm(alpha_p1h,a_p1h,F2_p1h) - sbTerm(alpha_m1h,a_m1h,F2_m1h))

	s = S(u, r, phi1, phi2, z, alpha, a, rho0)

	return -firstFrac * firstBracket - 1/dr * secondBracket + s

@njit
def S(u,r,phi1,phi2,z,alpha,a,rho0):
	"""
	the value of S at a grid
	alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
	"""
	th = theta(u,phi1,phi2,z, r,alpha,a,rho0)
	om = omega(u,phi1,phi2,z, r,alpha,a,rho0)
	return np.array([om + th, om - th])

@njit
def theta(u, phi1, phi2, z, r, alpha, a, rho0):
	"""
	the value of theta at a gridpoint
	alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
	"""
	Phi, Pi = u
	rho_ = rho(u, rho0)
	p = P(rho_)
	v = V(u,p)

	Tf_tt, Tf_tr, Tf_rr = calcT_f(u,v,p,a,alpha)
	Tb_tt, Tb_tr, Tb_rr = calcT_b(r,phi1,phi2,z,a,alpha)

	Ttot_tt = Tf_tt + Tb_tt
	Ttot_tr = Tf_tr + Tb_tr
	Ttot_rr = Tf_rr + Tb_rr

	first_scale = 4 * np.pi * G * r * a * alpha
	first_brackets = 2 * (alpha**2/a**2) * Ttot_tr * Tf_tr + Ttot_rr * Tf_tt + Ttot_tt * Tf_rr
	second_scale = (alpha/a) * (a**2 - 1)/(2*r)
	second_brackets = 1/2 * ((Pi - Phi) * v - (Pi+Phi)) + p

	return first_scale * first_brackets + second_scale * second_brackets

@njit
def omega(u, phi1, phi2, z, r, alpha, a, rho0):

	rho_ = rho(u, rho0)
	p = P(rho_)
	v = V(u,p)

	Tf_tt, Tf_tr, Tf_rr = calcT_f(u,v,p,a,alpha)
	Tb_tt, Tb_tr, Tb_rr = calcT_b(r,phi1,phi2,z,a,alpha)

	Ttot_tt = Tf_tt + Tb_tt
	Ttot_tr = Tf_tr + Tb_tr
	Ttot_rr = Tf_rr + Tb_rr

	scale = -4*np.pi*G*r*alpha**2

	return scale * (Tf_tr * (Ttot_rr - Ttot_tt) - Ttot_tr * (Tf_rr - Tf_tt))



# ========== ENERGY MOMENTUM TENSOR COMPONENTS

@njit
def calcT_f(u,v,p,a,alpha):
	"""
	u: conservative variables at given point
	rho0: previous value of rho at current point
	"""
	Pi, Phi = u

	T_tt = -(1/2) * (Pi+Phi)
	T_tr = (a/(2*alpha)) * (Pi-Phi)
	T_rr = (1/2) * (Pi-Phi)*v + p

	return np.array([T_tt,T_tr,T_rr])

@njit
def calcT_b(r,phi1,phi2,z,a,alpha):

	lphi1, X1, Y1 = phi1
	lphi2, X2, Y2 = phi2

	U = calc_scalar_potential(lphi1,lphi2)

	T_tt = -(1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - U - z**2/(2*r**4)
	T_rr = (1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - U - z**2/(2*r**4)
	T_tr = -(2/(alpha*a)) * (Y1*X1 + Y2*X2)

	return np.array([T_tt,T_rr,T_tr])

@njit
def calc_scalar_potential(lphi1,lphi2):
	return mu**2 * (lphi1**2 + lphi2**2) + Lambda * (lphi1**2 + lphi2**2)

@njit
def calcTtot_rr(r,cons,prim,sc1,sc2,z,a):

	Pi, Phi = cons
	p, rho, v = prim

	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	U = calc_scalar_potential(lphi1,lphi2)

	Tf_rr = (1/2) * (Pi-Phi)*v + p
	Tb_rr = (1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - U - z**2/(2*r**4)

	return Tf_rr + Tb_rr

@njit
def calcTtot_tt(r,cons,sc1,sc2,z,a):
	Pi, Phi = cons

	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	U = calc_scalar_potential(lphi1,lphi2)

	Tf_tt = -(1/2) * (Pi+Phi)
	Tb_tt = -(1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - U - z**2/(2*r**4)

	return Tf_tt + Tb_tt

@njit
def calcTtot_tr(cons, sc1, sc2, a, alpha):
	
	Pi, Phi = cons

	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	Tf_tr = (a/(2*alpha)) * (Pi-Phi)
	Tb_tr = -(2/(alpha*a)) * (Y1*X1 + Y2*X2)
	
	return Tf_tr + Tb_tr


# ========== GRAVITY FUNCTIONS


# initial a
@njit
# def fa0(i, r,a,Pi,Phi):
def fa0(r, a, cons, sc1, sc2, z):

	if r == 0:
		return 0
	else:
		Ttot_tt = calcTtot_tt(r, cons, sc1, sc2, z, a)
		return -4 * np.pi * r * a**3 * Ttot_tt - a * (a**2 - 1) / (2 * r)

# time evolution of a
@njit
def fa(i,alpha,a,cons,sc1,sc2):
	r = R(i)
	Ttot_tr = calcTtot_tr(cons, sc1, sc2, a, alpha)
	return -4 * np.pi * r * alpha * a**2 * Ttot_tr 

# alpha (eq. 46)
@njit
def falpha(alpha_p1, r, cons, prim, sc1, sc2, z, a):
	# brackets = 1/2 * (Pi - Phi) * v + P
	brackets = calcTtot_rr(r,cons,prim,sc1,sc2,z,a)
	braces = 4 * np.pi * r * a**2 * brackets + (a**2 - 1)/(2*r)
	exponential = np.exp(-dr*braces)
	return alpha_p1 * exponential

# ========== SCALAR FUNCTIONS

@njit
def dphi1_dt(sc1, sc2, a, alpha, Omega):
	"""
	dphi_dt at a cell boundary; make sure that a is an averaged value (p1h)
	"""

	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return (alpha/a)*Y1 - g*(alpha/a)*Omega * lphi2

@njit
def dphi2_dt(sc1, sc2, a, alpha, Omega):
	"""
	dphi_dt at a cell boundary; make sure that a is an averaged value (p1h)
	"""

	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return (alpha/a)*Y2 + g*(alpha/a)*Omega * lphi1

@njit
def dX1_dt(i, sc1_p3h, sc1_m1h, sc2_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

	r_p3h = R(i+3/2)
	r_p1h = R(i+1/2)
	r_m1h = R(i-1/2)

	Y_p3h, Y_m1h = sc1_p3h[Y_i], sc1_m1h[Y_i]

	lphi2, X2, Y2 = sc2_p1h
	
	def fact(a, alpha, Y): return (alpha/a)*Y

	first_term = 1/(r_p3h - r_m1h) * (fact(a_p3h, alpha_p3h, Y_p3h) - fact(a_m1h, alpha_m1h, Y_m1h))
	last_terms = -g*((alpha_p1h/a_p1h)*Omega_p1h)*X2 + g*A_r_p1h*((alpha_p1h/a_p1h)*Y2) + g*((alpha_p1h * a_p1h)/r_p1h**2)*z_p1h*lphi2
	
	return first_term + last_terms

@njit
def dX2_dt(i, sc2_p3h, sc2_m1h, sc1_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

	r_p3h = R(i+3/2)
	r_p1h = R(i+1/2)
	r_m1h = R(i-1/2)

	Y_p3h, Y_m1h = sc2_p3h[Y_i], sc2_m1h[Y_i]

	lphi1, X1, Y1 = sc1_p1h
	
	def fact(a, alpha, Y): return (alpha/a)*Y

	first_term = 1/(r_p3h - r_m1h) * (fact(a_p3h, alpha_p3h, Y_p3h) - fact(a_m1h, alpha_m1h, Y_m1h))
	last_terms = +g*((alpha_p1h/a_p1h)*Omega_p1h)*X1 - g*A_r_p1h*((alpha_p1h/a_p1h)*Y1) - g*((alpha_p1h * a_p1h)/r_p1h**2)*z_p1h*lphi1
	
	return first_term + last_terms

# TODO: create dY1_dt, dY2_dt; be attentive to signs

@njit
def dY1_dt(i, sc1_p1h, sc1_p3h, sc1_m1h, sc2_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

	r_p1h = R(i+1/2)
	r_p3h = R(i+3/2)
	r_m1h = R(i-1/2)

	X_p3h, X_m1h = sc1_p3h[X_i], sc1_m1h[X_i]
	lphi1, X1, Y1 = sc1_p1h
	lphi2, X2, Y2 = sc2_p1h

	def fact(r, a, alpha, X): return (r**2 * alpha/a)*X

	first_term = 	(3/(r_p3h**3 - r_m1h**3)) * (fact(r_p3h, a_p3h, alpha_p3h, X_p3h) - fact(r_m1h, a_m1h, alpha_m1h, X_m1h))
	second_term = 	g*(Y2*(alpha_p1h/a_p1h)*Omega_p1h - (alpha_p1h/a_p1h)*X2*A_r_p1h)
	third_term = 	alpha_p1h * a_p1h * (mu**2 + 2*Lambda*(lphi1**2 + lphi2**2)) * lphi1

	return first_term - second_term - third_term

@njit
def dY2_dt(i, sc2_p1h, sc2_p3h, sc2_m1h, sc1_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

	r_p1h = R(i+1/2)
	r_p3h = R(i+3/2)
	r_m1h = R(i-1/2)

	X_p3h, X_m1h = sc2_p3h[X_i], sc2_m1h[X_i]
	lphi2, X2, Y2 = sc2_p1h
	lphi1, X1, Y1 = sc1_p1h

	def fact(r, a, alpha, X): return (r**2 * alpha/a)*X

	first_term = 	(3/(r_p3h**3 - r_m1h**3)) * (fact(r_p3h, a_p3h, alpha_p3h, X_p3h) - fact(r_m1h, a_m1h, alpha_m1h, X_m1h))
	second_term = 	g*(Y1*(alpha_p1h/a_p1h)*Omega_p1h - (alpha_p1h/a_p1h)*X1*A_r_p1h)
	third_term = 	alpha_p1h * a_p1h * (mu**2 + 2*Lambda*(lphi1**2 + lphi2**2)) * lphi2

	return first_term + second_term - third_term

@njit
def outer_dphi1_dt(r, sc1, sc2, A_r):
	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return -(lphi1/r) - X1 + g*A_r*lphi2

@njit
def outer_dphi2_dt(r, sc1, sc2, A_r):
	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return -(lphi2/r) - X2 - g*A_r*lphi1

@njit
def outer_X1(r, sc1, sc2, Omega, A_r):
	"""
	all values are expected at a cellboundary, p1h
	"""
	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return -(lphi1/r) - Y1 + g*lphi2*(Omega + A_r)

@njit
def outer_X2(r, sc1, sc2, Omega, A_r):
	"""
	all values are expected at a cellboundary, p1h
	"""
	lphi1, X1, Y1 = sc1
	lphi2, X2, Y2 = sc2

	return -(lphi2/r) - Y2 - g*lphi1*(Omega + A_r)

@njit
def outer_dY_dt(r,sc,sc_m1,sc_m2):
	Y = sc[Y_i]
	Y_m1 = sc_m1[Y_i]
	Y_m2 = sc_m2[Y_i]

	dY_dr = (Y_m2 - 4 * Y_m1 + 3 * Y)/(2 * dr)

	return -(Y/r) - dY_dr

# ========== DARK PHOTON EQUATIONS

@njit
def dOmega_dt(i, a_p3h, a_m1h, alpha_p3h, alpha_m1h, A_r_p3h, A_r_m1h):

	r_p1h = R(i+1/2)
	r_p3h = R(i+3/2)
	r_m1h = R(i-1/2)

	def fact(r, a, alpha, A_r): return (r**2 * alpha/a)*A_r

	return (3/(r_p3h**3 - r_m1h**3)) * (fact(r_p3h, a_p3h, alpha_p3h, A_r_p3h) - fact(r_m1h, a_m1h, alpha_m1h,A_r_m1h))

@njit
def dA_r_dt(i, z_p1h, Omega_p3h, Omega_m1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

	r_p1h = R(i+1/2)

	def fact(alpha, a, Omega): return (alpha/a)*Omega

	first_term = ((alpha_p1h * a_p1h)/(r_p1h**2)) * z_p1h
	dA_t_dt = (1/dr)*(1/2)*(fact(alpha_p3h, a_p3h, Omega_p3h) - fact(alpha_m1h, a_m1h, Omega_m1h))
	
	return first_term + dA_t_dt

@njit
def dz_dt(i, alpha_p1h, a_p1h, sc1_p1h, sc2_p1h):

	r_p1h = R(i+1/2)

	lphi1, X1, Y1 = sc1_p1h
	lphi2, X2, Y2 = sc2_p1h

	Jr = -(g/(a_p1h**2)) * (lphi1 * X2 - lphi2*X1)

	return -alpha_p1h*a_p1h*(r_p1h**2)*Jr

@njit
def outer_dA_r_dt(A_r_p1h, A_r_m1h, A_r_m3h):
	return -(1/2) * (3*A_r_p1h - 4*A_r_m1h + 1*A_r_m3h)/dr

@njit
def outer_dOmega_dt(Omega_p1h, Omega_m1h, Omega_m3h):
	return -(1/2) * (3*Omega_p1h - 4*Omega_m1h + 1*Omega_m3h)/dr


@njit
def outer_dz_dt(i, sc1_p1h, sc2_p1h):
	r_p1h = R(i)

	lphi1, X1, Y1 = sc1_p1h
	lphi2, X2, Y2 = sc2_p1h

	return r_p1h**2 * g*(lphi1*X2 - lphi2*X1)
