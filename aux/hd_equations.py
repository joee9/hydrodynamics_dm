# Joe Nyhan, 30 June 2021
# The equations used for simulation.

# from logging import debug
from numba.core.decorators import jit
from hd_params import *
from aux.hd_eos import *
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
def du_dt(i, rho0, u, phi1, phi2, a, a_p1h, a_m1h, alpha, alpha_p1h, alpha_m1h, F1_p1h, F1_m1h, F2_p1h, F2_m1h):
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

    s = S(r, u, phi1, phi2, a, alpha, rho0)

    return -firstFrac * firstBracket - 1/dr * secondBracket + s

@njit
def S(r, u, phi1, phi2, a, alpha, rho0):
    """
    the value of S at a grid
    alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
    """
    th = theta(r, u, phi1, phi2, a, alpha, rho0)
    om = omega(r, u, phi1, phi2, a, alpha, rho0)
    return np.array([om+th, om-th])

@njit
def theta(r, u, phi1, phi2, a, alpha, rho0):
    """
    the value of theta at a gridpoint
    alpha: make sure to pass an averaged value, for surrounding boundaries (i.e. (i+1) + i/2 for indices) 
    """
    Pi, Phi = u
    p, rho, v = calcPrims(u, rho0)

    Tf_tt, Tf_tr, Tf_rr = calc_Tf(u, a, alpha, rho0)
    Tb_tt, Tb_tr, Tb_rr = calc_Tb(phi1, phi2, a, alpha)

    Ttot_tt = Tf_tt + Tb_tt
    Ttot_tr = Tf_tr + Tb_tr
    Ttot_rr = Tf_rr + Tb_rr

    first_scale = 4 * np.pi * G * r * a * alpha
    first_brackets = 2 * (alpha**2/a**2) * Ttot_tr * Tf_tr + Ttot_rr * Tf_tt + Ttot_tt * Tf_rr
    second_scale = (alpha/a) * (a**2 - 1)/(2*r)
    second_brackets = 1/2 * ((Pi - Phi) * v - (Pi+Phi)) + p

    return first_scale * first_brackets + second_scale * second_brackets

@njit
def omega(r, u, phi1, phi2, a, alpha, rho0):

    Tf_tt, Tf_tr, Tf_rr = calc_Tf(u, a, alpha, rho0)
    Tb_tt, Tb_tr, Tb_rr = calc_Tb(phi1, phi2, a, alpha)

    Ttot_tt = Tf_tt + Tb_tt
    Ttot_tr = Tf_tr + Tb_tr
    Ttot_rr = Tf_rr + Tb_rr

    scale = -4*np.pi*G*r*alpha**2

    return scale * (Tf_tr * (Ttot_rr - Ttot_tt) - Ttot_tr * (Tf_rr - Tf_tt))

# ========== GRAVITY FUNCTIONS

# initial a
@njit
def fa0(r, u, phi1, phi2, a, alpha, rho0):

    if r == 0:
        return 0
    else:
        # Pi, Phi = u 
        # return 4 * np.pi * r * a**3 * (Pi + Phi) / 2 - a * (a**2 - 1) / (2 * r)
        Ttot_tt = calc_Ttot_tt(u, phi1, phi2, a, alpha, rho0)
        return -4 * np.pi * r * a**3 * Ttot_tt - a * (a**2 - 1) / (2 * r)

# time evolution of a
@njit
def fa(r, u, phi1, phi2, a, alpha, rho0):
# def fa(i,alpha,a,cons):
    Pi, Phi = u
    Ttot_tr = calc_Ttot_tr(u, phi1, phi2, a, alpha, rho0)
    return -4 * np.pi * r * alpha * a**2 * Ttot_tr

# alpha (eq. 46)
@njit
def falpha(alpha_p1, r, u, phi1, phi2, a, rho0):
    # brackets = 1/2 * (Pi - Phi) * v + P
    brackets = calc_Ttot_rr(u, phi1, phi2, a, alpha_p1, rho0) # alpha value not actually sed in this function, just for consistency
    braces = 4 * np.pi * r * a**2 * brackets + (a**2 - 1)/(2*r)
    exponential = np.exp(-dr*braces)
    return alpha_p1 * exponential

# ========== SCALAR FIELD

@njit
def calc_scalar_potential(sc1, sc2):
    varphi1, X1, Y1 = sc1
    varphi2, X2, Y2 = sc2

    return mu**2 * (varphi1**2 + varphi2**2) + Lambda * (varphi1**2 + varphi2**2)**2

# fermion sector
@njit
def calc_Tf_tt(u, a, alpha, rho0):
    Pi, Phi = u
    return -(1/2)*(Pi + Phi)

@njit
def calc_Tf_tr(u, a, alpha, rho0):
    Pi, Phi = u
    return (a/(2*alpha)) * (Pi - Phi)

@njit
def calc_Tf_rr(u, a, alpha, rho0):
    Pi, Phi = u
    p, rho, v = calcPrims(u, rho0)
    return (1/2)*(Pi - Phi)*v + p

# boson sector
@njit
def calc_Tb_tt(sc1, sc2, a, alpha):
    varphi1, X1, Y1 = sc1
    varphi2, X2, Y2 = sc2

    return -(1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - calc_scalar_potential(sc1,sc2)

@njit
def calc_Tb_rr(sc1, sc2, a, alpha):
    varphi1, X1, Y1 = sc1
    varphi2, X2, Y2 = sc2

    return +(1/a**2) * (X1**2 + X2**2 + Y1**2 + Y2**2) - calc_scalar_potential(sc1,sc2)

@njit
def calc_Tb_tr(sc1, sc2, a, alpha):
    varphi1, X1, Y1 = sc1
    varphi2, X2, Y2 = sc2

    return -(2/(a*alpha)) * (X1*Y1 + X2*Y2)

# totals
@njit
def calc_Ttot_tt(u, sc1, sc2, a, alpha, rho0):
    Tf_tt = calc_Tf_tt(u, a, alpha, rho0)
    Tb_tt = calc_Tb_tt(sc1, sc2, a, alpha)
    return Tf_tt + Tb_tt

@njit
def calc_Ttot_tr(u, sc1, sc2, a, alpha, rho0):
    Tf_tr = calc_Tf_tr(u, a, alpha, rho0)
    Tb_tr = calc_Tb_tr(sc1, sc2, a, alpha)
    return Tf_tr + Tb_tr

@njit
def calc_Ttot_rr(u, sc1, sc2, a, alpha, rho0):
    Tf_rr = calc_Tf_rr(u, a, alpha, rho0)
    Tb_rr = calc_Tb_rr(sc1, sc2, a, alpha)
    return Tf_rr + Tb_rr

# all values
@njit
def calc_Tb(sc1, sc2, a, alpha):
    Tb_tt = calc_Tb_tt(sc1,sc2,a,alpha)
    Tb_tr = calc_Tb_tr(sc1,sc2,a,alpha)
    Tb_rr = calc_Tb_rr(sc1,sc2,a,alpha)
    return np.array([Tb_tt, Tb_tr, Tb_rr])

@njit
def calc_Tf(u, a, alpha, rho0):
    Tf_tt = calc_Tf_tt(u,a,alpha,rho0)
    Tf_tr = calc_Tf_tr(u,a,alpha,rho0)
    Tf_rr = calc_Tf_rr(u,a,alpha,rho0)
    return np.array([Tf_tt, Tf_tr, Tf_rr])

# ========== SCALAR FUNCTIONS

@njit
def dphi1_dt(sc1, sc2, a, alpha):
    """
    dphi_dt at a cell boundary; make sure that a is an averaged value (p1h)
    """

    lphi1, X1, Y1 = sc1
    lphi2, X2, Y2 = sc2

    return (alpha/a)*Y1

@njit
def dphi2_dt(sc1, sc2, a, alpha):
    """
    dphi_dt at a cell boundary; make sure that a is an averaged value (p1h)
    """

    lphi1, X1, Y1 = sc1
    lphi2, X2, Y2 = sc2

    return (alpha/a)*Y2

@njit
def dX1_dt(i, sc1_p3h, sc1_m1h, sc2_p1h, a_p3h, a_m1h, alpha_p3h, alpha_m1h):

    r_p3h = R(i+3/2)
    r_p1h = R(i+1/2)
    r_m1h = R(i-1/2)

    Y_p3h, Y_m1h = sc1_p3h[Y_i], sc1_m1h[Y_i]

    lphi2, X2, Y2 = sc2_p1h
    
    def fact(a, alpha, Y): return (alpha/a)*Y

    first_term = 1/(r_p3h - r_m1h) * (fact(a_p3h, alpha_p3h, Y_p3h) - fact(a_m1h, alpha_m1h, Y_m1h))
    # last_terms = -g*((alpha_p1h/a_p1h)*Omega_p1h)*X2 + g*A_r_p1h*((alpha_p1h/a_p1h)*Y2) + g*((alpha_p1h * a_p1h)/r_p1h**2)*z_p1h*lphi2
    
    return first_term

@njit
def dX2_dt(i, sc2_p3h, sc2_m1h, sc1_p1h, a_p3h, a_m1h, alpha_p3h, alpha_m1h):

    r_p3h = R(i+3/2)
    r_p1h = R(i+1/2)
    r_m1h = R(i-1/2)

    Y_p3h, Y_m1h = sc2_p3h[Y_i], sc2_m1h[Y_i]

    lphi1, X1, Y1 = sc1_p1h
    
    def fact(a, alpha, Y): return (alpha/a)*Y

    first_term = 1/(r_p3h - r_m1h) * (fact(a_p3h, alpha_p3h, Y_p3h) - fact(a_m1h, alpha_m1h, Y_m1h))
    # last_terms = +g*((alpha_p1h/a_p1h)*Omega_p1h)*X1 - g*A_r_p1h*((alpha_p1h/a_p1h)*Y1) - g*((alpha_p1h * a_p1h)/r_p1h**2)*z_p1h*lphi1
    
    return first_term

@njit
def dY1_dt(i, sc1_p1h, sc1_p3h, sc1_m1h, sc2_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

    r_p1h = R(i+1/2)
    r_p3h = R(i+3/2)
    r_m1h = R(i-1/2)

    X_p3h, X_m1h = sc1_p3h[X_i], sc1_m1h[X_i]
    lphi1, X1, Y1 = sc1_p1h
    lphi2, X2, Y2 = sc2_p1h

    def fact(r, a, alpha, X): return (r**2 * alpha/a)*X

    first_term = (3/(r_p3h**3 - r_m1h**3)) * (fact(r_p3h, a_p3h, alpha_p3h, X_p3h) - fact(r_m1h, a_m1h, alpha_m1h, X_m1h))
    # second_term =     g*(Y2*(alpha_p1h/a_p1h)*Omega_p1h - (alpha_p1h/a_p1h)*X2*A_r_p1h)
    second_term = alpha_p1h * a_p1h * (mu**2 + 2*Lambda*(lphi1**2 + lphi2**2)) * lphi1

    return first_term - second_term

@njit
def dY2_dt(i, sc2_p1h, sc2_p3h, sc2_m1h, sc1_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h):

    r_p1h = R(i+1/2)
    r_p3h = R(i+3/2)
    r_m1h = R(i-1/2)

    X_p3h, X_m1h = sc2_p3h[X_i], sc2_m1h[X_i]
    lphi2, X2, Y2 = sc2_p1h
    lphi1, X1, Y1 = sc1_p1h

    def fact(r, a, alpha, X): return (r**2 * alpha/a)*X

    first_term = (3/(r_p3h**3 - r_m1h**3)) * (fact(r_p3h, a_p3h, alpha_p3h, X_p3h) - fact(r_m1h, a_m1h, alpha_m1h, X_m1h))
    # second_term =     g*(Y1*(alpha_p1h/a_p1h)*Omega_p1h - (alpha_p1h/a_p1h)*X1*A_r_p1h)
    second_term = alpha_p1h * a_p1h * (mu**2 + 2*Lambda*(lphi1**2 + lphi2**2)) * lphi2

    return first_term - second_term

@njit
def outer_dphi_dt(r, sc):
    lphi, X, Y = sc

    return -(lphi/r) - X

@njit
def outer_X(r, sc):
    """
    all values are expected at a cellboundary, p1h
    """
    lphi, X, Y = sc

    return -(lphi/r) - Y

@njit
def outer_dY_dt(r,sc,sc_m1,sc_m2):
    Y = sc[Y_i]
    Y_m1 = sc_m1[Y_i]
    Y_m2 = sc_m2[Y_i]

    dY_dr = (Y_m2 - 4 * Y_m1 + 3 * Y)/(2 * dr)

    return -(Y/r) - dY_dr
