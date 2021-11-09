# Joe Nyhan, 30 June 2021# Joe Nyhan, 30 June 2021
# Equation of state for hydrodynamical simulation of neutron star.

from hd_params import *

# ========== NEWTON RHAPSON

# necessary functions for root finding
@njit
def f_rho(u, rho):
    Pi, Phi = u
    p = P(rho)
    return (Pi + Phi - 2 * rho) * (Pi+ Phi + 2 * p) - (Pi - Phi)**2

@njit
def df_drho(u, rho):
    Pi, Phi = u
    p = P(rho)
    return -2*(Pi + Phi + 2*p) + (Pi + Phi - 2*rho) * (2 * dP_drho(rho,p))


# root finding algorithm

@njit
def rootFinder(u, rho0):

    def tol(rho_new, rho_old):
        return np.abs(rho_new - rho_old)/(rho_new + rho_old)

    rho_old = rho0
    rho_new = 0

    for i in range(NR_MAX_ITERATIONS):
        rho_new = rho_old - f_rho(u, rho_old) / df_drho(u, rho_old)
        if rho_new < 0:
            rho_new = rho_old / 2
            # break
        elif tol(rho_new, rho_old) < NR_TOL:
            break
        else:
            rho_old = rho_new
    
    return rho_new


# ========== EOS implementations

if not eos_UR:
    # numerically solve for rho
    @njit
    def rho(u,rho0):
        return rootFinder(u,rho0)


if eos_UR:

    @njit
    def P(rho):
        return (Gamma - 1) * rho

    @njit
    def rho(u,rho0):
        Pi, Phi = u
        first_term = -(Pi+Phi)* (2-Gamma)/(4*(Gamma-1))
        second_term = 1/(Gamma-1)
        under_sqrt = (Phi+Phi)**2*((2-Gamma)/4)**2 + (Gamma-1)*Pi*Phi
        return first_term + second_term*np.sqrt(under_sqrt)


if eos_polytrope:

    @njit
    def P(rho):
        return K * rho**Gamma

    @njit
    def dP_drho(rho,p):
        return K * Gamma * rho**(Gamma-1)


if eos_SLy:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.005925
        a4  = 0.16326
        a5  = 6.48
        a6  = 11.4971
        a7  = 19.105
        a8  = 0.8938
        a9  = 6.54
        a10 = 11.4950
        a11 = -22.775
        a12 = 1.5707
        a13 = 4.3
        a14 = 14.08
        a15 = 27.80
        a16 = -1.653
        a17 = 1.50
        a18 = 14.67

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 + (a7+a8*xi)*f0_9_10 + (a11+a12*xi)*f0_13_14 + (a15+a16*xi)*f0_17_18

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.005925
        a4  = 0.16326
        a5  = 6.48
        a6  = 11.4971
        a7  = 19.105
        a8  = 0.8938
        a9  = 6.54
        a10 = 11.4950
        a11 = -22.775
        a12 = 1.5707
        a13 = 4.3
        a14 = 14.08
        a15 = 27.80
        a16 = -1.653
        a17 = 1.50
        a18 = 14.67

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6));
        d_f0_9_10  = + f0_9_10**2  * a9  * np.exp(a9 *(a10-xi));
        d_f0_13_14 = + f0_13_14**2 * a13 * np.exp(a13*(a14-xi));
        d_f0_17_18 = + f0_17_18**2 * a17 * np.exp(a17*(a18-xi));

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_10   + (a7+a8*xi)*d_f0_9_10 \
        + a12*f0_13_14 + (a11+a12*xi)*d_f0_13_14 \
        + a16*f0_17_18 + (a15+a16*xi)*d_f0_17_18

        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho

if eos_FPS:

    @njit
    def f0(x):
        return 1/(1+np.exp(x))

    @njit
    def P(rho):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.006004
        a4  = 0.16345
        a5  = 6.50
        a6  = 11.8440
        a7  = 17.24
        a8  = 1.065
        a9  = 6.54
        a10 = 11.8421
        a11 = -22.003
        a12 = 1.5552
        a13 = 9.3
        a14 = 14.19
        a15 = 27.73
        a16 = -1.508
        a17 = 1.79
        a18 = 15.13

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))

        zeta = ( (a1 + a2*xi + a3*xi**3) / (1.0+a4*xi) )*f0_5_6 + (a7+a8*xi)*f0_9_10 + (a11+a12*xi)*f0_13_14 + (a15+a16*xi)*f0_17_18

        # p = (4.7953e-39) * 10^\zeta
        # log10(4.7953e-39) = -38.319184217634302        
        sum = zeta - 38.319184217634302

        return 10**(sum)

    @njit
    def dP_drho(rho,p):

        a1  = 6.22
        a2  = 6.121
        a3  = 0.006004
        a4  = 0.16345
        a5  = 6.50
        a6  = 11.8440
        a7  = 17.24
        a8  = 1.065
        a9  = 6.54
        a10 = 11.8421
        a11 = -22.003
        a12 = 1.5552
        a13 = 9.3
        a14 = 14.19
        a15 = 27.73
        a16 = -1.508
        a17 = 1.79
        a18 = 15.13

        # rho = (4.30955e-18) * 10^\xi
        # log10(4.30955e-18) = - 17.365568076178008 
        xi = np.log10(rho) + 17.365568076178008

        f0_5_6   = f0(a5 *(xi-a6))
        f0_9_10  = f0(a9 *(a10-xi))
        f0_13_14 = f0(a13*(a14-xi))
        f0_17_18 = f0(a17*(a18-xi))
        
        d_f0_5_6   = - f0_5_6**2   * a5  * np.exp(a5 *(xi-a6));
        d_f0_9_10  = + f0_9_10**2  * a9  * np.exp(a9 *(a10-xi));
        d_f0_13_14 = + f0_13_14**2 * a13 * np.exp(a13*(a14-xi));
        d_f0_17_18 = + f0_17_18**2 * a17 * np.exp(a17*(a18-xi));

        dzeta_dxi = (((a2 + 3*a3*xi**2)/(1+a4*xi)) - a4*(a1 + a2*xi + a3*xi**3) / ((1+a4*xi)*(1+a4*xi)) )*f0_5_6  \
        + ((a1 + a2*xi + a3*xi**3) / (1+a4*xi))*d_f0_5_6 \
        + a8*f0_9_10   + (a7+a8*xi)*d_f0_9_10 \
        + a12*f0_13_14 + (a11+a12*xi)*d_f0_13_14 \
        + a16*f0_17_18 + (a15+a16*xi)*d_f0_17_18

        dP_dzeta = p

        dxi_drho = 1/rho

        return dP_dzeta * dzeta_dxi * dxi_drho