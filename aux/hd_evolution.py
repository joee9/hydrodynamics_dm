# Joe Nyhan, 08 July 2021
# Evolution equations for dark matter scalar field.

from hd_params import *
from aux.hd_equations import *
from aux.hd_riemann import *
from aux.hd_ops import *

# calculate k values in evolution
@njit
# def calculate_rk3_kval(h, ku, ka, kphi1, kphi2, kz, kOmega, kA_r, alpha, a, u, F1, F2, phi1, phi2, z, Omega, A_r, rhos):
def calculate_rk3_kval(h, ku, ka, alpha, a, u, F1, F2, rhos):

	for i in range(NUM_VPOINTS, NUM_SPOINTS-2):
		# print(f"{i=}\n")

		alpha_m1h = alpha[i-1]
		alpha_p1h = alpha[i]
		# alpha_p3h = alpha[i+1]
		alpha_ = (alpha_p1h + alpha_m1h)/2

		a_ = a[i]
		a_p1h = (a[i] + a[i+1])/2
		a_m1h = (a[i] + a[i-1])/2
		# a_p3h = (a[i+1] + a[i+2])/2

		# phi1_ = (phi1[i,:] + phi1[i-1,:])/2
		# phi2_ = (phi2[i,:] + phi2[i-1,:])/2
		
		# phi1_p1h = phi1[i,:]
		# phi1_p3h = phi1[i+1,:]
		# phi1_m1h = phi1[i-1,:]
		
		# phi2_p1h = phi2[i,:]
		# phi2_p3h = phi2[i+1,:]
		# phi2_m1h = phi2[i-1,:]

		# z_m1h = z[i-1]
		# z_p1h = z[i]
		# z_ = (z_p1h + z_m1h)/2

		# Omega_p1h = Omega[i]
		# Omega_p3h = Omega[i+1]
		# Omega_m1h = Omega[i-1]
		# # Omega_ = (Omega_p1h + Omega_m1h)/2

		# A_r_p1h = A_r[i]
		# A_r_p3h = A_r[i+1]
		# A_r_m1h = A_r[i-1]
		# # A_r_ = (A_r_p1h + A_r_m1h)/2


		# if i == NUM_VPOINTS:
		# 	kphi1[i,phi_i]  = h * dphi1_dt (phi1[i,:], phi2[i,:], a_p1h, alpha_p1h, Omega_p1h)
		# 	kphi1[i,X_i]	= h * dX1_dt   (i, phi1_p3h, phi1_m1h, phi2_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
		# 	kphi1[i,Y_i]	= h * dY1_dt   (i, phi1_p1h, phi1_p3h, phi1_m1h, phi2_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
			
		# 	kphi2[i,phi_i]  = h * dphi2_dt (phi1[i,:], phi2[i,:], a_p1h, alpha_p1h, Omega_p1h)
		# 	kphi2[i,X_i]	= h * dX2_dt   (i, phi2_p3h, phi2_m1h, phi1_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
		# 	kphi2[i,Y_i]	= h * dY2_dt   (i, phi2_p1h, phi2_p3h, phi2_m1h, phi1_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)

		# 	kz[i] 			= h * dz_dt	   (i, alpha_p1h, a_p1h, phi1_p1h, phi2_p1h)
		# 	kOmega[i]		= h * dOmega_dt(i, a_p3h, a_m1h, alpha_p3h, alpha_m1h, A_r_p3h, A_r_m1h)
		# 	kA_r[i]			= h * dA_r_dt  (i, z_p1h, Omega_p3h, Omega_m1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)

		if i>NUM_VPOINTS:
			ku[i,:] 		= h * du_dt(i, rhos[i], u[i,:], a_, a_p1h, a_m1h, \
				alpha_, alpha_p1h, alpha_m1h, F1[i,:], F1[i-1,:], F2[i,:], F2[i-1,:])
			ka[i] 			= h * fa(i, alpha_, a_, u[i,:])

			# kphi1[i,phi_i] = h * dphi1_dt(phi1[i,:], phi2[i,:], a_p1h, alpha_p1h, Omega_p1h)
			# kphi1[i,X_i]	= h * dX1_dt(i, phi1_p3h, phi1_m1h, phi2_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
			# kphi1[i,Y_i]	= h * dY1_dt(i, phi1_p1h, phi1_p3h, phi1_m1h, phi2_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
			
			# kphi2[i,phi_i] = h * dphi2_dt(phi1[i,:], phi2[i,:], a_p1h, alpha_p1h, Omega_p1h)
			# kphi2[i,X_i]	= h * dX2_dt(i, phi2_p3h, phi2_m1h, phi1_p1h, A_r_p1h, Omega_p1h, z_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)
			# kphi2[i,Y_i]	= h * dY2_dt(i, phi2_p1h, phi2_p3h, phi2_m1h, phi1_p1h, Omega_p1h, A_r_p1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)

			# kz[i] 			= h * dz_dt	   (i, alpha_p1h, a_p1h, phi1_p1h, phi2_p1h)
			# kOmega[i]		= h * dOmega_dt(i, a_p3h, a_m1h, alpha_p3h, alpha_m1h, A_r_p3h, A_r_m1h)
			# kA_r[i]			= h * dA_r_dt  (i, z_p1h, Omega_p3h, Omega_m1h, a_p1h, a_p3h, a_m1h, alpha_p1h, alpha_p3h, alpha_m1h)


@njit
# def evolution_charge(cons,prim,sc1,sc2,A_r,z,Omega,a,alpha,curr,next):
def evolution(cons,prim,a,alpha,curr,next):

	# Pi, Phi across space (i, func) at a given timestep
	u = cons[curr,:,:]

	# phi1 = sc1[curr,:,:]
	# phi2 = sc2[curr,:,:]

	# print(phi2[0,:])

	rhos = prim[curr,:,1] # used for Newton-Rhapson method as guesses

	# ========== RK3 ========== #
	h = dt

	# cell reconstruction for the grid
	# compute uL and uR for a given u; this is done for all space (shape: NUM_SPOINTS,2)
	uL = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
	uR = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
	cell_reconstruction(u, uL, uR)

	# print(uR[NUM_SPOINTS-10:,:])
	F1 = np.zeros((NUM_SPOINTS,2))
	F2 = np.zeros((NUM_SPOINTS,2))

	# calculate all fluxes
	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
		F1[i,:], F2[i,:] = findFluxes(i, uL[i,:], uR[i,:], rhos[i], rhos[i-1], rhos[i+1])

	
	# initialize all k values
	k1u = np.zeros((NUM_SPOINTS,2))
	k2u = np.zeros((NUM_SPOINTS,2))
	k3u = np.zeros((NUM_SPOINTS,2))

	# k1phi1 = np.zeros((NUM_SPOINTS,3))
	# k2phi1 = np.zeros((NUM_SPOINTS,3))
	# k3phi1 = np.zeros((NUM_SPOINTS,3))

	# k1phi2 = np.zeros((NUM_SPOINTS,3))
	# k2phi2 = np.zeros((NUM_SPOINTS,3))
	# k3phi2 = np.zeros((NUM_SPOINTS,3))

	k1a = np.zeros(NUM_SPOINTS)  
	k2a = np.zeros(NUM_SPOINTS)
	k3a = np.zeros(NUM_SPOINTS)

	# k1z = np.zeros(NUM_SPOINTS)  
	# k2z = np.zeros(NUM_SPOINTS)
	# k3z = np.zeros(NUM_SPOINTS)

	# k1Omega = np.zeros(NUM_SPOINTS)  
	# k2Omega = np.zeros(NUM_SPOINTS)
	# k3Omega = np.zeros(NUM_SPOINTS)

	# k1A_r = np.zeros(NUM_SPOINTS)  
	# k2A_r = np.zeros(NUM_SPOINTS)
	# k3A_r = np.zeros(NUM_SPOINTS)

	# calculate k1

	# calculate_rk3_kval(h, k1u, k1a, k1phi1, k1phi2, k1z, k1Omega, k1A_r, alpha[curr,:], a[curr,:], u, F1, F2, phi1, phi2, z[curr,:], Omega[curr,:], A_r[curr,:], rhos)
	calculate_rk3_kval(h, k1u, k1a, alpha[curr,:], a[curr,:], u, F1, F2, rhos)

	# intermediate calculations
	uk1 = u+k1u
	# floor the updated u values
	checkFloor(uk1[:,Pi_i])
	checkFloor(uk1[:,Phi_i])

	# inner boundary for updated u values
	Phi2 = uk1[2,Pi_i]
	Phi3 = uk1[3,Phi_i]
	r2 = R(2)
	r3 = R(3)
	uk1[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
	uk1[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
	initializeEvenVPs(uk1)

	# outer boundary for updated u values
	uk1[NUM_SPOINTS-1] = uk1[NUM_SPOINTS-3]
	uk1[NUM_SPOINTS-2] = uk1[NUM_SPOINTS-3]

	# calculate outer k1 for outer points
	i = NUM_SPOINTS-2
	r_p1h = R(i+1/2)

	k1a[i] 			= h * fa(i, (alpha[curr,i]+alpha[curr,i-1])/2, a[curr,i], u[i,:])

	# # k1phi1[i,phi_i] = h * outer_dphi1_dt	(r_p1h, phi1[i,:])
	# k1phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1[i,:], phi2[i,:], A_r[curr,i])
	# k1phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1[i,:], phi1[i-1,:], phi1[i-2,:])
	# k1phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1[i,:], phi2[i,:], A_r[curr,i])
	# k1phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2[i,:], phi2[i-1,:], phi2[i-2,:])

	# k1A_r[i]	  	= h * outer_dA_r_dt(A_r[curr,i], A_r[curr,i-1], A_r[curr,i-2])
	# k1z[i]			= h * outer_dz_dt(i, phi1[i,:], phi2[i-1,:])
	# k1Omega[i]		= h * outer_dOmega_dt(Omega[curr, i], Omega[curr,i-1], Omega[curr,i-2])

	i = NUM_SPOINTS-1
	r_p1h = R(i+1/2)

	k1a[i] 			= h * fa(i, 1/a[curr,i], a[curr,i], u[i,:])

	# k1phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1[i,:], phi2[i,:], A_r[curr,i])
	# k1phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1[i,:], phi1[i-1,:], phi1[i-2,:])
	# k1phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1[i,:], phi2[i,:], A_r[curr,i])
	# k1phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2[i,:], phi2[i-1,:], phi2[i-2,:])

	# k1A_r[i]	  	= h * outer_dA_r_dt(A_r[curr,i], A_r[curr,i-1], A_r[curr,i-2])
	# k1z[i]			= h * outer_dz_dt(i, phi1[i,:], phi2[i-1,:])
	# k1Omega[i]		= h * outer_dOmega_dt(Omega[curr, i], Omega[curr,i-1], Omega[curr,i-2])

	# construct shifted arrays
	ak1 = a[curr,:]+k1a
	# zk1 = z[curr,:]+k1z
	# Omegak1 = Omega[curr,:]+k1Omega
	# A_rk1 = A_r[curr,:]+k1A_r
	# phi1k1 = phi1+k1phi1
	# phi2k1 = phi2+k1phi2

	# calculate shifted X for phi1, phi2 arrays
	# i = NUM_SPOINTS-2
	# r_p1h = R(i+1/2)
	# # phi1k1_p1h = phi1k1[i,:]
	# # phi2k1_p1h = phi2k1[i,:]
	# # Omegak1_p1h = Omegak1[i,:]
	# # A_rk1_p1h = A_r
	# # phi1k1[i,X_i] = outer_X1(r_p1h, phi1k1[i,:])
	# phi1k1[i,X_i] = outer_X1(r_p1h, phi1k1[i,:], phi2k1[i,:], Omegak1[i], A_rk1[i])
	# phi2k1[i,X_i] = outer_X2(r_p1h, phi1k1[i,:], phi2k1[i,:], Omegak1[i], A_rk1[i])

	# i = NUM_SPOINTS-1
	# r_p1h = R(i+1/2)
	# phi1k1[i,X_i] = outer_X1(r_p1h, phi1k1[i,:], phi2k1[i,:], Omegak1[i], A_rk1[i])
	# phi2k1[i,X_i] = outer_X2(r_p1h, phi1k1[i,:], phi2k1[i,:], Omegak1[i], A_rk1[i])
		
	ak1[NUM_VPOINTS] = 1

	# initialize all virtual points
	initializeEvenVPs(ak1)
	# initializeEvenVPs(phi1k1[:,phi_i], "staggered")
	# initializeOddVPs(phi1k1[:,X_i])
	# initializeEvenVPs(phi1k1[:,Y_i], "staggered")
	
	# initializeEvenVPs(phi2k1[:,phi_i], "staggered")
	# initializeOddVPs(phi2k1[:,X_i])
	# initializeEvenVPs(phi2k1[:,Y_i], "staggered")

	# initializeEvenVPs(Omegak1, "staggered")
	# initializeOddVPs(zk1)
	# initializeOddVPs(A_rk1)
	# cell reconstruction
	uLk1 = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
	uRk1 = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
	cell_reconstruction(uk1,uLk1,uRk1)

	F1k1 = np.zeros((NUM_SPOINTS,2))
	F2k1 = np.zeros((NUM_SPOINTS,2))

	# calculate all fluxes
	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
		F1k1[i,:], F2k1[i,:] = findFluxes(i, uLk1[i,:], uRk1[i,:], rhos[i], rhos[i-1], rhos[i+1])

	# calculate k2
	# calculate_rk3_kval(h, k2u, k2a, k2phi1, k2phi2, k2z, k2Omega, k2A_r, alpha[curr,:], ak1, uk1, F1k1, F2k1, phi1k1, phi2k1, zk1, Omegak1, A_rk1, rhos)
	calculate_rk3_kval(h, k2u, k2a, alpha[curr,:], ak1, uk1, F1k1, F2k1, rhos)
	# calculate_rk3_kval(h, k1u, k1a, alpha[curr,:], a[curr,:], u, F1, F2, rhos)

	uk12 = u+(k1u + k2u)/4
	# floor the updated u values
	checkFloor(uk12[:,Pi_i])
	checkFloor(uk12[:,Phi_i])

	# inner boundary for updated u values
	Phi2 = uk12[2,Pi_i]
	Phi3 = uk12[3,Phi_i]
	r2 = R(2)
	r3 = R(3)
	uk12[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
	uk12[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
	initializeEvenVPs(uk12)

	# outer boundary for updated u values
	uk12[NUM_SPOINTS-1] = uk12[NUM_SPOINTS-3]
	uk12[NUM_SPOINTS-2] = uk12[NUM_SPOINTS-3]

	# k2 vals for outer points: a, scalar fields
	i = NUM_SPOINTS-2
	r_p1h = R(i+1/2)

	k2a[i] 			= h * fa(i, (alpha[curr,i]+alpha[curr,i-1])/2, ak1[i], uk1[i,:])

	# k2phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1k1[i,:], phi2k1[i,:], A_rk1[i])
	# k2phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1k1[i,:], phi1k1[i-1,:], phi1k1[i-2,:])
	# k2phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1k1[i,:], phi2k1[i,:], A_rk1[i])
	# k2phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2k1[i,:], phi2k1[i-1,:], phi2k1[i-2,:])

	# k2A_r[i]	  	= h * outer_dA_r_dt(A_rk1[i], A_rk1[i-1], A_rk1[i-2])
	# k2z[i]			= h * outer_dz_dt(i, phi1k1[i,:], phi2k1[i-1,:])
	# k2Omega[i]		= h * outer_dOmega_dt(Omegak1[i], Omegak1[i-1], Omegak1[i-2])
	
	i = NUM_SPOINTS-1
	r_p1h = R(i+1/2)

	k2a[i] 			= h * fa(i, 1/ak1[i], ak1[i], uk1[i,:])

	# k2phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1k1[i,:], phi2[i,:], A_rk1[i])
	# k2phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1k1[i,:], phi1k1[i-1,:], phi1k1[i-2,:])
	# k2phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1k1[i,:], phi2k1[i,:], A_rk1[i])
	# k2phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2k1[i,:], phi2k1[i-1,:], phi2k1[i-2,:])

	# k2A_r[i]	  	= h * outer_dA_r_dt(A_rk1[i], A_rk1[i-1], A_rk1[i-2])
	# k2z[i]			= h * outer_dz_dt(i, phi1k1[i,:], phi2k1[i-1,:])
	# k2Omega[i]		= h * outer_dOmega_dt(Omegak1[i], Omegak1[i-1], Omegak1[i-2])

	# compute shifted arrays
	ak12 = a[curr,:]+(k1a + k2a)/4
	# zk12 = z[curr,:]+(k1z + k2z)/4
	# Omegak12 = Omega[curr,:]+(k1Omega + k2Omega)/4
	# A_rk12 = A_r[curr,:]+(k1A_r + k2A_r)/4
	# phi1k12 = phi1+(k1phi1 + k2phi1)/4
	# phi2k12 = phi2+(k1phi2 + k2phi2)/4
		
	# calculate shifted X for phi1, phi2 arrays
	# i = NUM_SPOINTS-2
	# r_p1h = R(i+1/2)
	# phi1k12[i,X_i] = outer_X1(r_p1h, phi1k12[i,:], phi2k12[i,:], Omegak12[i], A_rk12[i])
	# phi2k12[i,X_i] = outer_X2(r_p1h, phi1k12[i,:], phi2k12[i,:], Omegak12[i], A_rk12[i])

	# i = NUM_SPOINTS-1
	# r_p1h = R(i+1/2)
	# phi1k12[i,X_i] = outer_X1(r_p1h, phi1k12[i,:], phi2k12[i,:], Omegak12[i], A_rk12[i])
	# phi2k12[i,X_i] = outer_X2(r_p1h, phi1k12[i,:], phi2k12[i,:], Omegak12[i], A_rk12[i])


	ak12[NUM_VPOINTS] = 1

	# initialize all virtual points
	initializeEvenVPs(ak12)
	# initializeEvenVPs(phi1k12[:,phi_i], "staggered")
	# initializeOddVPs(phi1k12[:,X_i])
	# initializeEvenVPs(phi1k12[:,Y_i], "staggered")

	# initializeEvenVPs(phi2k12[:,phi_i], "staggered")
	# initializeOddVPs(phi2k12[:,X_i])
	# initializeEvenVPs(phi2k12[:,Y_i], "staggered")

	# initializeEvenVPs(Omegak12, "staggered")
	# initializeOddVPs(zk12)
	# initializeOddVPs(A_rk12)

	uLk12 = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
	uRk12 = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
	cell_reconstruction(uk12,uLk12,uRk12)

	F1k12 = np.zeros((NUM_SPOINTS,2))
	F2k12 = np.zeros((NUM_SPOINTS,2))

	# calculate all fluxes
	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
		F1k12[i,:], F2k12[i,:] = findFluxes(i, uLk12[i,:], uRk12[i,:], rhos[i], rhos[i-1], rhos[i+1])
		

	# calculate_rk3_kval(h, k3u, k3a, k3phi1, k3phi2, k3z, k3Omega, k3A_r, alpha[curr,:], ak12, uk12, F1k12, F2k12, phi1k12, phi2k12, zk12, Omegak12, A_rk12, rhos)
	calculate_rk3_kval(h, k3u, k3a, alpha[curr,:], ak12, uk12, F1k12, F2k12, rhos)
	# calculate_rk3_kval(h, k2u, k2a, alpha[curr,:], ak1, uk1, F1k1, F2k1, rhos)


	i = NUM_SPOINTS-2
	r_p1h = R(i+1/2)
	k3a[i] 			= h * fa(i, (alpha[curr,i]+alpha[curr,i-1])/2, ak12[i], uk12[i,:])

	# k3phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1k12[i,:], phi2k12[i,:], A_rk12[i])
	# k3phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1k12[i,:], phi1k12[i-1,:], phi1k12[i-2,:])
	# k3phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1k12[i,:], phi2k12[i,:], A_rk12[i])
	# k3phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2k12[i,:], phi2k12[i-1,:], phi2k12[i-2,:])

	# k3A_r[i]	  	= h * outer_dA_r_dt(A_rk12[i], A_rk12[i-1], A_rk12[i-2])
	# k3z[i]			= h * outer_dz_dt(i, phi1k12[i,:], phi2k12[i-1,:])
	# k3Omega[i]		= h * outer_dOmega_dt(Omegak12[i], Omegak12[i-1], Omegak12[i-2])

	i = NUM_SPOINTS-1
	r_p1h = R(i+1/2)

	k3a[i] 			= h * fa(i, 1/ak12[i], ak12[i], uk12[i,:])

	# k3phi1[i,phi_i] = h * outer_dphi1_dt(r_p1h, phi1k12[i,:], phi2k12[i,:], A_rk12[i])
	# k3phi1[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi1k12[i,:], phi1k12[i-1,:], phi1k12[i-2,:])
	# k3phi2[i,phi_i] = h * outer_dphi2_dt(r_p1h, phi1k12[i,:], phi2k12[i,:], A_rk12[i])
	# k3phi2[i,Y_i] 	= h * outer_dY_dt(r_p1h, phi2k12[i,:], phi2k12[i-1,:], phi2k12[i-2,:])

	# k3A_r[i]	  	= h * outer_dA_r_dt(A_rk12[i], A_rk12[i-1], A_rk12[i-2])
	# k3z[i]			= h * outer_dz_dt(i, phi1k12[i,:], phi2k12[i-1,:])
	# k3Omega[i]		= h * outer_dOmega_dt(Omegak12[i], Omegak12[i-1], Omegak12[i-2])

	# calculate a values across the entire array
	for i in range(NUM_VPOINTS+1,NUM_SPOINTS):
		a[next,i] = a[curr,i] + 1/6 * (k1a[i] + k2a[i] + 4*k3a[i])
	
	a[next,NUM_VPOINTS] = 1
	initializeEvenVPs(a[next,:])

	# conservative functions
	for i in range(NUM_VPOINTS+1,NUM_SPOINTS-2):
		cons[next,i,:] = cons[curr,i,:] + 1/6 * (k1u[i,:] + k2u[i,:] + 4*k3u[i,:])
		
	# # scalar fields
	# for i in range(NUM_VPOINTS, NUM_SPOINTS-2):

	# 	sc1[next,i,:] = sc1[curr,i,:] + 1/6 * (k1phi1[i,:] + k2phi1[i,:] + 4*k3phi1[i,:])
	# 	sc2[next,i,:] = sc2[curr,i,:] + 1/6 * (k1phi2[i,:] + k2phi2[i,:] + 4*k3phi2[i,:])
	# 	Omega[next,i] = Omega[curr,i] + 1/6 * (k1Omega[i] + k2Omega[i] + 4*k3Omega[i])
	# 	z[next,i] = z[curr,i] + 1/6 * (k1z[i] + k2z[i] + 4*k3z[i])
	# 	A_r[next,i] = A_r[curr,i] + 1/6 * (k1A_r[i] + k2A_r[i] + 4*k3A_r[i])

	# for i in [NUM_SPOINTS-2, NUM_SPOINTS-1]:
	# 	# determine lphi, Y using predetermined k values 
	# 	Omega[next,i] = Omega[curr,i] + 1/6 * (k1Omega[i] + k2Omega[i] + 4*k3Omega[i])
	# 	z[next,i] = z[curr,i] + 1/6 * (k1z[i] + k2z[i] + 4*k3z[i])
	# 	A_r[next,i] = A_r[curr,i] + 1/6 * (k1A_r[i] + k2A_r[i] + 4*k3A_r[i])

	# 	sc1[next, i, phi_i] = sc1[curr, i, phi_i] + 1/6 * (k1phi1[i, phi_i] + k2phi1[i, phi_i] + 4 * k3phi1[i,phi_i])
	# 	sc1[next, i, Y_i] 	= sc1[curr, i, Y_i]	  + 1/6 * (k1phi1[i, Y_i]   + k2phi1[i, Y_i]   + 4 * k3phi1[i,Y_i])

	# 	sc2[next, i, phi_i] = sc2[curr, i, phi_i] + 1/6 * (k1phi2[i, phi_i] + k2phi2[i, phi_i] + 4 * k3phi2[i,phi_i])
	# 	sc2[next, i, Y_i] 	= sc2[curr, i, Y_i]   + 1/6 * (k1phi2[i, Y_i]   + k2phi2[i, Y_i]   + 4 * k3phi2[i,Y_i])

	# 	# compute X algebraically using those new values
	# 	sc1[next, i, X_i] 	= outer_X1(R(i+1/2), sc1[next,i,:], sc2[next,i,:], Omega[next,i], A_r[next,i])
	# 	sc2[next, i, X_i] 	= outer_X2(R(i+1/2), sc1[next,i,:], sc2[next,i,:], Omega[next,i], A_r[next,i])

	checkFloor(cons[next,:,Pi_i])
	checkFloor(cons[next,:,Phi_i])

	cons[next, NUM_SPOINTS-1,:] = cons[next, NUM_SPOINTS-3,:]
	cons[next, NUM_SPOINTS-2,:] = cons[next, NUM_SPOINTS-3,:]

	u = cons[next,:,:]

	Phi2 = cons[next,2,Phi_i]
	Phi3 = cons[next,3,Phi_i]
	r2 = R(2)
	r3 = R(3)
	cons[next,1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
	cons[next,1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)

	# virtual points
	initializeEvenVPs(cons[next,:,Pi_i])
	initializeEvenVPs(cons[next,:,Phi_i])
	
	# # inner boundary for scalar fields
	# initializeEvenVPs(sc1[next,:,phi_i], "staggered")
	# initializeOddVPs(sc1[next,:,X_i])
	# initializeEvenVPs(sc1[next,:,Y_i], "staggered")

	# initializeEvenVPs(sc2[next,:,phi_i], "staggered")
	# initializeOddVPs(sc2[next,:,X_i])
	# initializeEvenVPs(sc2[next,:,Y_i], "staggered")

	# initializeEvenVPs(Omega[next,:], "staggered")
	# initializeOddVPs(z[next,:])
	# initializeOddVPs(A_r[next,:])

	# ========== CALCULATE PRIMITIVES ========== #

	for i in range(NUM_VPOINTS, NUM_SPOINTS):
		u = cons[next,i,:]
		rho_ = rho(u, rhos[i])
		p = P(rho_)
		v = V(u,p)
		prim[next,i,P_i] = p
		prim[next,i,rho_i] = rho_
		prim[next,i,v_i] = v 

	initializeEvenVPs(prim[next,:,P_i])
	initializeEvenVPs(prim[next,:,rho_i])

	# ========== CALCULATE ALPHA ========== #

	# # calculate alpha at each cell boundary, storing it at the gridpoint to its left

	alpha[next,NUM_SPOINTS-1] = (1/2 * (a[next,NUM_SPOINTS-1] + a[next,NUM_SPOINTS-2]))**(-1) # average of last two a values
	for i in range(NUM_SPOINTS-1, NUM_VPOINTS, -1): # start at last boundary and move inwards
		# phi1_p1h = sc1[next,i,:]
		# phi1_m1h = sc1[next,i-1,:]
		# phi2_p1h = sc2[next,i,:]
		# phi2_m1h = sc2[next,i-1,:]

		# phi1 = (phi1_p1h + phi1_m1h)/2
		# phi2 = (phi2_p1h + phi2_m1h)/2

		# z_p1h = z[next,i]
		# z_m1h = z[next,i-1]
		# z_ = (z_p1h + z_m1h)/2

		alpha[next,i-1] = falpha(alpha[next,i], R(i), cons[next,i,:], prim[next,i,:], a[next,i])

	initializeEvenVPs(alpha[next,:], spacing="staggered")

# @njit
# def evolution_nocharge(cons,prim,sc1,sc2,a,alpha,curr,next):

# 	# Pi, Phi across space (i, func) at a given timestep
# 	u = cons[curr,:,:]

# 	phi1 = sc1[curr,:,:]
# 	phi2 = sc2[curr,:,:]

# 	# print(phi2[0,:])

# 	rhos = prim[curr,:,1] # used for Newton-Rhapson method as guesses

# 	# ========== RK3 ========== #
# 	h = dt

# 	# cell reconstruction for the grid
# 	# compute uL and uR for a given u; this is done for all space (shape: NUM_SPOINTS,2)
# 	uL = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
# 	uR = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
# 	cell_reconstruction(u, uL, uR)

# 	# print(uR[NUM_SPOINTS-10:,:])
# 	F1 = np.zeros((NUM_SPOINTS,2))
# 	F2 = np.zeros((NUM_SPOINTS,2))

# 	# calculate all fluxes
# 	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
# 		F1[i,:], F2[i,:] = findFluxes(i, uL[i,:], uR[i,:], rhos[i], rhos[i-1], rhos[i+1])

	
# 	k1u = np.zeros((NUM_SPOINTS,2))
# 	k2u = np.zeros((NUM_SPOINTS,2))
# 	k3u = np.zeros((NUM_SPOINTS,2))

# 	k1phi1 = np.zeros((NUM_SPOINTS,3))
# 	k2phi1 = np.zeros((NUM_SPOINTS,3))
# 	k3phi1 = np.zeros((NUM_SPOINTS,3))

# 	k1phi2 = np.zeros((NUM_SPOINTS,3))
# 	k2phi2 = np.zeros((NUM_SPOINTS,3))
# 	k3phi2 = np.zeros((NUM_SPOINTS,3))

# 	k1a = np.zeros(NUM_SPOINTS)  
# 	k2a = np.zeros(NUM_SPOINTS)
# 	k3a = np.zeros(NUM_SPOINTS)

# 	# calculate k1

# 	calculate_rk3_kval(h, k1u, k1a, k1phi1, k1phi2, alpha[curr,:], a[curr,:], u, F1, F2, phi1, phi2, rhos)

# 	# intermediate calculations
# 	uk1 = u+k1u
# 	# floor the updated u values
# 	checkFloor(uk1[:,Pi_i])
# 	checkFloor(uk1[:,Phi_i])

# 	# inner boundary for updated u values
# 	Phi2 = uk1[2,Pi_i]
# 	Phi3 = uk1[3,Phi_i]
# 	r2 = R(2)
# 	r3 = R(3)
# 	uk1[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
# 	uk1[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
# 	initializeEvenVPs(uk1)

# 	# outer boundary for updated u values
# 	uk1[NUM_SPOINTS-1] = uk1[NUM_SPOINTS-3]
# 	uk1[NUM_SPOINTS-2] = uk1[NUM_SPOINTS-3]

# 	# calculate outer k1 for outer points
# 	i = NUM_SPOINTS-2
# 	r_p1h = R(i+1/2)

# 	k1a[i] 			= h * da_dt(i, (alpha[curr,i]+alpha[curr,i-1])/2, a[curr,i], u[i,:], phi1[i,:], phi2[i,:])

# 	k1phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1[i,:])
# 	k1phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1[i,:], phi1[i-1,:], phi1[i-2,:])
# 	k1phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2[i,:])
# 	k1phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2[i,:], phi2[i-1,:], phi2[i-2,:])

# 	i = NUM_SPOINTS-1
# 	r_p1h = R(i+1/2)

# 	k1a[i] 			= h * da_dt(i, 1/a[curr,i], a[curr,i], u[i,:], phi1[i,:], phi2[i,:])

# 	k1phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1[i,:])
# 	k1phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1[i,:], phi1[i-1,:], phi1[i-2,:])
# 	k1phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2[i,:])
# 	k1phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2[i,:], phi2[i-1,:], phi2[i-2,:])

# 	# construct shifted arrays
# 	ak1 = a[curr,:]+k1a
# 	phi1k1 = phi1+k1phi1
# 	phi2k1 = phi2+k1phi2

# 	# calculate shifted X for phi1, phi2 arrays
# 	i = NUM_SPOINTS-2
# 	r_p1h = R(i+1/2)
# 	phi1k1[i,X_i] = outer_X(r_p1h, phi1k1[i,:])
# 	phi2k1[i,X_i] = outer_X(r_p1h, phi2k1[i,:])

# 	i = NUM_SPOINTS-1
# 	r_p1h = R(i+1/2)
# 	phi1k1[i,X_i] = outer_X(r_p1h, phi1k1[i,:])
# 	phi2k1[i,X_i] = outer_X(r_p1h, phi2k1[i,:])
		
# 	ak1[NUM_VPOINTS] = 1

# 	# initialize all virtual points
# 	initializeEvenVPs(ak1)
# 	initializeEvenVPs(phi1k1[:,phi_i], "staggered")
# 	initializeOddVPs(phi1k1[:,X_i])
# 	initializeEvenVPs(phi1k1[:,Y_i], "staggered")
	
# 	initializeEvenVPs(phi2k1[:,phi_i], "staggered")
# 	initializeOddVPs(phi2k1[:,X_i])
# 	initializeEvenVPs(phi2k1[:,Y_i], "staggered")

# 	# cell reconstruction
# 	uLk1 = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
# 	uRk1 = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
# 	cell_reconstruction(uk1,uLk1,uRk1)

# 	F1k1 = np.zeros((NUM_SPOINTS,2))
# 	F2k1 = np.zeros((NUM_SPOINTS,2))

# 	# calculate all fluxes
# 	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
# 		F1k1[i,:], F2k1[i,:] = findFluxes(i, uLk1[i,:], uRk1[i,:], rhos[i], rhos[i-1], rhos[i+1])

# 	# calculate k2
# 	calculate_rk3_kval(h, k2u, k2a, k2phi1, k2phi2, alpha[curr,:], ak1, uk1, F1k1, F2k1, phi1k1, phi2k1, rhos)

# 	uk12 = u+(k1u + k2u)/4
# 	# floor the updated u values
# 	checkFloor(uk12[:,Pi_i])
# 	checkFloor(uk12[:,Phi_i])

# 	# inner boundary for updated u values
# 	Phi2 = uk12[2,Pi_i]
# 	Phi3 = uk12[3,Phi_i]
# 	r2 = R(2)
# 	r3 = R(3)
# 	uk12[1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
# 	uk12[1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
# 	initializeEvenVPs(uk12)

# 	# outer boundary for updated u values
# 	uk12[NUM_SPOINTS-1] = uk12[NUM_SPOINTS-3]
# 	uk12[NUM_SPOINTS-2] = uk12[NUM_SPOINTS-3]

# 	# k2 vals for outer points: a, scalar fields
# 	i = NUM_SPOINTS-2
# 	r_p1h = R(i+1/2)

# 	k2a[i] = h * da_dt(i, (alpha[curr,i]+alpha[curr,i-1])/2, ak1[i], uk1[i,:], phi1k1[i,:], phi2k1[i,:])

# 	k2phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1k1[i,:])
# 	k2phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1k1[i,:], phi1k1[i-1,:], phi1k1[i-2,:])
# 	k2phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2k1[i,:])
# 	k2phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2k1[i,:], phi2k1[i-1,:], phi2k1[i-2,:])
	
# 	i = NUM_SPOINTS-1
# 	r_p1h = R(i+1/2)

# 	k2a[i] = h * da_dt(i, 1/ak1[i], ak1[i], uk1[i,:], phi1k1[i,:], phi2k1[i,:])

# 	k2phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1k1[i,:])
# 	k2phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1k1[i,:], phi1k1[i-1,:], phi1k1[i-2,:])
# 	k2phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2k1[i,:])
# 	k2phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2k1[i,:], phi2k1[i-1,:], phi2k1[i-2,:])

# 	# compute shifted arrays
# 	ak12 = a[curr,:]+(k1a + k2a)/4
# 	phi1k12 = phi1+(k1phi1 + k2phi1)/4
# 	phi2k12 = phi2+(k1phi2 + k2phi2)/4
		
# 	# calculate shifted X for phi1, phi2 arrays
# 	i = NUM_SPOINTS-2
# 	r_p1h = R(i+1/2)
# 	phi1k12[i,X_i] = outer_X(r_p1h, phi1k12[i,:])
# 	phi2k12[i,X_i] = outer_X(r_p1h, phi2k12[i,:])

# 	i = NUM_SPOINTS-1
# 	r_p1h = R(i+1/2)
# 	phi1k12[i,X_i] = outer_X(r_p1h, phi1k12[i,:])
# 	phi2k12[i,X_i] = outer_X(r_p1h, phi2k12[i,:])


# 	ak12[NUM_VPOINTS] = 1

# 	# initialize all virtual points
# 	initializeEvenVPs(ak1)
# 	initializeEvenVPs(phi1k12[:,phi_i], "staggered")
# 	initializeOddVPs(phi1k12[:,X_i])
# 	initializeEvenVPs(phi1k12[:,Y_i], "staggered")

# 	initializeEvenVPs(phi2k12[:,phi_i], "staggered")
# 	initializeOddVPs(phi2k12[:,X_i])
# 	initializeEvenVPs(phi2k12[:,Y_i], "staggered")

# 	uLk12 = np.zeros((NUM_SPOINTS,2))		# u as calculated at the cell boarder using the left values
# 	uRk12 = np.zeros((NUM_SPOINTS,2)) 		# u as calculated at C.B. using the right values
# 	cell_reconstruction(uk12,uLk12,uRk12)

# 	F1k12 = np.zeros((NUM_SPOINTS,2))
# 	F2k12 = np.zeros((NUM_SPOINTS,2))

# 	# calculate all fluxes
# 	for i in range(NUM_VPOINTS, NUM_SPOINTS-2): # needed at NUM_VPOINTS to calculate du_dt at i = NUM_VPOINTS+1
# 		F1k12[i,:], F2k12[i,:] = findFluxes(i, uLk12[i,:], uRk12[i,:], rhos[i], rhos[i-1], rhos[i+1])
		

# 	calculate_rk3_kval(h, k3u, k3a, k3phi1, k3phi2, alpha[curr,:], ak12, uk12, F1k12, F2k12, phi1k12, phi2k12, rhos)


# 	i = NUM_SPOINTS-2
# 	r_p1h = R(i+1/2)
# 	k3a[i] = h * da_dt(i, (alpha[curr,i]+alpha[curr,i-1])/2, ak12[i], uk12[i,:], phi1k12[i,:], phi2k12[i,:])

# 	k3phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1k12[i,:])
# 	k3phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1k12[i,:], phi1k12[i-1,:], phi1k12[i-2,:])
# 	k3phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2k12[i,:])
# 	k3phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2k12[i,:], phi2k12[i-1,:], phi2k12[i-2,:])

# 	i = NUM_SPOINTS-1
# 	r_p1h = R(i+1/2)
# 	k3a[i] = h * da_dt(i, 1/ak12[i], ak12[i], uk12[i,:], phi1k12[i,:], phi2k12[i,:])

# 	k3phi1[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi1k12[i,:])
# 	k3phi1[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi1k12[i,:], phi1k12[i-1,:], phi1k12[i-2,:])
# 	k3phi2[i,phi_i] = h * outer_dphi_dt	(r_p1h, phi2k12[i,:])
# 	k3phi2[i,Y_i] 	= h * outer_dY_dt	(r_p1h, phi2k12[i,:], phi2k12[i-1,:], phi2k12[i-2,:])

# 	# calculate a values across the entire array
# 	for i in range(NUM_VPOINTS+1,NUM_SPOINTS):
# 		a[next,i] = a[curr,i] + 1/6 * (k1a[i] + k2a[i] + 4*k3a[i])
	
# 	a[next,NUM_VPOINTS] = 1
# 	initializeEvenVPs(a[next,:])

# 	# calculate updated functions

# 	# conservative functions
# 	for i in range(NUM_VPOINTS+1,NUM_SPOINTS-2):
# 		cons[next,i,:] = cons[curr,i,:] + 1/6 * (k1u[i,:] + k2u[i,:] + 4*k3u[i,:])
		
# 	# scalar fields
# 	for i in range(NUM_VPOINTS, NUM_SPOINTS):
# 		if i == NUM_SPOINTS-2 or i == NUM_SPOINTS-1:
# 			# determine lphi, Y using predetermined k values
# 			sc1[next, i, phi_i] = sc1[curr, i, phi_i] + 1/6 * (k1phi1[i, phi_i] + k2phi1[i, phi_i] + 4 * k3phi1[i,phi_i])
# 			sc1[next, i, Y_i] 	= sc1[curr, i, Y_i]	  + 1/6 * (k1phi1[i, Y_i]   + k2phi1[i, Y_i]   + 4 * k3phi1[i,Y_i])

# 			sc2[next, i, phi_i] = sc2[curr, i, phi_i] + 1/6 * (k1phi2[i, phi_i] + k2phi2[i, phi_i] + 4 * k3phi2[i,phi_i])
# 			sc2[next, i, Y_i] 	= sc2[curr, i, Y_i]   + 1/6 * (k1phi2[i, Y_i]   + k2phi2[i, Y_i]   + 4 * k3phi2[i,Y_i])

# 			# compute X algebraically using those new values
# 			sc1[next, i, X_i] 	= outer_X(R(i+1/2), sc1[next, i, :])
# 			sc2[next, i, X_i] 	= outer_X(R(i+1/2), sc2[next, i, :])
# 		else:
# 			sc1[next,i,:] = sc1[curr,i,:] + 1/6 * (k1phi1[i,:] + k2phi1[i,:] + 4*k3phi1[i,:])
# 			sc2[next,i,:] = sc2[curr,i,:] + 1/6 * (k1phi2[i,:] + k2phi2[i,:] + 4*k3phi2[i,:])

# 	checkFloor(cons[next,:,Pi_i])
# 	checkFloor(cons[next,:,Phi_i])

# 	cons[next, NUM_SPOINTS-1,:] = cons[next, NUM_SPOINTS-3,:]
# 	cons[next, NUM_SPOINTS-2,:] = cons[next, NUM_SPOINTS-3,:]

# 	u = cons[next,:,:]

# 	Phi2 = cons[next,2,Phi_i]
# 	Phi3 = cons[next,3,Phi_i]
# 	r2 = R(2)
# 	r3 = R(3)
# 	cons[next,1,Pi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)
# 	cons[next,1,Phi_i] = (Phi2*r3**2 - Phi3*r2**2)/(r3**2 - r2**2) # see eq. (78)

# 	# virtual points
# 	initializeEvenVPs(cons[next,:,Pi_i])
# 	initializeEvenVPs(cons[next,:,Phi_i])
	
# 	# inner boundary for scalar fields
# 	initializeEvenVPs(sc1[next,:,phi_i], "staggered")
# 	initializeOddVPs(sc1[next,:,X_i])
# 	initializeEvenVPs(sc1[next,:,Y_i], "staggered")

# 	initializeEvenVPs(sc2[next,:,phi_i], "staggered")
# 	initializeOddVPs(sc2[next,:,X_i])
# 	initializeEvenVPs(sc2[next,:,Y_i], "staggered")

# 	# ========== CALCULATE PRIMITIVES ========== #

# 	for i in range(NUM_VPOINTS, NUM_SPOINTS):
# 		u = cons[next,i,:]
# 		rho_ = rho(u, rhos[i])
# 		p = P(rho_)
# 		v = V(u,p)
# 		prim[next,i,P_i] = p
# 		prim[next,i,rho_i] = rho_
# 		prim[next,i,v_i] = v 

# 	initializeEvenVPs(prim[next,:,P_i])
# 	initializeEvenVPs(prim[next,:,rho_i])

# 	# ========== CALCULATE ALPHA ========== #

# 	# # calculate alpha at each cell boundary, storing it at the gridpoint to its left

# 	alpha[next,NUM_SPOINTS-1] = (1/2 * (a[next,NUM_SPOINTS-1] + a[next,NUM_SPOINTS-2]))**(-1) # average of last two a values
# 	for i in range(NUM_SPOINTS-1, NUM_VPOINTS, -1): # start at last boundary and move inwards
# 		phi1_p1h = sc1[next,i,:]
# 		phi1_m1h = sc1[next,i-1,:]
# 		phi2_p1h = sc2[next,i,:]
# 		phi2_m1h = sc2[next,i-1,:]

# 		phi1 = (phi1_p1h + phi1_m1h)/2
# 		phi2 = (phi2_p1h + phi2_m1h)/2

# 		alpha[next,i-1] = falpha(alpha[next,i], R(i), cons[next,i,:], prim[next,i,:], phi1, phi2, a[next,i])

# 	initializeEvenVPs(alpha[next,:], spacing="staggered")
