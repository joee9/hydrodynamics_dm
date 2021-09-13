# Joe Nyhan, 30 June 2021
# The simulation of a neutron star.
# 	auxillary files: aux_dm folder

# jitted code; the evolution function, in this file is jitted for much more speed than before

#%%
from hd_params import *

from aux_dm.hd_eos 			import *
from aux_dm.hd_equations 	import *
from aux_dm.hd_ic 			import *
from aux_dm.hd_ops 			import *
from aux_dm.hd_riemann 		import *
from aux_dm.hd_write 		import *
from aux_dm.hd_evolution 	import *


NUM_STORED = T_STEPS_ARRAY


prim = np.zeros((NUM_STORED,NUM_SPOINTS,3))	# contains primitive variables: P, rho, v (0,1,2)
cons = np.zeros((NUM_STORED,NUM_SPOINTS,2))	# contains conservative variables: Pi, Phi (0,1)	

# scalar fields 1 and 2
sc1 = np.zeros((NUM_STORED, NUM_SPOINTS, 3)) # X_i, Y_i indexed as 1, 2 respectiely; for both scalar fields
sc2 = np.zeros((NUM_STORED, NUM_SPOINTS, 3))

# charge fields
A_r = np.zeros((NUM_STORED, NUM_SPOINTS))
z = np.zeros((NUM_STORED, NUM_SPOINTS))
Omega = np.zeros((NUM_STORED, NUM_SPOINTS))

# gravity fields
alpha = np.zeros((NUM_STORED,NUM_SPOINTS))		
a = np.zeros((NUM_STORED,NUM_SPOINTS))


rs = np.zeros(NUM_SPOINTS)
for i in range(NUM_SPOINTS):
	rs[i] = R(i)

## ========== INITIAL DATA ========== #
# initialize primitive variables
prim[0,:,P_i] = initial_p_vals
prim[0,:,rho_i] = initial_rho_vals
prim[0,:,v_i] = np.zeros(NUM_SPOINTS) # velocity initialized to zero

# implement a floor on primitives
prim[0,:,P_i] += 1E-13
prim[0,:,rho_i] += 1E-13

# intialize scalar fields
sc1[0,:,phi_i] = initial_phi1_vals
sc1[0,:,X_i] = initial_X1_vals
sc1[0,:,Y_i] = initial_Y1_vals

sc2[0,:,phi_i] = initial_phi2_vals
sc2[0,:,X_i] = initial_X2_vals
sc2[0,:,Y_i] = initial_Y2_vals

A_r[0,:] = initial_A_r_vals
z[0,:] = initial_z_vals
Omega[0,:] = initial_Omega_vals

# conservative data
for i in range(NUM_SPOINTS): 
	cons[0,i,Pi_i] = Pi(prim[0,i,:])
	cons[0,i,Phi_i] = Phi(prim[0,i,:])

# make sure none of the primitive or conservative variables (except for velocity) fall below the floor
checkFloor(prim[0,:,P_i])
checkFloor(prim[0,:,rho_i])
checkFloor(cons[0,:,Pi_i])
checkFloor(cons[0,:,Phi_i])

# construct gravity equations at n = 0

# calculate a across the spatial points at t = 0

a[0,NUM_VPOINTS] = 1
h = dr

for i in range(NUM_VPOINTS,NUM_SPOINTS-1):
	r_ = R(i)
	a_ = a[0,i]

	u = cons[0,i,:]
	u_p1 = cons[0,i+1,:]
	u_p1h = (u+u_p1)/2

	phi1_p1h = sc1[0,i,:]
	phi1_m1h = sc1[0,i-1,:]
	phi2_p1h = sc2[0,i,:]
	phi2_m1h = sc2[0,i-1,:]
	phi1 = (phi1_p1h + phi1_m1h)/2
	phi2 = (phi2_p1h + phi2_m1h)/2

	z_p1h = z[0,i]
	z_m1h = z[0,i-1]
	z_ = (z_p1h + z_m1h)/2

	k1 = h * fa0(r_,a_,u,phi1,phi2,z_)
	k2 = h * fa0(r_ + h/2, a_ + k1/2, u_p1h,phi1_p1h, phi2_p1h, z_p1h)

	a[0,i+1] = a[0,i] + k2


# calculate alpha at each cell boundary, storing it at the gridpoint to its left
alpha[0,NUM_SPOINTS-1] = (1/2 * (a[0,NUM_SPOINTS-1] + a[0,NUM_SPOINTS-2]))**(-1) # average of last two a values
for i in range(NUM_SPOINTS-1, NUM_VPOINTS, -1): # start at last boundary and move inwards
	phi1_p1h = sc1[0,i,:]
	phi1_m1h = sc1[0,i-1,:]
	phi2_p1h = sc2[0,i,:]
	phi2_m1h = sc2[0,i-1,:]

	phi1 = (phi1_p1h + phi1_m1h)/2
	phi2 = (phi2_p1h + phi2_m1h)/2

	z_p1h = z[0,i]
	z_m1h = z[0,i-1]
	z_ = (z_p1h + z_m1h)/2

	alpha[0,i-1] = falpha(alpha[0,i], R(i), cons[0,i,:], prim[0,i,:], phi1, phi2, z_, a[0,i])
	# alpha[0,i] = falpha(alpha[0,i+1], R(i), prim[0,i,P_i], prim[0,i,v_i], cons[0,i,Pi_i], cons[0,i,Phi_i], a[0,i])

# initialize all virtual points
initializeEvenVPs(prim[0,:,Pi_i])
initializeEvenVPs(prim[0,:,rho_i])

initializeEvenVPs(cons[0,:,Pi_i])
initializeEvenVPs(cons[0,:,Phi_i])

initializeEvenVPs(a[0,:])
initializeEvenVPs(alpha[0,:], spacing="staggered")


# write initial data to file

arrays = [
	cons[0,:,Pi_i], 
	cons[0,:,Phi_i], 

	prim[0,:,P_i], 
	prim[0,:,rho_i], 
	prim[0,:,v_i], 

	sc1[0,:,phi_i],
	sc1[0,:,X_i],
	sc1[0,:,Y_i],

	sc2[0,:,phi_i],
	sc2[0,:,X_i],
	sc2[0,:,Y_i],

	A_r[0,:],
	z[0,:],
	Omega[0,:],

	alpha[0,:], 
	a[0,:]
]

output_write(arrays,0)

# ========== TIME EVOLUTION ========== #

first_time = process_time()
start_time = process_time()
curr_time = process_time()

for n in range(NUM_TPOINTS-1):

	if n % PRINT_INTERVAL == 0:

		curr_time = process_time()
		elapsed_time = int(curr_time - first_time)
		e_string = f"{elapsed_time // 60:02d}:{elapsed_time%60:02d}"
		secs = curr_time - start_time
		start_time = curr_time
		elapsed_time = curr_time - first_time
		print(f"{n}, {n*dt:.2f}, {secs/PRINT_INTERVAL:.4f}/timestep, Elapsed time: {e_string:s}")
	
	t = n % NUM_STORED
	curr = t
	next = t + 1
	if t == NUM_STORED-1:
		next = 0

	evolution_charge(cons,prim,sc1,sc2,A_r,z,Omega,a,alpha,curr,next)

	arrays = [
		cons[next,:,Pi_i], 
		cons[next,:,Phi_i], 

		prim[next,:,P_i], 
		prim[next,:,rho_i], 
		prim[next,:,v_i], 

		sc1[next,:,phi_i],
		sc1[next,:,X_i],
		sc1[next,:,Y_i],

		sc2[next,:,phi_i],
		sc2[next,:,X_i],
		sc2[next,:,Y_i],

		A_r[next,:],
		z[next,:],
		Omega[next,:],

		alpha[next,:], 
		a[next,:]
	]


	output_write(arrays,n+1)


# close all output files
last_time = process_time()
output_close()
elapsed_time = int(last_time - first_time)
e_string = f"{elapsed_time // 60:02d}:{elapsed_time%60:02d}"

print(f"Total time: {e_string:s}\n\nFINISHED.")