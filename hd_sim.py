# Joe Nyhan, 30 June 2021
# The simulation of a neutron star.
#     auxillary files: aux folder

# jitted code; the evolution function, in this file is jitted for much more speed than before

#%%
from hd_params           import *

from aux.hd_eos          import *
from aux.hd_equations    import *
from aux.hd_ic           import *
from aux.hd_ops          import *
from aux.hd_riemann      import *
from aux.hd_write        import *
from aux.hd_evolution    import *

NUM_STORED = T_STEPS_ARRAY


prim = np.zeros((NUM_STORED,NUM_SPOINTS,3))    # contains primitive variables: P, rho, v (0,1,2)
cons = np.zeros((NUM_STORED,NUM_SPOINTS,2))    # contains conservative variables: Pi, Phi (0,1)    

# gravity fields
alpha = np.zeros((NUM_STORED,NUM_SPOINTS))        
a = np.zeros((NUM_STORED,NUM_SPOINTS))

sc1 = np.zeros((NUM_STORED, NUM_SPOINTS, 3)) # X_i, Y_i indexed as 1, 2 respectiely; for both scalar fields
sc2 = np.zeros((NUM_STORED, NUM_SPOINTS, 3))

rs = np.zeros(NUM_SPOINTS)
for i in range(NUM_SPOINTS):
    rs[i] = R(i)

## ========== INITIAL DATA ========== #
# initialize primitive variables
prim[0,:,P_i] = initial_p_vals
prim[0,:,rho_i] = initial_rho_vals
prim[0,:,v_i] = np.zeros(NUM_SPOINTS) # velocity initialized to zero

# implement a floor on primitives
prim[0,:,P_i] += 1E-15
prim[0,:,rho_i] += 1E-15

sc1[0,:,phi_i] = initial_phi1_vals
sc1[0,:,X_i] = initial_X1_vals
sc1[0,:,Y_i] = initial_Y1_vals

sc2[0,:,phi_i] = initial_phi2_vals
sc2[0,:,X_i] = initial_X2_vals
sc2[0,:,Y_i] = initial_Y2_vals

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
calc_gravity_functions(a[0,:], alpha[0,:], cons[0,:,:], prim[0,:,:], sc1[0,:,:], sc2[0,:,:], prim[0,:,rho_i])

# initialize all virtual points
initializeEvenVPs(prim[0,:,Pi_i])
initializeEvenVPs(prim[0,:,rho_i])

initializeEvenVPs(cons[0,:,Pi_i])
initializeEvenVPs(cons[0,:,Phi_i])

initializeEvenVPs(sc1[0,:,phi_i], spacing="staggered")
initializeOddVPs (sc1[0,:,X_i])
initializeEvenVPs(sc1[0,:,Y_i], spacing="staggered")

initializeEvenVPs(sc2[0,:,phi_i], spacing="staggered")
initializeOddVPs (sc2[0,:,X_i])
initializeEvenVPs(sc2[0,:,Y_i], spacing="staggered")

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

        e_string = ""
        hh = elapsed_time // 3600
        mm = elapsed_time // 60
        ss = elapsed_time % 60

        if hh > 0:
            e_string += f"{hh:d}:"

        e_string += f"{mm - hh*60:02d}:{ss:02d}"
        secs = curr_time - start_time
        start_time = curr_time
        elapsed_time = curr_time - first_time
        print(f"{n}, {n*dt:.2f}, {secs/PRINT_INTERVAL:.4f}/timestep, Elapsed time: {e_string:s}")
    
    t = n % NUM_STORED
    curr = t
    next = t + 1
    if t == NUM_STORED-1:
        next = 0

    if int_rk3:
        evolution_rk3(cons,prim,sc1,sc2,a,alpha,curr,next)

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

        alpha[next,:], 
        a[next,:]
    ]


    output_write(arrays,n+1)


# close all output files
last_time = process_time()
output_close()
elapsed_time = int(last_time - first_time)

e_string = ""
hh = elapsed_time // 3600
mm = elapsed_time // 60
ss = elapsed_time % 60

if hh > 0:
    e_string += f"{hh:d}:"

e_string += f"{mm - hh*60:02d}:{ss:02d}"

print(f"Total time: {e_string:s}\n\nFINISHED.")
