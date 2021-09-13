# Joe Nyhan, 30 June 2021
# Functions and operations for writing to file

from hd_params import *
from aux_dm.hd_equations import R

# TODO: create a file containing all written r values and t values

# initialization functions
def params_print():
	with open(f"{path}{output_number}-0params.txt", "w") as f:
		# commonly checked parameters
		f.write(f"Print interval	= {PRINT_INTERVAL}\n")
		f.write(f"dr				= {dr}\n")
		f.write(f"dt				= {dt}\n")
		f.write(f"rmin			= {rmin}\n")
		f.write(f"rmax			= {rmax}\n")
		f.write(f"gamma			= {gamma}\n")
		f.write(f"tmin			= {tmin}\n")
		f.write(f"tmax			= {tmax}\n")

		f.write("\n")

		f.write(f"Dark Matter		= {darkmatter}\n")
		f.write(f"EOS				= {eos}\n")

		if eos_UR:
			f.write(f"	Gamma			= {Gamma}\n")
		if eos_polytrope:
			f.write(f"	K			= {K}\n")
			f.write(f"	Gamma		= {Gamma}\n")
		f.write(f"\n")
		f.write(f"Primitive IC	= {PRIM_IC}\n")
		f.write(f"Scalar Field IC = {SF_IC}\n")

		# TODO: add print statments that print out the number of tsteps, rsteps, etc. that are actually saved to file

		# TODO: print out initial conditions for dark matter configuration



# actual printing
if record_data:

	# initialize output files
	Pi_out 		= open(f"{path}{output_number}-Pi.txt", "w")
	Phi_out 	= open(f"{path}{output_number}-Phi.txt", "w")
	P_out 		= open(f"{path}{output_number}-P.txt", "w")
	rho_out 	= open(f"{path}{output_number}-rho.txt", "w")
	v_out 		= open(f"{path}{output_number}-v.txt", "w")

	phi1_out	= open(f"{path}{output_number}-phi1.txt", "w")
	X1_out  	= open(f"{path}{output_number}-X1.txt", "w")
	Y1_out 		= open(f"{path}{output_number}-Y1.txt", "w")

	phi2_out	= open(f"{path}{output_number}-phi2.txt", "w")
	X2_out  	= open(f"{path}{output_number}-X2.txt", "w")
	Y2_out 		= open(f"{path}{output_number}-Y2.txt", "w")

	A_r_out	    = open(f"{path}{output_number}-A_r.txt", "w")
	z_out  	    = open(f"{path}{output_number}-z.txt", "w")
	Omega_out 	= open(f"{path}{output_number}-Omega.txt", "w")

	alpha_out 	= open(f"{path}{output_number}-alpha.txt", "w")
	a_out 		= open(f"{path}{output_number}-a.txt", "w")

	files = [
		Pi_out,
		Phi_out,
		
		P_out,
		rho_out,
		v_out,

		phi1_out,
		X1_out,
		Y1_out,

		phi2_out,
		X2_out,
		Y2_out,

		A_r_out,
		z_out,
		Omega_out,

		alpha_out,
		a_out,
	]

	# write all r values that correspond to spatial values in other files
	rs_out = open(f"{path}{output_number}-r.txt", "w")

	# for i in range(int(rmin/dr), int(rmax/dr), R_STEP_FILE):
	for i in range(NUM_SPOINTS):
		if i == NUM_SPOINTS-1:
			rs_out.write(f"{R(i):.15e}")
		else:
			rs_out.write(f"{R(i):.15e},")

	rs_out.close()

	# file containing all t values
	ts_out = open(f"{path}{output_number}-t.txt", "w")


	# print out all parameters
	params_print()


if record_ringdown:
	ringdown = open(f"{path}{output_number}-ringdown.txt", "w")


def output_write(array, n):
	"""
	write all relevant data to corresponding files
	funcs: a list containing all relevant functions, parallel to the files array
	"""
	if record_data and ((n+1) % RECORD_INTERVAL == 0 or n == 0):
		for f in range(len(files)):
			curr = array[f]
			output = files[f]
			if n==0: pass
			else: output.write("\n")
			# for i in range(int(rmin/dr), int(rmax/dr), R_STEP_FILE):
			for i in range(NUM_SPOINTS):
				if i == NUM_SPOINTS-1:
					output.write(f"{curr[i]:.15e}")
				else:
					output.write(f"{curr[i]:.15e},")
		
		if n == 0:
			ts_out.write(f"{0:.2f},")
		else:
			ts_out.write(f"{(n+1)*dt:.2f},")
		
	
	if record_ringdown and ((n+1) % RING_INTERVAL == 0 or n == 0):
		i = NUM_VPOINTS # record at r = 0
		ringdown.write(f"{n}, {n*dt:.2f},")
		for n in range(len(array)):
			curr = array[n]
			ringdown.write(f"{curr[i]:.15e},")
		ringdown.write("\n")


def output_close():
	"""
	close the output files for all variables
	"""
	if record_data:
		Pi_out.close()
		Phi_out.close()

		P_out.close()
		rho_out.close()
		v_out.close()

		phi1_out.close()
		X1_out.close()
		Y1_out.close()

		phi2_out.close()
		X2_out.close()
		Y2_out.close()

		A_r_out.close()
		z_out.close()
		Omega_out.close()

		alpha_out.close()
		a_out.close()

		ts_out.close()

	if record_ringdown:
		ringdown.close()

	return

	# print out the r value that matches up with each index