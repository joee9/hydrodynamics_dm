# Joe Nyhan, 30 June 2021; edited 20 July 2021, 20 September 2021
# For reading and plotting data fils from /data
#%%
from hd_params import NUM_SPOINTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift

#%%

output_number = 2

# dim = "s"
dim = "ringdown"
FFT = 1

save_fig = 0

t = 3500
# for "s", this is the time snapshot.
# for "ringdown", the max time plotted to. -1 will plot all values

f_Pi	= 0
f_Phi	= 0

f_P 	= 1
f_rho	= 0
f_v		= 0

f_alpha = 0
f_a		= 0


# ========== READING IN FILES, PARAMETERS

path = f"data/{output_number:d}"

with open(f"{path:s}-0params.txt", "r") as params:
	s = params.readline()
	interval = int(s.replace("Write interval	= ", ""))
	s = params.readline()
	ring_interval = int(s.replace("Ring interval\t\t= ", ""))

dt = 0.01
i = round(t /(interval * dt))

# Plotting parameters
if f_Pi:
	file_path = f"{path:s}-Pi.txt"
	title = "$\\Pi$"
	save_name = "Pi"
	ring_idx = 2
if f_Phi:
	file_path = f"{path:s}-Phi.txt"
	title = "$\\Phi$"
	save_name = "Phi"
	ring_idx = 3
if f_P:
	file_path = f"{path:s}-P.txt"
	title = "$P$"
	save_name = "P"
	ring_idx = 4
if f_rho:
	file_path = f"{path:s}-rho.txt"
	title = "$\\rho$"
	save_name = "rho"
	ring_idx = 5
if f_v:
	file_path = f"{path:s}-v.txt"
	title = "$v$"
	save_name = "v"
	ring_idx = 6
if f_alpha:
	file_path = f"{path}-alpha.txt"
	title = "$\\alpha$"
	save_name = "alpha"
	ring_idx = 7
if f_a:
	file_path = f"{path:s}-a.txt"
	title = "$a$"
	save_name = "a"
	ring_idx = 8
	
title += f", Trial: {output_number:d}"
save_name = f"plots/{output_number:d}-{save_name}"

if dim == "s":

	df = pd.read_csv(file_path, header=None)
	rs = pd.read_csv(f"{path:s}-r.txt", header=None)

	h_axis = rs.to_numpy().flatten()
	v_axis = df.iloc[i,:].to_numpy()
	h_label = "$r$"

	title += f", $t$ = {t:.1f}"
	save_name += f",{t:.2f}"

if dim == "ringdown":

	df = pd.read_csv(f"{path:s}-ringdown.txt", header=None)

	tmin = df.head(n=1).iloc[:,1].to_numpy()[0]
	tmax = df.tail(n=1).iloc[:,1].to_numpy()[0]

	if t == -1: 
		last_idx = round(tmax/(ring_interval * dt))
	else:
		last_idx = round(t/(ring_interval * dt))
		tmax = t

	title += f", $r = 0$"
	h_axis = df.iloc[:last_idx,1].to_numpy() # time
	v_axis = df.iloc[:last_idx,ring_idx].to_numpy() # pressure
	h_label = "t"

	save_name += f",ringdown"

	if FFT:

		N = len(v_axis)
		FT = fftshift(fft(v_axis))

		dt = df.head(n=3).iloc[:,1].to_numpy()[2] - df.head(n=2).iloc[:,1].to_numpy()[1]

		Df = 1/(tmax-tmin)
		freqs = np.arange(-1/(2*dt), +1/(2*dt), Df) * 2*np.pi
		plt.xscale("log")
		plt.yscale("log")

		h_axis = freqs
		v_axis = np.abs(FT)

		save_name += f",fft"
		title += f", FFT"
		h_label = f"$f$"

save_name += f".pdf"

# sly_vals = [
# 	0.148462348345648,
# 	0.209995270508381,
# 	0.254676789337527,
# 	0.327847798115150,
# 	0.364224267545439,
# 	0.440072685837974,
# 	0.498652374377900
# ]
# sly_vals = np.array(sly_vals)
# # sly_vals = sly_vals/(2*np.pi)

# for i in range(len(sly_vals)):
# 	plt.plot([sly_vals[i], sly_vals[i]], [1e-7,1e-1])



plt.title(title)
# plt.xlim(1e-1,1e0)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)
		
if save_fig:
	plt.savefig(save_name)

# %%

# static solutions
eos_UR = 0
eos_polytrope = 0
eos_SLy = 1

p0 = 1e-4
vphi0 = 1e-3
Lambda = 0
dr = 0.02
rmin = 0
rmax = 100

f_P 	= 1
f_rho	= 0

if eos_polytrope:
	eos = "polytrope_"
	K = 100
	Gamma = 2
	params = f"K{K:.1f}_gamma{Gamma:.1f}_"
elif eos_SLy:
	eos = "SLy_"
	params = ""

s = f"input/{eos}{params}p{p0:.8f}_dr{dr:.3f}_"

if f_P:
	s += "P.txt"
	df = pd.read_csv(s, header=None)
	title = "$P$"
elif f_rho:
	s += "rho.txt"
	df = pd.read_csv(s, header=None)
	title = "$\\rho$"

rs = np.arange(rmin, rmax, dr)
h_axis = rs
v_axis = df.iloc[:,0].to_numpy()[:len(rs)]
h_label = "$r$"

plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)