# Joe Nyhan, 30 June 2021; edited 20 July 2021, 20 September 2021
# For reading and plotting data fils from /data
#%%
from hd_params import NUM_SPOINTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift

#%%

output_number = 1
save_fig = 0

# dim = "s"
dim = "ringdown"

t = 2495

f_alpha = 0
f_a		= 0

f_Pi	= 0
f_Phi	= 0

f_P 	= 1
f_rho	= 0
f_v		= 0


# ========== READING IN FILES, PARAMETERS

with open(f"data/{output_number:d}-0params.txt", "r") as params:
	s = params.readline()
	interval = int(s.replace("Write interval	= ", ""))

dt = 0.01
i = round(t /(interval * dt))

# plotting parameters: t and r
rs = pd.read_csv(f"data/{output_number:d}-r.txt", header=None)
# ts = pd.read_csv(f"data/{output_number:d}-t.txt", header=None)


# Files
if f_alpha:
	df = pd.read_csv(f"data/{output_number:d}-alpha.txt", header=None)
	title = "$\\alpha$"
	save_name = "alpha"
if f_a:
	df = pd.read_csv(f"data/{output_number:d}-a.txt", header=None)
	title = "$a$"
	save_name = "a"
if f_Pi:
	df = pd.read_csv(f"data/{output_number:d}-Pi.txt", header=None)
	title = "$\\Pi$"
if f_Phi:
	df = pd.read_csv(f"data/{output_number:d}-Phi.txt", header=None)
	title = "$\\Phi$"
	save_name = "Phi"
if f_P:
	df = pd.read_csv(f"data/{output_number:d}-P.txt", header=None)
	title = "$P$"
	save_name = "P"
if f_rho:
	df = pd.read_csv(f"data/{output_number:d}-rho.txt", header=None)
	title = "$\\rho$"
	save_name = "rho"
if f_v:
	df = pd.read_csv(f"data/{output_number:d}-v.txt", header=None)
	title = "$v$"
	save_name = "v"

	
if dim == "s":
	h_axis = rs.to_numpy().flatten()
	v_axis = df.iloc[i,:].to_numpy()
	title += f", $t$ = {t:.1f}"
	h_label = "$r$"


if dim == "ringdown":
	title += f", r = 0"
	df = pd.read_csv(f"data/{output_number:d}-ringdown.txt", header=None)
	h_axis = df.iloc[:,1].to_numpy() # time
	v_axis = df.iloc[:,4].to_numpy() # pressure
	h_label = "t"



title += f", Trial: {output_number:d}"
save_name = f"plots/{output_number:d}-{save_name},t{t:.2f}.pdf"

plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)

		
if save_fig:
	plt.savefig(save_name)

# %%

# fft
N = len(v_axis)
FT = fftshift(fft(v_axis))

tmin = df.head(n=1).iloc[:,1].to_numpy()[0]
tmax = df.tail(n=1).iloc[:,1].to_numpy()[0]

dt = df.head(n=3).iloc[:,1].to_numpy()[2] - df.head(n=2).iloc[:,1].to_numpy()[1]


Df = 1/(tmax-tmin)
freqs = np.arange(-1/(2*dt), +1/(2*dt), Df)
plt.xscale("log")
plt.yscale("log")
plt.plot(freqs, np.abs(FT[1:])) # need to find a way to exclude one value in a better way


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