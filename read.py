# Joe Nyhan, 30 June 2021; edited 20 July 2021
# For reading and plotting data fils from /data
#%%
from hd_params import NUM_SPOINTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft,fftshift

#%%

# NUM_SPOINTS = 5000
output_number = 2
save_fig = 0

dim = "s"
# dim = "ringdown"
t = 45

f_alpha = 0
f_a		= 0

f_Pi	= 0
f_Phi	= 0

f_P 	= 1
f_rho	= 0
f_v		= 0

# f_phi1	= 0
# f_X1 	= 0
# f_Y1	= 0

# f_phi2	= 0
# f_X2 	= 0
# f_Y2	= 0

# f_A_r	= 0
# f_z 	= 0
# f_Omega = 0

# absphi	= 1

with open(f"data/{output_number:d}-0params.txt", "r") as params:
	s = params.readline()
	interval = int(s.replace("Write interval	= ", ""))

print(interval)

# s = "Quickly"
# str = s.replace("ly", "")
# print(str)

dt = 0.01
# interval = 100
i = round(t /(interval * dt))
# i = 320



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
# if f_phi1:
# 	df = pd.read_csv(f"data/{output_number:d}-phi1.txt", header=None)
# 	title = "$\\varphi_1$"
# 	save_name = "varphi1"
# if f_X1:
# 	df = pd.read_csv(f"data/{output_number:d}-X1.txt", header=None)
# 	title = "$X_1$"
# 	save_name = "X1"
# if f_Y1:
# 	df = pd.read_csv(f"data/{output_number:d}-Y1.txt", header=None)
# 	title = "$Y_1$"
# 	save_name = "Y1"
# if f_phi2:
# 	df = pd.read_csv(f"data/{output_number:d}-phi2.txt", header=None)
# 	title = "$\\varphi_2$"
# 	save_name = "varphi2"
# if f_X2:
# 	df = pd.read_csv(f"data/{output_number:d}-X2.txt", header=None)
# 	title = "$X_2$"
# 	save_name = "X2"
# if f_Y2:
# 	df = pd.read_csv(f"data/{output_number:d}-Y2.txt", header=None)
# 	title = "$Y_2$"
# 	save_name = "Y2"
# if f_A_r:
# 	df = pd.read_csv(f"data/{output_number:d}-A_r.txt", header=None)
# 	title = "$A_r$"
# 	save_name = "Ar"
# if f_z:
# 	df = pd.read_csv(f"data/{output_number:d}-z.txt", header=None)
# 	title = "$Z$"
# 	save_name = "z"
# if f_Omega:
# 	df = pd.read_csv(f"data/{output_number:d}-Y2.txt", header=None)
# 	title = "$\\Omega$"
# 	save_name = "Omega"
# if absphi:
# 	phi1 = pd.read_csv(f"data/{output_number:d}-phi1.txt", header=None)
# 	phi2 = pd.read_csv(f"data/{output_number:d}-phi2.txt", header=None)
# 	phi1_vals = phi1.iloc[i,:].to_numpy()
# 	phi2_vals = phi2.iloc[i,:].to_numpy()

# 	mag = []
# 	for k in range(NUM_SPOINTS):
# 		varphi1 = phi1_vals[k]
# 		varphi2 = phi2_vals[k]
# 		# print(f"{k=}{varphi1=}{varphi2=}")
# 		mag.append(np.sqrt(varphi1**2 + varphi2**2))
	
# 	title = "$|\\varphi|$"
# 	save_name = "absvarphi"

	
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



# plt.ylim([0,1.1E-6])
# plt.xlim([0,20])
title += f", Trial: {output_number:d}"
# save_name = f"plots/{output_number:d}-{save_name},t{t:.2f}.pdf"
# plt.yscale("log")
plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)

		
if save_fig:
	plt.savefig(save_name)
# plt.scatter(h_axis,v_axis)
# print(v_axis[0:5])

# %%

# fft
N = len(v_axis)
FT = fftshift(fft(v_axis))

tmin = df.head(n=1).iloc[:,2].to_numpy()[0]
tmax = df.tail(n=1).iloc[:,2].to_numpy()[0]

dt = df.head(n=2).iloc[:,2].to_numpy()[1] - df.head(n=1).iloc[:,2].to_numpy()[0]


Df = 1/(tmax-tmin)
freqs = np.arange(-1/(2*dt), +1/(2*dt), Df)
plt.xscale("log")
plt.yscale("log")
plt.plot(freqs, np.abs(FT[0:]))


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

# f_phi1	= 0
# f_X1 	= 0
# f_Y2	= 1

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
# elif f_phi1:
# 	s += "varphi.txt"
# 	df = pd.read_csv(s, header=None)
# 	title = "$\\varphi$"
# elif f_X1:
# 	s += "X.txt"
# 	df = pd.read_csv(s, header=None)
# 	title = "$X$"
# elif f_Y2:
# 	s += "Y.txt"
# 	df = pd.read_csv(s, header=None)
# 	title = "$Y$"

rs = np.arange(rmin, rmax, dr)
h_axis = rs
v_axis = df.iloc[:,0].to_numpy()[:len(rs)]
h_label = "$r$"

plt.title(title)
plt.xlabel(h_label)
plt.plot(h_axis,v_axis)




# %%
