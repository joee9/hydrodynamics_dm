#%%
from os.path import exists

eos = 'SLy'
lmbda = 0
dr = 0.02

p0 = 10**3.1
varphi0 = 10**-1.7

p0 = p0 * 7.6804e-6

# p0 = 0.00305762
# varphi0 = 0.03162278

if exists(f'./input/{eos}_p{p0:.8f}_vc{varphi0:.8f}_lam{lmbda:.3f}_dr{dr:.3f}_rho.txt'):
    print(f'Exists! {p0=}, {varphi0=}.')
else: 
    print(f'Does not exist! {p0=}, {varphi0=}.')
