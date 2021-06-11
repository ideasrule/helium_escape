import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from typing import TypeVar, Iterable, Tuple
import torch

k_B = 1.38e-16
m_H = 1.67e-24
G = 6.67e-8
R_earth = 6.378e8
M = 5.5 * 5.97e27
Rp = 2.15 * R_earth
k = 0.9 / 1.3
eV = 1.6e-12
E_ion = 13.6 * eV
sigma_ion = 1.6e-18
efficiency = 1 #0.15
C_cool = 7.5e-19
Tc = 118348
F_UV = 118 / 0.0719**2
H_to_He = 9
gamma = 5./3

length_scale = 1e9
vel_scale = 1e6
rho_scale = 1e-12
mass_scale = rho_scale * length_scale**3
time_scale = length_scale / vel_scale
T_scale = 1e4


def plot_iters(depths, quantity, starts, title=""):
    plt.figure()
    for i in range(len(starts) - 1):
        #if i != len(starts) - 2: continue
        if i % 10 != 0: continue
        first = starts[i]
        end = starts[i+1]
        plt.semilogy(radii[first:end], 10**quantity[first:end])
    plt.title(title)

def get_derivs(r, vals):
    derivs = torch.zeros(len(vals))
    derivs[1:-1] = (vals[2:] - vals[:-2]) / (r[2:] - r[:-2])
    return derivs
        
def differential_equations(scaled_r, scaled_v, scaled_rho, fion, scaled_T, margin=5):
    r = scaled_r * length_scale
    v = scaled_v * vel_scale
    rho = scaled_rho * rho_scale / scaled_r**2
    T = scaled_T * T_scale
    
    v_derivs = get_derivs(r, v)
    rho_derivs = get_derivs(r, rho)
    #fion_derivs = get_derivs(r, rho)

    n_H = k * rho / m_H
    n_He = n_H / H_to_He
    n_HII = n_H * fion
    n_HI = n_H - n_HII
    n_e = n_HII
    mu = rho / (n_H + n_He + n_e)

    dr = r[1:] - r[0:-1]
    tau = torch.zeros(len(r))
    tau[0:-1] = torch.flip(torch.cumsum(torch.flip(n_HI[:-1] * sigma_ion * dr, [0]), dim=0), dims=[0])
    #tau = torch.append(tau, 0)
    #tau = -cumtrapz((n_HI * sigma_ion)[::-1], r[::-1])[::-1]
    #tau = torch.append(tau, 0)

    #plt.loglog(r / Rp, tau)
    #plt.show()
    
    continuity_diffs = get_derivs(r, r**2 * rho * v) / (mass_scale / length_scale / time_scale)
    momentum_diffs = r**2 * (v*v_derivs + rho**-1 * get_derivs(r, (n_H + n_He + n_e) * k_B * T) + G*M/r**2) / (length_scale**3 / time_scale**2) / 1e4
    heating = efficiency * F_UV * torch.exp(-tau) * sigma_ion * n_HI
    cooling = C_cool * n_HI * n_e * torch.exp(-Tc / T)
    energy_diffs = r**2 * (rho * v * get_derivs(r, k_B * T / (gamma - 1) / mu) - k_B * T * v / mu * get_derivs(r, rho) - heating + cooling) / (mass_scale * length_scale / time_scale**3)
    loss = (continuity_diffs**2 + momentum_diffs**2 + energy_diffs**2)[margin:-margin].sum()

    #Enforce inner boundary condition
    loss += ((v[0:5] * 1e5)**2 + (T[0:5] - 1000)**2 / 0.01**2 + (fion[0:5] * 1e3)**2).sum()
    
    #if torch.isnan(loss):
    #    import pdb
    #    pdb.set_trace()
    
    return loss
    
    #plt.loglog(r / Rp, r**2 * rho * v / (mass_scale / length_scale / time_scale))
    plt.loglog(r / Rp, r**2 * v*v_derivs)
    plt.loglog(r / Rp,  -r**2 * rho**-1 * get_derivs(r, (1.1*n_H_tpci + n_e_tpci) * k_B * T))
    plt.loglog(r / Rp, G*M / r**2 * r**2)
    plt.show()
    
    plt.loglog(r / Rp, n_e)
    plt.loglog(r / Rp, n_e_tpci)
    plt.show()
    
    plt.plot(r / Rp, continuity_diffs)
    plt.figure()
    plt.plot(r / Rp, momentum_diffs)
    plt.figure()
    #plt.plot(r / Rp, rho * v * get_derivs(r, k_B * T / (gamma - 1) / mu) - k_B * T * v / mu * get_derivs(r, rho))
    #plt.plot(r / Rp, heating_tpci - cooling_tpci)
    plt.plot(r / Rp, energy_diffs)
    plt.show()
    return loss
    

number = sys.argv[1]
N = 790
data = np.loadtxt("cl_data.{}.over.tab".format(number))[0:N][::-1]
print(data.shape)
data[:,1:] = 10**data[:,1:]
depths, Te, heating, n_H, n_e, H_molec_ratio, HI, HII, HeI, HeII, HeIII = data.T[0:11]
radii = 14.9 * Rp - depths
rho = m_H * n_H * 1.3
wind_vel = -np.loadtxt("cl_data.{}.wind.tab".format(number), unpack=True, usecols=(2))[0:N][::-1]
tau = -scipy.integrate.cumulative_trapezoid((n_H * HI * sigma_ion)[::-1], radii[::-1])[::-1]
tau = np.append(tau, 0)
data = np.loadtxt("cl_data.{}.cool.tab".format(number), usecols=(0,2,3))[::-1]
heating = np.interp(radii, 14.9 * Rp - data[:,0], data[:,1])
cooling = np.interp(radii, 14.9 * Rp - data[:,0], data[:,2])

dtype = torch.float64
device = torch.device("cpu")
tensor_radii = torch.tensor(radii / length_scale, device=device, dtype=dtype)
wind_vel = torch.tensor(wind_vel / vel_scale, device=device, dtype=dtype, requires_grad=True)
rho = torch.tensor(rho / rho_scale * (radii / length_scale)**2, device=device, dtype=dtype, requires_grad=True)
fion = torch.tensor(np.copy(HII), device=device, dtype=dtype, requires_grad=True)
#T = torch.tensor(Te.copy() / T_scale, device=device, dtype=dtype, requires_grad=True)
T = torch.tensor(np.ones(len(radii)), device=device, dtype=dtype, requires_grad=True)
minimum_loss = np.inf
optimizer = torch.optim.Adam([wind_vel, rho, fion, T], lr=1e-4, eps=1e-8)

for t in range(int(1e5)):
    optimizer.zero_grad()    
    loss = differential_equations(tensor_radii, wind_vel, rho, fion, T)
    if loss.item() < minimum_loss and not np.isnan(loss.item()):
        minimum_loss = loss.item()
        best_yet = [wind_vel, rho, fion, T]

    if loss.item() < 1: break
        
    #print(loss.item())
    if t % 100 == 0:
        print(t, loss.item())
    loss.backward()
    optimizer.step()
    

plt.plot(radii, best_yet[3].detach().numpy())
plt.show()
