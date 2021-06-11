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

length_scale = Rp
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

def padded_diff(vals):
    result = np.diff(vals)
    result = np.append(0, result)
    return result

def differential_equations(scaled_r, scaled_v, scaled_rho, scaled_fion, scaled_T, margin=5):
    '''coeffs = np.polyfit(1/scaled_r.detach(), np.log(scaled_rho.detach()), 10)
    predicted = np.polyval(coeffs, 1/scaled_r)
    plt.plot(scaled_r.detach(), np.log(scaled_rho.detach()))
    plt.plot(scaled_r.detach(), predicted)
    plt.show()'''
    
    r = scaled_r * length_scale
    v = torch.cumsum(scaled_v * vel_scale, 0) + 400
    rho = torch.exp(torch.cumsum(scaled_rho, 0) + 4.809) * rho_scale / scaled_r**2
    fion = torch.cumsum(scaled_fion, 0)
    T = torch.cumsum(scaled_T * T_scale, 0) + 1173

    '''plt.loglog(r.detach() / Rp, v.detach())
    plt.loglog(r.detach() / Rp, rho.detach())
    plt.loglog(r.detach() / Rp, fion.detach())
    plt.loglog(r.detach() / Rp, T.detach())
    plt.show()'''
    
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
    #loss += ((v[0:5] * 1e5)**2 + (T[0:5] - 1000)**2 / 0.01**2 + (fion[0:5] * 1e3)**2).sum()
    
    #if torch.isnan(loss):
    #    import pdb
    #    pdb.set_trace()

    '''plt.figure()
    plt.plot(r.detach() / Rp, continuity_diffs.detach())
    plt.plot(r.detach() / Rp, momentum_diffs.detach())
    plt.plot(r.detach() / Rp, energy_diffs.detach())
    plt.show()'''
    if torch.isinf(loss):
        loss *= 0
        loss += 1e9
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
wind_vel = torch.tensor(padded_diff(wind_vel / vel_scale), device=device, dtype=dtype, requires_grad=True)
rho = torch.tensor(padded_diff(np.log(rho / rho_scale * (radii / length_scale)**2)), device=device, dtype=dtype, requires_grad=True)
fion = torch.tensor(padded_diff(np.copy(HII)), device=device, dtype=dtype, requires_grad=True)
#T = torch.tensor(padded_diff(Te.copy()) / T_scale, device=device, dtype=dtype, requires_grad=True)
T = torch.tensor(np.zeros(len(radii)), device=device, dtype=dtype, requires_grad=True)
minimum_loss = np.inf
parameters = [wind_vel, rho, fion, T]
optimizer = torch.optim.Adam(parameters, lr=1e-6, eps=1e-4)

for t in range(int(1e4)):
    optimizer.zero_grad()    
    loss = differential_equations(tensor_radii, wind_vel, rho, fion, T)
    if loss.item() < minimum_loss and not np.isnan(loss.item()):
        minimum_loss = loss.item()
        best_yet = [torch.clone(wind_vel), torch.clone(rho), torch.clone(fion), torch.clone(T)]

    #if loss.item() < 1: break
        
    #print(loss.item())
    if t % 1 == 0:
        print(t, loss.item())
    loss.backward()
    if t % 1000 == 0:
        print(torch.min(wind_vel.grad).detach().numpy(),
              torch.max(wind_vel.grad).detach().numpy(),
              torch.min(rho.grad).detach().numpy(),
              torch.max(rho.grad).detach().numpy(),
              torch.min(fion.grad).detach().numpy(),
              torch.max(fion.grad).detach().numpy(),
              torch.min(T).detach().numpy(),
              torch.max(T).detach().numpy())
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=1e4)
    optimizer.step()
    
print(minimum_loss)
plt.plot(radii, 1100 + T_scale * np.cumsum(best_yet[3].detach().numpy()))
plt.show()
