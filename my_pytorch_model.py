import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
from typing import TypeVar, Iterable, Tuple
import torch
import copy

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
efficiency = 0.3 #0.15
C_cool = 7.5e-19
Tc = 118348
F_UV = 118 / 0.0719**2
H_to_He = 9
gamma = 5./3
photo_rate = 1.1e-3

length_scale = Rp
vel_scale = 1e6
rho_scale = 1e-12
mass_scale = rho_scale * length_scale**3
time_scale = length_scale / vel_scale
T_scale = 1e4
mass_loss_scale = 1e11

def get_derivs(r, vals):
    derivs = torch.zeros(len(vals))
    derivs[1:-1] = (vals[2:] - vals[:-2]) / (r[2:] - r[:-2])
    #derivs[1:] = (vals[1:] - vals[:-1]) / (r[1:] - r[:-1])
    return derivs

def padded_diff(vals):
    result = np.diff(vals)
    result = np.append(0, result)
    return result

def differential_equations(scaled_M_dot, scaled_r, scaled_rho, scaled_fion, scaled_T, margin=5, debug=False, eps=1e-4):
    '''coeffs = np.polyfit(1/scaled_r.detach(), np.log(scaled_rho.detach()), 10)
    predicted = np.polyval(coeffs, 1/scaled_r)
    plt.plot(scaled_r.detach(), np.log(scaled_rho.detach()))
    plt.plot(scaled_r.detach(), predicted)
    plt.show()'''
    M_dot = scaled_M_dot * mass_loss_scale
    r = scaled_r * length_scale
    rho = torch.exp(torch.cumsum(-torch.abs(scaled_rho), 0) + 4.809) * rho_scale / scaled_r**2
    v = torch.abs(M_dot / (4*np.pi*r**2*rho)) + eps
    fion = torch.abs(torch.cumsum(scaled_fion, 0))
    T = torch.abs(torch.cumsum(scaled_T * T_scale, 0) + 1173)

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

    #recomb coeff
    
    #tau = torch.append(tau, 0)
    #tau = -cumtrapz((n_HI * sigma_ion)[::-1], r[::-1])[::-1]
    #tau = torch.append(tau, 0)

    #plt.loglog(r / Rp, tau)
    #plt.show()
    
    momentum_diffs = r**2 * (v*v_derivs + rho**-1 * get_derivs(r, (n_H + n_He + n_e) * k_B * T) + G*M/r**2) / (length_scale**3 / time_scale**2) / 1e4
    heating = efficiency * F_UV * torch.exp(-tau) * sigma_ion * n_HI
    cooling = C_cool * n_HI * n_e * torch.exp(-Tc / T)
    energy_diffs = r**2 * (rho * v * get_derivs(r, k_B * T / (gamma - 1) / mu) - k_B * T * v / mu * get_derivs(r, rho) - heating + cooling) / (mass_scale * length_scale / time_scale**3)

    alpha_rec = 2.59e-13 * (T/1e4)**-0.7
    fion_diffs = ((1 - fion) * photo_rate * torch.exp(-tau) / v - fion * n_HII * alpha_rec / v - get_derivs(r, fion)) * length_scale
    loss = (momentum_diffs**2 + energy_diffs**2 + fion_diffs**2)[margin:-margin].sum()

    
    if debug:
        print(
              (momentum_diffs**2)[margin:-margin].sum().detach().numpy(),
              (energy_diffs**2)[margin:-margin].sum().detach().numpy(),
              (fion_diffs**2)[margin:-margin].sum().detach().numpy())
        plt.plot(r/Rp, v.detach())
        plt.title("v")
        plt.figure()
        plt.semilogy(r/Rp, rho.detach())
        plt.title("rho")
        plt.figure()
        plt.plot(r/Rp, fion.detach())
        plt.title("fion")
        plt.figure()
        plt.plot(r/Rp, T.detach())
        plt.title("T")
        plt.show()
        
        import pdb
        pdb.set_trace()
    
    #Enforce transsonic
    if v[-margin] < 1.5e6:
        loss += (v[-margin] - 1.5e6)**2 / 1e4**2
 
    if torch.isinf(loss):
        loss *= 0
        loss += 1e9
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

#plt.loglog(radii/Rp, wind_vel)
#plt.show()

dtype = torch.float64
device = torch.device("cpu")
M_dot = torch.tensor((8e11 / mass_loss_scale), device=device, dtype=dtype, requires_grad=True)
tensor_radii = torch.tensor(radii / length_scale, device=device, dtype=dtype)
#wind_vel = torch.tensor(padded_diff(wind_vel / vel_scale), device=device, dtype=dtype, requires_grad=True)
#wind_vel = torch.tensor(np.zeros(len(radii)), device=device, dtype=dtype, requires_grad=True)
vel_guess = np.linspace(0, 2e6, len(radii))
wind_vel = torch.tensor(padded_diff(vel_guess / vel_scale), device=device, dtype=dtype, requires_grad=True)

rho = torch.tensor(padded_diff(np.log(rho / rho_scale * (radii / length_scale)**2)), device=device, dtype=dtype, requires_grad=True)
fion = torch.tensor(padded_diff(np.copy(HII)), device=device, dtype=dtype, requires_grad=True)
#T = torch.tensor(padded_diff(Te.copy()) / T_scale, device=device, dtype=dtype, requires_grad=True)
T = torch.tensor(np.zeros(len(radii)), device=device, dtype=dtype, requires_grad=True)
minimum_loss = np.inf
parameters = [M_dot, rho, fion, T]
optimizer = torch.optim.Adam(parameters, lr=1e-6, eps=1e-8)

for t in range(int(1e6)):
    optimizer.zero_grad()    
    loss = differential_equations(M_dot, tensor_radii, rho, fion, T)
    if loss.item() < minimum_loss and not np.isnan(loss.item()):
        minimum_loss = loss.item()
        best_yet = copy.deepcopy(parameters) #[M_dot.clone(), torch.clone(wind_vel), torch.clone(rho), torch.clone(fion), torch.clone(T)]

    #if loss.item() < 1: break
        
    #print(loss.item())
    if t % 100 == 0:\
        print(t, loss.item())
    loss.backward()
    # if t % 1000 == 0:
    #     print(torch.min(wind_vel.grad).detach().numpy(),
    #           torch.max(wind_vel.grad).detach().numpy(),
    #           torch.min(rho.grad).detach().numpy(),
    #           torch.max(rho.grad).detach().numpy(),
    #           torch.min(fion.grad).detach().numpy(),
    #           torch.max(fion.grad).detach().numpy(),
    #           torch.min(T).detach().numpy(),
    #           torch.max(T).detach().numpy())
    #torch.nn.utils.clip_grad_norm_(parameters, max_norm=1e4)
    optimizer.step()
    
print(minimum_loss)
differential_equations(best_yet[0], tensor_radii, best_yet[1], best_yet[2], best_yet[3], debug=True)

plt.plot(radii / Rp, np.abs(1173 + T_scale * np.cumsum(best_yet[3].detach().numpy())))
plt.plot(radii / Rp, Te)
plt.figure()
#plt.plot(radii / Rp, np.cumsum(best_yet[1].detach().numpy()))
#plt.figure()
plt.plot(radii / Rp, np.abs(np.cumsum(best_yet[2].detach())))
plt.show()
