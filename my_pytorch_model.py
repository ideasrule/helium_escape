import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate

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
    derivs = np.zeros(len(vals))
    derivs[1:-1] = (vals[2:] - vals[:-2]) / (r[2:] - r[:-2])
    return derivs
        
def differential_equations(r, v, rho, fion, tau, T, Rp, heating_tpci, cooling_tpci, n_H_tpci, n_e_tpci):
    length_scale = 1e9
    vel_scale = 1e6
    rho_scale = 1e-10
    mass_scale = rho_scale * length_scale**3
    time_scale = length_scale / vel_scale    
    
    v_derivs = get_derivs(r, v)
    rho_derivs = get_derivs(r, rho)
    #fion_derivs = get_derivs(r, rho)

    n_H = k * rho / m_H
    n_He = n_H / H_to_He
    n_HII = n_H * fion
    n_HI = n_H - n_HII
    n_e = n_HII
    mu = rho / (n_H + n_He + n_e)
    
    continuity_diffs = get_derivs(r, r**2 * rho * v) / (mass_scale / length_scale / time_scale)
    momentum_diffs = r**2 * (v*v_derivs + rho**-1 * get_derivs(r, (n_H + n_He + n_e) * k_B * T) + G*M/r**2) / (length_scale**3 / time_scale**2)    
    #momentum_diffs = v*v_derivs + rho**-1 * get_derivs(r, (1.1*n_H_tpci + n_e_tpci) * k_B * T) + G*M/r**2
    
    heating = efficiency * F_UV * np.exp(-tau) * sigma_ion * n_HI
    cooling = C_cool * n_HI * n_e * np.exp(-Tc / T)
    energy_diffs = r**2 * (rho * v * get_derivs(r, k_B * T / (gamma - 1) / mu) - k_B * T * v / mu * get_derivs(r, rho) - heating_tpci + cooling_tpci) / (mass_scale * length_scale / time_scale**3)
    loss = continuity_diffs**2 + momentum_diffs**2 + energy_diffs**2

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

#Set up pytorch

differential_equations(radii, wind_vel, rho, HII, tau, Te, Rp, heating, cooling, n_H, n_e)
