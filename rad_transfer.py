import scipy.special
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt

e = 4.8032e-10
m_e = 9.11e-28
c = 3e10
A = 1.0216e7
m_He = 4 * 1.67e-24
k_B = 1.38e-16
R_sun = 7e10
Rs = 0.464 * R_sun

def get_tau(wavenum, b, n3_interp, T0, max_r, v_interp):
    wavenums, fs = np.loadtxt("He_line_data", unpack=True)
    sigma_0 = np.pi * e**2 * fs / m_e / c**2
    doppler_broadening = np.sqrt(k_B * T0 / m_He) * wavenums / c
    
    def integrand(theta):
        r = b / np.cos(theta)
        dtheta_to_dr = np.abs(b*np.sin(theta)/np.cos(theta)**2)
        #dr_to_dtheta = np.abs(np.cos(theta)**2 / b / np.sin(theta))
        gamma = A / 4 / np.pi / c
        lorentz = 1.0/np.pi * gamma / ((wavenum - wavenums)**2 + gamma**2)
        v_offset = v_interp(r) * np.sin(theta) 
        
        z = (wavenum - wavenums - wavenums/c*v_offset + gamma * 1j) / doppler_broadening / np.sqrt(2)
        profile = np.real(scipy.special.wofz(z)) / doppler_broadening / np.sqrt(2*np.pi)
        per_line = n3_interp(r) * sigma_0 * profile * r / np.sqrt(r**2 - b**2) * dtheta_to_dr
        #print(r, wavenum - wavenums[1], doppler_broadening[1], per_line[1])
        #print(r, lorentz[1], profile[1])
        return np.sum(per_line)

    max_theta = np.arccos(b / max_r)
    min_theta = -max_theta
    result, error = scipy.integrate.quad(integrand, min_theta, max_theta, limit=100, points=[0])
    #print(result, error)
    return result

r_mesh = np.load("r_mesh.npy")
n_triplet = np.load("n_triplet.npy")
all_v = np.load("all_v.npy")
n3_interp = scipy.interpolate.interp1d(r_mesh, n_triplet)
v_interp = scipy.interpolate.interp1d(r_mesh, all_v)
Rp = np.min(r_mesh)

radii = np.linspace(Rp, 6 * Rp, 100)
dr = np.median(np.diff(radii))
wavenums = np.linspace(9230, 9233, 100)

#taus = []
tot_extra_depth = 0
transit_spectrum = []

for wavenum in wavenums:
    tot_extra_depth = 0
    for r in radii:
        tau = get_tau(wavenum, r, n3_interp, 5000, np.max(r_mesh), v_interp)
        #taus.append(tau)
        extra_depth = 2 / Rs**2 * r * dr * (1 - np.exp(-tau))
        tot_extra_depth += extra_depth        
        #print(r, tau, extra_depth*1e6)
    print(wavenum, tot_extra_depth * 1e6)
    transit_spectrum.append(tot_extra_depth)
    
#print(tot_extra_depth * 1e6)
np.save("transit_spectrum.npy", transit_spectrum)
plt.plot(wavenums, transit_spectrum)
plt.show()
    
