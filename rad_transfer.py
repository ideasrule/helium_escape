import scipy.special
import numpy as np
import scipy.integrate
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
import scipy.ndimage.filters
from simulate_escape import get_solution

e = 4.8032e-10
m_e = 9.11e-28
c = 3e10
A = 1.0216e7
m_He = 4 * 1.67e-24
k_B = 1.38e-16
R_sun = 7e10
M_jup = 1.898e30
M_earth = 5.97e27
R_jup = 7.1e9
R_earth = 6.378e8
parsec = 3.086e18
AU = 1.496e13


def get_tau(wavenum, b, n3_interp, T0, max_r, v_interp, epsilon=0.01):
    line_wavenums, fs = np.loadtxt("He_line_data", unpack=True)
    assert(len(line_wavenums) == 3)
    #wavenums = 9230.868568
    #fs = 1.7974e-1
    
    sigma_0 = np.pi * e**2 * fs / m_e / c**2
    doppler_broadening = np.sqrt(k_B * T0 / m_He) * line_wavenums / c
    
    def integrand(theta, line_index):
        r = b / np.cos(theta)
        dtheta_to_dr = np.abs(b*np.sin(theta)/np.cos(theta)**2)
        gamma = A / 4 / np.pi / c
        lorentz = 1.0/np.pi * gamma / ((wavenum - line_wavenums[line_index])**2 + gamma**2)
        v_offset = v_interp(r) * np.sin(theta) 
        
        z = (wavenum - line_wavenums[line_index] - line_wavenums[line_index]/c*v_offset + gamma * 1j) / doppler_broadening[line_index] / np.sqrt(2)
        profile = np.real(scipy.special.wofz(z)) / doppler_broadening[line_index] / np.sqrt(2*np.pi)
        per_line = n3_interp(r) * sigma_0[line_index] * profile * r / np.sqrt(r**2 - b**2) * dtheta_to_dr
        return per_line

    max_theta = np.arccos(b / max_r)
    min_theta = -max_theta

    #Rough integration
    thetas = np.linspace(min_theta + 1e-2, max_theta - 1e-2, 250)
    integrands = np.sum([integrand(thetas, i) for i in range(len(line_wavenums))], axis=0)
    result = np.trapz(integrands, thetas)
    return result


def predict_depths(wavenums, spectrum_file, Mp, Rp, T0, mass_loss_rate, Rs, D_over_a, max_r_over_Rp=9.9):
    r_mesh, v_interp, n3_interp = get_solution(spectrum_file, Mp, Rp, T0, mass_loss_rate, D_over_a, max_r_over_Rp*1.01)
    radii = np.linspace(Rp, max_r_over_Rp * Rp, 100)
    dr = np.median(np.diff(radii))

    tot_extra_depth = 0
    transit_spectrum = []

    for wavenum in wavenums:
        tot_extra_depth = 0 #(Rp/Rs)**2 #0
        for r in radii:
            tau = get_tau(wavenum, r, n3_interp, T0, np.max(radii), v_interp)
            extra_depth = 2 / Rs**2 * r * dr * (1 - np.exp(-tau))
            tot_extra_depth += extra_depth        

        #print(wavenum, tot_extra_depth * 1e6)
        transit_spectrum.append(tot_extra_depth)

    return np.array(transit_spectrum)

if __name__ == "__main__":
    res = 375000
    wavenums = np.exp(np.arange(np.log(9230), np.log(9233), 1./res))

    #GJ 436b
    #transit_spectrum = predict_depths(wavenums, sys.argv[1], 0.073 * M_jup, 0.38 * R_jup, 5000, 2e10, 0.464 * R_sun, (9.76 * parsec / 0.03 / AU), 5.98)

    #HD 97658b
    #transit_spectrum = predict_depths(wavenums, sys.argv[1], 0.03 * M_jup, 0.21 * R_jup, 3000, 1e8, 0.74 * R_sun, (21.6 * parsec / 0.08 / AU), 9.9)

    #GJ 3470b
    #transit_spectrum = predict_depths(wavenums, sys.argv[1], 0.0437 * M_jup, 0.408 * R_jup, 7000, 4.6e10, 0.547 * R_sun, (29.45 * parsec / 0.0355 / AU), 9.9)

    #55 Cnc e
    #transit_spectrum = predict_depths(wavenums, sys.argv[1], 8.08 * M_earth, 1.91 * R_earth, 5000, 1.38e10, 0.943 * R_sun, (12.59 * parsec / 0.0155 / AU), 9)
    #print(np.max(transit_spectrum) * 1e6)

    #TOI 1726.01
    transit_spectrum = predict_depths(wavenums, sys.argv[1], 6.2 * M_earth, 2.3 * R_earth, 8000, 4e9, 0.9 * R_sun, (22.3 * parsec / 0.073 / AU), 11)

    #TOI 1726.02
    #transit_spectrum = predict_depths(wavenums, sys.argv[1], 9.1 * M_earth, 2.9 * R_earth, 5000, 2e9, 0.9 * R_sun, (22.3 * parsec / 0.15 / AU), 24)
    
    #np.save("transit_spectrum_new.npy", transit_spectrum)
    
    filtered_spectrum = scipy.ndimage.filters.gaussian_filter(transit_spectrum, res/37500/2.355)

    error = 440e-6
    obs_wavenums = np.arange(9230, 9233, 9233/115000)
    obs_depths = np.interp(obs_wavenums, wavenums, filtered_spectrum)
    obs_depths += np.random.normal(0, error, len(obs_depths))

    plt.plot(1e8/wavenums, 1e6*transit_spectrum, label="Unconvolved")
    #plt.plot(1e8/wavenums, filtered_spectrum * 1e6, label="Convolved")
    #plt.errorbar(1e8/obs_wavenums, obs_depths * 1e6, yerr=error*1e6, fmt='.')
    plt.xlabel("Wavelength (A)", fontsize=14)
    plt.ylabel("Transit depth (ppm)", fontsize=14)
    #plt.legend()
    plt.show()
