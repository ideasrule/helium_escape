import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.special
import numpy as np
import scipy.integrate
import scipy.interpolate
import sys
import scipy.ndimage.filters
from simulate_escape import get_solution
import time
import corner
import pickle
import dynesty.utils
from dynesty import NestedSampler


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
HOUR_TO_SEC = 3600
DAY_TO_SEC = 86400
line_wavenums = np.array([9231.856483, 9230.868568, 9230.792143])
fs = np.array([5.9902e-2, 1.7974e-1, 2.9958e-1])

def get_tau(wavenum, b, n3_interp, T0, max_r, v_interp, epsilon=0.01):
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
    thetas = np.linspace(min_theta + 1e-2, max_theta - 1e-2, 100)
    integrands = np.sum([integrand(thetas, i) for i in range(len(line_wavenums))], axis=0)
    result = np.trapz(integrands, thetas)
    return result


def predict_depths(wavenums, spectrum_file, Mp, Rp, T0, mass_loss_rate, Rs, D_over_a, max_r_over_Rp=9.9):
    r_mesh, v_interp, n3_interp = get_solution(spectrum_file, Mp, Rp, T0, mass_loss_rate, D_over_a, max_r_over_Rp*1.01)
    radii = np.linspace(Rp, max_r_over_Rp * Rp, 100)
    dr = np.median(np.diff(radii))

    tot_extra_depth = 0
    transit_spectrum = []

    start = time.time()
    taus = np.zeros((len(wavenums), len(radii)))
    for w, wavenum in enumerate(wavenums):
        tot_extra_depth = 0
        for r_index, r in enumerate(radii):
            tau = get_tau(wavenum, r, n3_interp, T0, np.max(radii), v_interp)
            extra_depth = 2 / Rs**2 * r * dr * (1 - np.exp(-tau))
            tot_extra_depth += extra_depth
            taus[w,r_index] = tau

        #print(wavenum, tot_extra_depth * 1e6)
        transit_spectrum.append(tot_extra_depth)

    end = time.time()
    #print("Rad transfer took", end - start)
    return np.array(transit_spectrum), taus, radii

def simulate_transit(params, wavs, times, tau_interp, v_flow, N=100):
    radius = N/2
    x_s = N/2
    y_s = N/2
    pixel_scale = params["Rs"] / radius
    
    x_mesh, y_mesh = np.meshgrid(np.arange(N), np.arange(N))
    r = np.sqrt((x_mesh - x_s)**2 + (y_mesh - y_s)**2)
    
    mu = np.sqrt(1 - r**2 / radius**2)
    star = 1 - np.sum([params["limb_dark"][k-1] * (1 - mu**k) for k in range(1,5)], axis=0)
    star[r > radius] = 0
    profile_2D = []
    
    for t in times:
        x_p = x_s + 2*np.pi*(params["aRs"] * params["Rs"]) * t / (params["P"] * DAY_TO_SEC * pixel_scale)
        y_p = y_s - params["b"] * params["Rs"] / pixel_scale
        planet_dist = np.sqrt((x_mesh - x_p)**2 + (y_mesh - y_p)**2) * pixel_scale
        images = star * np.exp(-tau_interp(planet_dist))
        curr_profile = 1 - np.sum(images, axis=(1,2)) / np.sum(star)
        v = params["acceleration"] * t + v_flow
        curr_profile = np.interp((1 - v/c) * wavs, wavs, curr_profile)
        curr_profile = scipy.ndimage.filters.gaussian_filter(curr_profile, 110000/25000/2.355)
        profile_2D.append(curr_profile)

    #profile_2D = np.array(profile_2D)
    #profile_2D = 1 - profile_2D
    return np.array(profile_2D)
    

def transform_prior(cube):
    new_cube = np.zeros(len(cube))
    
    #Shift
    mins =  [3e3, 9, -10e5]
    maxes = [1e4, 11, 10e5]

    for i in range(len(mins)):
        new_cube[i] = mins[i] + (maxes[i] - mins[i]) * cube[i]
    return new_cube

star_spectrum_filename = "/home/stanley/backup_external/toi560/final_stellar_spectrum.txt"
spectrum_filename = "toi560_excess_night1"
error_filename = "toi560_excess_night1_errors"
with open(spectrum_filename) as f:
    obs_times = np.array([float(t) for t in f.readline().strip().split(",")])
    obs_wavs = np.array([float(t) for t in f.readline().strip().split(",")])
    obs_wavenums = 1e8 / obs_wavs

obs_excess = 0.01 * np.loadtxt(spectrum_filename, skiprows=2, delimiter=",")
obs_excess_error = 0.01 * np.loadtxt(error_filename, skiprows=5, delimiter=",")
cond = np.logical_and(obs_wavs > 10831, obs_wavs < 10835)
obs_wavs = obs_wavs[cond]
obs_excess = obs_excess[:,cond]
obs_excess_error = obs_excess_error[:,cond]
obs_wavenums = 1e8 / obs_wavs

transit_params = {"Rs": 0.665 * R_sun,
          "aRs": 19.98,
          "b": 0.566,
          "P": 6.3980420,
          "acceleration": 117,
          "limb_dark": [0.0949594, 0.34250499, 0.51543851, -0.28607617]
         }

def get_ln_like(params, plot=False):
    T, log_mass_loss_rate, v_flow = params
    mass_loss_rate = 10**log_mass_loss_rate
    transit_spectrum, taus, physical_radii = predict_depths(obs_wavenums, star_spectrum_filename, 11 * M_earth, 2.8 * R_earth, T, mass_loss_rate, 0.665 * R_sun, (31.6 * parsec / 0.0596 / AU), 11)
    tau_interp = scipy.interpolate.interp1d(physical_radii, taus, bounds_error=False, fill_value=(0, 0))
    profile_2D = simulate_transit(transit_params, obs_wavs, obs_times * HOUR_TO_SEC, tau_interp, v_flow)
    residuals = obs_excess - profile_2D
    ln_like = -0.5 * np.sum(residuals**2 / obs_excess_error**2 + np.log(2 * np.pi * obs_excess_error**2))

    #plt.imshow(obs_excess)
    #plt.figure()
    #plt.imshow(profile_2D)
    #plt.show()
    print(np.sum(residuals**2/obs_excess_error**2)/(residuals.shape[0] * residuals.shape[1]), T, log_mass_loss_rate, v_flow/1e5)
    
    if plot:
        plt.figure()
        plt.imshow(residuals)

    return ln_like

sampler = NestedSampler(get_ln_like, transform_prior, 3, bound='multi',
                        nlive=100)
sampler.run_nested()
result = sampler.results
normalized_weights = np.exp(result.logwt - np.max(result.logwt))
normalized_weights /= np.sum(normalized_weights)
result.weights = normalized_weights
with open("dynesty_result.pkl", "wb") as f:
    pickle.dump(result, f)


best_params = result.samples[np.argmax(result.logl)]
get_ln_like(best_params, plot=True)
plt.savefig("best_fit.png")

plt.figure()
fig = corner.corner(result.samples, weights=result.weights,
                    range=[0.99] * result.samples.shape[1],
                    labels=["T", "M_dot", "v_flow"])
plt.savefig("corner.png")
plt.show()
