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
from configparser import ConfigParser
import dynesty.utils
from dynesty import NestedSampler
from constants import e, m_e, c, A, m_He, k_B, R_sun, R_earth, M_earth, parsec, AU, HOUR_TO_SEC, DAY_TO_SEC, CM_TO_ANGSTROM


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

        transit_spectrum.append(tot_extra_depth)

    end = time.time()
    #print("Rad transfer took", end - start)
    return np.array(transit_spectrum), taus, radii

def simulate_transit(params, wavs, times, tau_interp, v_flow, N=100):
    radius = N/2
    x_s = N/2
    y_s = N/2
    pixel_scale = params["Rs"] * R_sun / radius
    
    x_mesh, y_mesh = np.meshgrid(np.arange(N), np.arange(N))
    r = np.sqrt((x_mesh - x_s)**2 + (y_mesh - y_s)**2)
    
    mu = np.sqrt(1 - r**2 / radius**2)
    star = 1 - np.sum([params["limb_dark"][k-1] * (1 - mu**(k/2)) for k in range(1,5)], axis=0)
    star[r > radius] = 0
    profile_2D = []
    
    for t in times:
        x_p = x_s + 2*np.pi*(params["a"] * AU) * t / (params["P"] * DAY_TO_SEC * pixel_scale)
        y_p = y_s - params["b"] * params["Rs"] * R_sun / pixel_scale
        planet_dist = np.sqrt((x_mesh - x_p)**2 + (y_mesh - y_p)**2) * pixel_scale
        images = star * np.exp(-tau_interp(planet_dist))
        curr_profile = 1 - np.sum(images, axis=(1,2)) / np.sum(star)
        v = params["acceleration"] * t + v_flow
        curr_profile = np.interp((1 - v/c) * wavs, wavs, curr_profile)
        curr_profile = scipy.ndimage.filters.gaussian_filter(curr_profile, 110000/25000/2.355)
        profile_2D.append(curr_profile)

    return np.array(profile_2D)
    

def get_obs_data(spectrum_filename, error_filename, min_wav, max_wav):
    with open(spectrum_filename) as f:
        obs_times = np.array([float(t) for t in f.readline().strip().split(",")])
        obs_wavs = np.array([float(t) for t in f.readline().strip().split(",")])

    obs_excess = 0.01 * np.loadtxt(spectrum_filename, skiprows=2, delimiter=",")
    obs_excess_error = 0.01 * np.loadtxt(error_filename, skiprows=2, delimiter=",")
    cond = np.logical_and(obs_wavs > min_wav, obs_wavs < max_wav)
    obs_wavs = obs_wavs[cond]
    obs_excess = obs_excess[:,cond]
    obs_excess_error = obs_excess_error[:,cond]
    return obs_times, obs_wavs, obs_excess, obs_excess_error

def get_ln_like(params, config, obs_times, obs_wavs, obs_excess, obs_excess_error, plot=False):
    T, log_mass_loss_rate, v_flow = params
    mass_loss_rate = 10**log_mass_loss_rate
    obs_wavenums = CM_TO_ANGSTROM / obs_wavs
    transit_spectrum, taus, physical_radii = predict_depths(obs_wavenums, config["stellar_spectrum"], config["Mp"] * M_earth, config["Rp"] * R_earth, T, mass_loss_rate, config["Rs"] * R_sun, (config["dist"] * parsec / config["a"] / AU), 11)
    tau_interp = scipy.interpolate.interp1d(physical_radii, taus, bounds_error=False, fill_value=(0, 0))
    profile_2D = simulate_transit(config, obs_wavs, obs_times * HOUR_TO_SEC, tau_interp, v_flow)
    residuals = obs_excess - profile_2D
    ln_like = -0.5 * np.sum(residuals**2 / obs_excess_error**2 + np.log(2 * np.pi * obs_excess_error**2))

    print(np.sum(residuals**2/obs_excess_error**2)/(residuals.shape[0] * residuals.shape[1]), T, log_mass_loss_rate, v_flow/1e5)
    
    if plot:
        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(obs_excess, vmin=-0.01, vmax=0.02)
        plt.subplot(1,3,2)
        plt.imshow(profile_2D, vmin=-0.01, vmax=0.02)
        plt.subplot(1,3,3)
        plt.imshow(residuals, vmin=-0.01, vmax=0.02)

    return ln_like

def transform_prior(cube):
    new_cube = np.zeros(len(cube))
    
    #Shift
    mins =  [3e3, 9, -10e5]
    maxes = [1.5e4, 11, 10e5]

    for i in range(len(mins)):
        new_cube[i] = mins[i] + (maxes[i] - mins[i]) * cube[i]
    return new_cube

parser = ConfigParser()
parser.optionxform = str
parser.read(sys.argv[1])
config = dict(parser.items("DEFAULT"))
for key in config:
    try:
        config[key] = eval(config[key])
    except:
        pass
print(config)

obs_times, obs_wavs, obs_excess, obs_excess_error = get_obs_data(
    config["spectrum"], config["error"], config["min_wav"], config["max_wav"])

def multinest_ln_like(cube):
    return get_ln_like(cube, config, obs_times, obs_wavs, obs_excess, obs_excess_error)

sampler = NestedSampler(multinest_ln_like, transform_prior, 3, bound='multi',
                        nlive=100)
sampler.run_nested()
result = sampler.results
normalized_weights = np.exp(result.logwt - np.max(result.logwt))
normalized_weights /= np.sum(normalized_weights)
result.weights = normalized_weights
with open("dynesty_result.pkl", "wb") as f:
    pickle.dump(result, f)

best_params = result.samples[np.argmax(result.logl)]
get_ln_like(best_params, config, obs_times, obs_wavs, obs_excess, obs_excess_error, plot=True)
plt.savefig("best_fit.png")

plt.figure()
fig = corner.corner(result.samples, weights=result.weights,
                    range=[0.99] * result.samples.shape[1],
                    labels=["T", "M_dot", "v_flow"])
plt.savefig("corner.png")
plt.show()
