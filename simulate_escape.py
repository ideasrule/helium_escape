import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import sys
from scipy.integrate import solve_bvp, solve_ivp, odeint
import scipy.interpolate
from scipy.interpolate import interp1d

M_jup = 1.898e30
R_jup = 7.1e9
c = 3e10
h = 6.626e-27
CM_TO_ANGSTROM = 1e8
WAVELENGTH_0 = 912 / CM_TO_ANGSTROM
AMU = 1.67e-24
G = 6.67e-8
k_B = 1.38e-16
D_over_a = 69e6

def get_stellar_spectrum(filename):
    with astropy.io.fits.open(filename) as hdul:
        return hdul[1].data["WAVELENGTH"] / CM_TO_ANGSTROM, hdul[1].data["FLUX"] * CM_TO_ANGSTROM * D_over_a**2

def photoionization_cross_section(wavelength):
    ionized = wavelength < WAVELENGTH_0
    epsilon = np.sqrt(WAVELENGTH_0 / wavelength - 1)
    cross_sections = 6.3e-18 * np.exp(4 - 4*np.arctan(epsilon)/epsilon) / (1 - np.exp(-2*np.pi/epsilon)) * (wavelength / WAVELENGTH_0)**4
    cross_sections[~ionized] = 0
    return cross_sections

def avg_cross_section(wavelengths, fluxes):
    ionized = wavelengths < WAVELENGTH_0
    wavelengths = wavelengths[ionized]
    fluxes = fluxes[ionized]
    
    cross_sections = photoionization_cross_section(wavelengths)
    weighted_sum = np.trapz(fluxes * cross_sections, wavelengths)
    total_fluxes = np.trapz(fluxes, wavelengths)
    avg_cross_section = weighted_sum / total_fluxes
    return avg_cross_section

    print(avg_cross_section)
    plt.loglog(wavelengths, cross_sections)
    plt.figure()
    plt.loglog(wavelengths, fluxes)
    plt.show()

def photoionization_rate(wavelengths, fluxes):
    ionized = wavelengths < WAVELENGTH_0
    wavelengths = wavelengths[ionized]
    fluxes = fluxes[ionized]
    cross_sections = photoionization_cross_section(wavelengths)
    rate = np.trapz(fluxes * cross_sections / (h * c / wavelengths), wavelengths)
    print("Rate", rate)
    #plt.loglog(wavelengths, fluxes)
    #plt.show()
    return rate

def parker_solution(T0, mu, planet_mass, mass_loss_rate):
    sound_speed = np.sqrt(k_B * T0 / mu / AMU)
    sonic_r = G * planet_mass / 2 / sound_speed**2
    rho_crit = mass_loss_rate / (4 * np.pi * sonic_r**2 * sound_speed)
    print(sound_speed, sonic_r, rho_crit)
    
    Rs = np.load("Rs.npy")
    Vs = np.load("Vs.npy")
    Rhos = np.exp(2/Rs - 3./2 - Vs**2/2)

    v_s = Vs * sound_speed
    r_s = sonic_r * Rs
    rho_s = Rhos * rho_crit

    return v_s, r_s, rho_s, T0


def iter_fion_sol(all_r, all_rho, all_v, a0, photo_rate, T0, Rp):
    rho_interp = scipy.interpolate.interp1d(all_r, all_rho)
    v_interp = scipy.interpolate.interp1d(all_r, all_v)

    def tau_derivs(r, variables, fion_interp):
        alpha_rec = 2.59e-13 * (T0/1e4)**-0.7
        neutral_fraction = 1 - fion_interp(r)
        tau_deriv = -neutral_fraction * rho_interp(r) * 0.9 * a0 / 1.3 / AMU
        assert(tau_deriv < 0)
        return [tau_deriv]

    def fion_derivs(r, variables, tau_interp):
        f_ion = variables[0]
        tau_0 = tau_interp(r)
        alpha_rec = 2.59e-13 * (T0/1e4)**-0.7
        neutral_fraction = 1 - f_ion
        fion_deriv = neutral_fraction / v_interp(r) * photo_rate * np.exp(-tau_0) - 0.9 * rho_interp(r) * f_ion**2 * alpha_rec / (1.3 * AMU * v_interp(r))
        print(r, neutral_fraction / v_interp(r) * photo_rate * np.exp(-tau_0), 0.9 * rho_interp(r) * f_ion**2 * alpha_rec / (1.3 * AMU * v_interp(r)))
        return [fion_deriv]

    r_mesh = np.linspace(Rp, 10*Rp, 100)
    fion_interp = interp1d(r_mesh, np.zeros(len(r_mesh)))
    taus_interp = None #interp1d(r_mesh, np.zeros(len(r_mesh)))
    
    while True:
        result = solve_ivp(lambda r, y: tau_derivs(r, y, fion_interp), (10*Rp, Rp), [0], t_eval=r_mesh[::-1])
        taus_interp = interp1d(r_mesh, result.y[0][::-1])
        result = solve_ivp(lambda r, y: fion_derivs(r, y, taus_interp), (Rp, 10*Rp), [0], t_eval=r_mesh)
        fion_interp = interp1d(r_mesh, result.y[0])
        
        print(fion_interp(2*Rp), taus_interp(2*Rp))
        plt.plot(r_mesh, fion_interp(r_mesh))
        plt.plot(r_mesh, taus_interp(r_mesh))
        plt.show()
        

def fion_sol(all_r, all_rho, all_v, a0, photo_rate, T0, Rp):
    rho_interp = scipy.interpolate.interp1d(all_r, all_rho)
    v_interp = scipy.interpolate.interp1d(all_r, all_v)

    def bc(ya, yb):
        initial_f_ion = ya[1]
        final_tau = yb[0]
        return np.array([initial_f_ion, final_tau])
    
    def get_derivs(r, variables):
        tau_0, f_ion = variables
        alpha_rec = 2.59e-13 * (T0/1e4)**-0.7
        neutral_fraction = 1 - f_ion
        
        tau_deriv = -neutral_fraction * rho_interp(r) * 0.9 * a0 / 1.3 / AMU   
        fion_deriv = neutral_fraction / v_interp(r) * photo_rate * np.exp(-tau_0) - 0.9 * rho_interp(r) * f_ion**2 * alpha_rec / (1.3 * AMU * v_interp(r))
        #if f_ion >= 1:
        #    fion_deriv = 0
        
        print(tau_0[0], f_ion[0], tau_deriv[0], fion_deriv[0])
        return np.vstack((tau_deriv, fion_deriv))

    r_mesh = np.linspace(Rp, 10*Rp, 5)
    guess = np.zeros((2, len(r_mesh)))
    #guess[0] = 0
    #guess[1] = np.linspace(0, 0.2, 5)
    result = solve_bvp(get_derivs, bc, r_mesh, guess)

    x_plot = np.linspace(Rp, 10*Rp, 100)
    print(result.sol(x_plot).shape)
    y_plot = result.sol(x_plot)
    print(y_plot[1])

    plt.plot(x_plot / Rp, y_plot[0], label="tau")
    plt.plot(x_plot / Rp, y_plot[1], label="fion")
    plt.legend()
    plt.show()
        

all_v, all_r, all_rho, T0 = parker_solution(5000, 1.3, 0.073 * M_jup, 2e10)
wavelengths, fluxes = get_stellar_spectrum(sys.argv[1])
a0 = avg_cross_section(wavelengths, fluxes)
photo_rate = photoionization_rate(wavelengths, fluxes)
iter_fion_sol(all_r, all_rho, all_v, a0, photo_rate, T0, 0.38 * R_jup)

wavelengths = np.linspace(1e-7, 90e-7, 1000)
frequencies = c / wavelengths
print(frequencies)
plt.loglog(wavelengths * 1e7, photoionization_cross_section(frequencies))
plt.show()
