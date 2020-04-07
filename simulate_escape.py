import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits
import sys
from scipy.integrate import solve_bvp, solve_ivp, odeint
import scipy.interpolate
from scipy.interpolate import interp1d
import time

M_jup = 1.898e30
R_jup = 7.1e9
c = 3e10
h = 6.626e-27
CM_TO_ANGSTROM = 1e8
WAVELENGTH_0 = 912 / CM_TO_ANGSTROM
SINGLET_ION = 505 / CM_TO_ANGSTROM
TRIPLET_ION = 2588 / CM_TO_ANGSTROM

AMU = 1.67e-24
G = 6.67e-8
k_B = 1.38e-16
eV_to_erg = 1.6e-12
Mb_to_cm2 = 1e-18

A31 = 1.272e-4
q13a = 4.5e-20
q31a = 2.6e-8
q31b = 4.0e-9
Q31 = 5e-10

def iter_He_sol(v_interp, rho_interp, a0, photo_rate, T0, Rp, a1, a3, rate1, rate3, fion_interp, max_r_over_Rp=10, num_grid=100, scale_factor=1e4, timeout=1200):
    start = time.time()
    alpha_He = He_recomb_coeffs(T0)

    def tau_derivs(r, variables, f1_interp, f3_interp):
        if time.time() - start > timeout:
            raise TimeoutError("timeout in tau_derivs of iter_He_sol")
        f1, f3 = variables
        f3 /= scale_factor
        tau1_deriv = -rho_interp(r) / 1.3 / AMU * (a0 * (1 - fion_interp(r)) * 0.9 + a1 * 0.1 * f1_interp(r))
        tau3_deriv = -rho_interp(r) / 1.3 / AMU * a3 * 0.1 * f3_interp(r)
        #assert(tau3_deriv <= 0)
        #assert(tau1_deriv <= 0)
        #print(tau1_deriv, tau3_deriv)
        return [tau1_deriv, tau3_deriv]

    def tau_jacob(r, variables, f1_interp, f3_interp):
        jac = np.zeros((2,2))
        jac[0,0] = -rho_interp(r) / 1.3 / AMU * a1 * 0.1 * f1_interp(r)
        jac[1,1] = -rho_interp(r) / 1.3 / AMU * a3 * 0.1
        return jac

    def f_jacob(r, variables, tau1_interp, tau3_interp):        
        f1, f3 = variables
        f3 /= scale_factor
        tau1 = tau1_interp(r)
        tau3 = tau3_interp(r)
        alpha1 = alpha_He / 4
        alpha3 = 3 * alpha_He / 4
        n_H = rho_interp(r) / 1.3 / AMU * 0.9
        n_H0 = (1 - fion_interp(r)) * n_H
        n_e = fion_interp(r) * n_H0
        ionized_frac = 1 - f1 - f3
        v = v_interp(r)
        
        jac = np.zeros((2,2))
        jac[0,0] = (-n_e * alpha1 - rate1 * np.exp(-tau1) - n_e * q13a) / v
        jac[0,1] = (-n_e * alpha1 + A31 + n_e * q13a + n_e * q31b + n_H0 * Q31) / v
        jac[1,0] = (-n_e * alpha3 + n_e * q13a) / v
        jac[1,1] = (-n_e * alpha3 - A31 - rate3 * np.exp(-tau3) - n_e * q31a - n_e * q31b - n_H0 * Q31) / v

        jac[0,1] /= scale_factor
        jac[1,0] *= scale_factor
        return jac
    
    def f_derivs(r, variables, tau1_interp, tau3_interp):
        if time.time() - start > timeout:
            raise TimeoutError("timeout in f_derivs of iter_fion_sol")
        f1, f3 = variables
        f3 /= scale_factor
        tau1 = tau1_interp(r)
        tau3 = tau3_interp(r)
        alpha1 = alpha_He / 4
        alpha3 = 3 * alpha_He / 4
        n_H = rho_interp(r) / 1.3 / AMU * 0.9
        n_H0 = (1 - fion_interp(r)) * n_H
        n_e = fion_interp(r) * n_H0
        ionized_frac = 1 - f1 - f3
        #if ionized_frac < 0: ionized_frac = 0
        #Q31 = 0
        #A31 = 0
        #rate1 = 0
        #rate3 = 0
        f1_deriv = (ionized_frac * n_e * alpha1 + f3 * A31 - f1 * rate1 * np.exp(-tau1_interp(r)) - f1 * n_e * q13a + f3 * n_e * q31a + f3 * n_e * q31b + f3 * n_H0 * Q31) / v_interp(r)
        #print("f1", alpha1, A31, -rate1 * np.exp(-tau1_interp(r)), -n_e * q13a, n_e * q31a, n_e * q31b, n_H0 * Q31)
        f3_deriv = scale_factor * (ionized_frac * n_e * alpha3 - f3 * A31 - f3 * rate3 * np.exp(-tau3_interp(r)) + f1 * n_e * q13a - f3 * n_e * q31a - f3 * n_e * q31b - f3 * n_H0 * Q31) / v_interp(r)
        #print(v_interp(r), n_e * alpha3, - f3 * A31, - f3 * rate3 * np.exp(-tau3_interp(r)), + f1 * n_e * q13a, - f3 * n_e * q31a, - f3 * n_e * q31b, - f3 * n_H0 * Q31)
        #if f3 <= 0: f3_deriv = 0
        #print(r, ionized_frac, f1, f3)
        #if f1 >= 1 and f1_deriv > 0: f1_deriv = 0
        #if f3 <= 0 and f3_deriv < 0: f3_deriv = 0
        #print("r={}, f1={}, f1'={}, f3={}, f3'={}".format(r, f1, f1_deriv, f3, f3_deriv))
        return [f1_deriv, f3_deriv]
        
    min_r = Rp
    r_mesh = np.linspace(min_r, max_r_over_Rp*Rp, num_grid)
    f1_interp = interp1d(r_mesh, np.ones(len(r_mesh)))
    f3_interp = interp1d(r_mesh, np.zeros(len(r_mesh))) #actually times scale_factor
    
    while True:
        result = solve_ivp(lambda r, y: tau_derivs(r, y, f1_interp, f3_interp), (max_r_over_Rp*Rp, min_r), [0, 0], t_eval=r_mesh[::-1],
                           method="Radau",
                           jac=lambda r, y: tau_jacob(r, y, f1_interp, f3_interp))
        tau1_interp = interp1d(r_mesh, result.y[0][::-1])
        tau3_interp = interp1d(r_mesh, result.y[1][::-1])
        #plt.semilogy(r_mesh, tau1_interp(r_mesh))
        #plt.semilogy(r_mesh, tau3_interp(r_mesh))
        #plt.show()
        
        result = solve_ivp(
            lambda r, y: f_derivs(r, y, tau1_interp, tau3_interp),
            (min_r, max_r_over_Rp*Rp),
            [1, 0],
            t_eval=r_mesh,
            method='Radau',
            jac=lambda r, y: f_jacob(r, y, tau1_interp, tau3_interp))

        max_diff_f1 = np.max(np.abs(f1_interp(r_mesh) - result.y[0]))
        max_diff_f3 = np.max(np.abs(f3_interp(r_mesh) - result.y[1]))
        if max_diff_f1 < 1e-3 and max_diff_f3 < 1e-3:
            break
                                                          
        '''if np.allclose(f1_interp(r_mesh), result.y[0]) and np.allclose(f3_interp(r_mesh), result.y[1]):
            break
        else:
            print(np.max(np.abs(f1_interp(r_mesh) - result.y[0])),
                  np.max(np.abs(f3_interp(r_mesh) - result.y[1])))'''
        
        f1_interp = interp1d(r_mesh, result.y[0])
        f3_interp = interp1d(r_mesh, result.y[1])
        
        '''plt.semilogy(r_mesh, f1_interp(r_mesh))
        plt.semilogy(r_mesh, f3_interp(r_mesh))
        plt.title("f1/f3")
        plt.figure()
        plt.semilogy(r_mesh, tau1_interp(r_mesh))
        plt.semilogy(r_mesh, tau3_interp(r_mesh))
        plt.title("Tau")
        plt.show()'''
        
    n_He = 0.1 * rho_interp(r_mesh) / (1.3 * AMU)
    n_singlet = n_He * f1_interp(r_mesh)
    n_triplet = n_He * f3_interp(r_mesh) / scale_factor
    n_HeII = n_He - n_singlet - n_triplet
    n3_interp = interp1d(r_mesh, n_triplet)
    
    '''np.save("r_mesh.npy", r_mesh)
    np.save("n_triplet.npy", n_triplet)
    np.save("all_v.npy", v_interp(r_mesh))
    
    plt.semilogy(r_mesh / Rp, n_singlet)
    plt.semilogy(r_mesh / Rp, n_triplet)
    plt.semilogy(r_mesh / Rp, n_HeII)
    plt.show()'''
    
    return n3_interp

def He_recomb_coeffs(T):
    #Taken from https://arxiv.org/pdf/astro-ph/9509083.pdf
    a = 3.294e-11
    T0 = 15.54
    T1 = 3.676e7
    b = 0.691
    alpha_r = a / (np.sqrt(T/T0) * (1 + np.sqrt(T/T0))**(1-b) * (1 + np.sqrt(T/T1))**(1 + b))

    #Taken from http://articles.adsabs.harvard.edu//full/1991A%26A...251..680P/0000681.000.html
    a = 8.295
    b = -0.5606
    c = 0.9164
    d = 0.2667
    t = T/1e4
    alt_alpha_r = 1e-13 * a * t**b / (1 + c * t**d)
    return (alpha_r + alt_alpha_r) / 2

def get_stellar_spectrum(filename, D_over_a):
    with astropy.io.fits.open(filename) as hdul:
        return hdul[1].data["WAVELENGTH"] / CM_TO_ANGSTROM, hdul[1].data["FLUX"] * CM_TO_ANGSTROM * D_over_a**2

def photoionization_cross_section(wavelength):
    ionized = wavelength < WAVELENGTH_0
    epsilon = np.sqrt(WAVELENGTH_0 / wavelength - 1)
    cross_sections = 6.3e-18 * np.exp(4 - 4*np.arctan(epsilon)/epsilon) / (1 - np.exp(-2*np.pi/epsilon)) * (wavelength / WAVELENGTH_0)**4
    cross_sections[~ionized] = 0
    return cross_sections

def avg_cross_section(wavelengths, fluxes, cross_sections, min_wavelength, max_wavelength):
    in_range = np.logical_and(wavelengths >= min_wavelength, wavelengths <= max_wavelength)
    wavelengths = wavelengths[in_range]
    fluxes = fluxes[in_range]
    cross_sections = cross_sections[in_range]
    
    weighted_sum = np.trapz(fluxes * cross_sections, wavelengths)
    total_fluxes = np.trapz(fluxes, wavelengths)
    avg_cross_section = weighted_sum / total_fluxes
    return avg_cross_section

    print(avg_cross_section)
    plt.loglog(wavelengths, cross_sections)
    plt.figure()
    plt.loglog(wavelengths, fluxes)
    plt.show()

def photoionization_rate(wavelengths, fluxes, cross_sections, min_wavelength, max_wavelength):
    in_range = np.logical_and(wavelengths >= min_wavelength, wavelengths <= max_wavelength) 
    wavelengths = wavelengths[in_range]
    fluxes = fluxes[in_range]
    cross_sections = cross_sections[in_range]
    rate = np.trapz(fluxes * cross_sections / (h * c / wavelengths), wavelengths)
    #print("Rate", rate)
    #plt.loglog(wavelengths, fluxes)
    #plt.show()
    return rate

def parker_solution(T0, mu, planet_mass, mass_loss_rate):
    sound_speed = np.sqrt(k_B * T0 / mu / AMU)
    sonic_r = G * planet_mass / 2 / sound_speed**2
    rho_crit = mass_loss_rate / (4 * np.pi * sonic_r**2 * sound_speed)
    #print(sound_speed, sonic_r, rho_crit)
    
    Rs = np.load("Rs.npy")
    Vs = np.load("Vs.npy")
    Rhos = np.exp(2/Rs - 3./2 - Vs**2/2)

    all_v = Vs * sound_speed
    all_r = sonic_r * Rs
    all_rho = Rhos * rho_crit

    return interp1d(all_r, all_v), interp1d(all_r, all_rho)


def iter_fion_sol(v_interp, rho_interp, a0, photo_rate, T0, Rp, max_r_over_Rp=10, num_grid=100, timeout=1200):
    start = time.time()
    alpha_rec = 2.59e-13 * (T0/1e4)**-0.7
            
    def tau_derivs(r, variables, fion_interp):
        if time.time() - start > timeout:
            raise TimeoutError("timeout in tau_derivs of iter_fion_sol")
        neutral_fraction = 1 - fion_interp(r)
        tau_deriv = -neutral_fraction * rho_interp(r) * 0.9 * a0 / 1.3 / AMU
        assert(tau_deriv < 0)
        return [tau_deriv]

    def tau_jac(r):
        return [0]
    
    def fion_derivs(r, variables, tau_interp):
        if time.time() - start > timeout:
            raise TimeoutError("timeout in fion_derivs of iter_fion_sol")
        
        f_ion = variables[0]
        tau_0 = tau_interp(r)
        neutral_fraction = 1 - f_ion
        fion_deriv = neutral_fraction / v_interp(r) * photo_rate * np.exp(-tau_0) - 0.9 * rho_interp(r) * f_ion**2 * alpha_rec / (1.3 * AMU * v_interp(r))
        #print(r, neutral_fraction / v_interp(r) * photo_rate * np.exp(-tau_0), 0.9 * rho_interp(r) * f_ion**2 * alpha_rec / (1.3 * AMU * v_interp(r)))
        #print(r, f_ion, fion_deriv)
        return [fion_deriv]

    def fion_jac(r, variables, tau_interp):
        f_ion = variables[0]
        jac = np.zeros((1,1))
        jac[0,0] = -1./v_interp(r) * photo_rate * np.exp(-tau_interp(r)) - 0.9 * rho_interp(r) * 2 * f_ion * alpha_rec / (1.3 * AMU * v_interp(r))
        return jac

    r_mesh = np.linspace(Rp, max_r_over_Rp*Rp, num_grid)
    fion_interp = interp1d(r_mesh, np.zeros(len(r_mesh)))
    taus_interp = None #interp1d(r_mesh, np.zeros(len(r_mesh)))
    
    while True:
        result = solve_ivp(lambda r, y: tau_derivs(r, y, fion_interp), (max_r_over_Rp*Rp, Rp), [0], t_eval=r_mesh[::-1], jac=lambda r, y: [[0]], method='Radau')
        taus_interp = interp1d(r_mesh, result.y[0][::-1])
        result = solve_ivp(lambda r, y: fion_derivs(r, y, taus_interp), (Rp, max_r_over_Rp*Rp), [0], t_eval=r_mesh, jac=lambda r, y: fion_jac(r, y, taus_interp), method='Radau')
        max_diff = np.max(np.abs(fion_interp(r_mesh) - result.y[0]))

        if max_diff < 1e-3:
            break
        #if np.allclose(fion_interp(r_mesh), result.y[0]):
        #    break
        fion_interp = interp1d(r_mesh, result.y[0])
        
        '''print(fion_interp(2*Rp), taus_interp(2*Rp))
        plt.plot(r_mesh, fion_interp(r_mesh))
        plt.plot(r_mesh, taus_interp(r_mesh))
        plt.show()'''
        
    return r_mesh, fion_interp, taus_interp


def singlet_cross_sections(wavelengths, data_filename="singlet_He_ionization"):
    energies, data_sigmas = np.loadtxt(data_filename, unpack=True)
    data_wavelengths = h * c / (energies * eV_to_erg)
    data_sigmas *= Mb_to_cm2
    argsort = np.argsort(data_wavelengths)
    data_wavelengths = data_wavelengths[argsort]
    data_sigmas = data_sigmas[argsort]
    
    sigmas_interp = np.interp(wavelengths, data_wavelengths, data_sigmas, left=0, right=0)
    return sigmas_interp

def triplet_cross_sections(wavelengths, data_filename="triplet_He_ionization"):
    data_wavelengths, data_sigmas = np.loadtxt(data_filename, unpack=True)
    data_wavelengths /= CM_TO_ANGSTROM
    data_sigmas *= 8.067e-18 #From paper

    argsort = np.argsort(data_wavelengths)
    data_wavelengths = data_wavelengths[argsort]
    data_sigmas = data_sigmas[argsort]
    sigmas_interp = np.interp(wavelengths, data_wavelengths, data_sigmas, left=0, right=0)
    return sigmas_interp


def get_solution(spectrum_file, Mp, Rp, T0, mass_loss_rate, D_over_a, max_r_over_Rp=10, lyman_alpha=False):    
    v_interp, rho_interp = parker_solution(T0, 1.3, Mp, mass_loss_rate)
    wavelengths, fluxes = get_stellar_spectrum(spectrum_file, D_over_a)

    #Helium parameters
    singlet_sigmas = singlet_cross_sections(wavelengths)
    a1 = avg_cross_section(wavelengths, fluxes, singlet_sigmas, 0, SINGLET_ION)
    triplet_sigmas = triplet_cross_sections(wavelengths)
    a3 = avg_cross_section(wavelengths, fluxes, triplet_sigmas, SINGLET_ION, TRIPLET_ION)

    photo_cross_sections = photoionization_cross_section(wavelengths)
    
    a0 = avg_cross_section(wavelengths, fluxes, photo_cross_sections, 0, WAVELENGTH_0)
    photo_rate = photoionization_rate(wavelengths, fluxes, photo_cross_sections, 0, WAVELENGTH_0)
    rate1 = photoionization_rate(wavelengths, fluxes, singlet_sigmas, 0, SINGLET_ION)
    rate3 = photoionization_rate(wavelengths, fluxes, triplet_sigmas, SINGLET_ION, TRIPLET_ION)

    #print("Params", a1, a3, a0, photo_rate, rate1, rate3)
    r_mesh, fion_interp, taus_interp = iter_fion_sol(v_interp, rho_interp, a0, photo_rate, T0, Rp, max_r_over_Rp)
    n3_interp = iter_He_sol(v_interp, rho_interp, a0, photo_rate, T0, Rp, a1, a3, rate1, rate3, fion_interp, max_r_over_Rp)
    if lyman_alpha:
        n_HI = (1 - fion_interp(r_mesh)) * 0.1 * rho_interp(r_mesh) / (1.3 * AMU)
        n_HI_interp = interp1d(r_mesh, n_HI)
        return r_mesh, v_interp, n_HI_interp
    
    return r_mesh, v_interp, n3_interp

#get_solution(sys.argv[1], 0.073 * M_jup, 0.38 * R_jup, 5000, 2e10)

