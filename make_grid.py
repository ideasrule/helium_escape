import numpy as np
import matplotlib.pyplot as plt
import sys
from rad_transfer import predict_depths
import pickle

R_sun = 7e10
M_earth = 5.97e27
R_earth = 6.378e8
parsec = 3.086e18
AU = 1.496e13
res = 375000

#55 Cnc e
Mp = 8.08 * M_earth
Rp = 1.91 * R_earth
Rs = 0.943 * R_sun
D_over_a = 12.59 * parsec / 0.0155 / AU
hill_over_Rp = 4
output_file = "final_grid_dict_55cnc.pkl"

'''#TOI 1726.01
Mp = 7.9 * M_earth
Rp = 2.3 * R_earth
Rs = 0.9 * R_sun
D_over_a = 22.3 * parsec / 0.073 / AU
hill_over_Rp = 11
output_file = "final_grid_dict_toi1726_01.pkl"

#TOI 1726.01
Mp = 9.1 * M_earth
Rp = 2.9 * R_earth
Rs = 0.9 * R_sun
D_over_a = 22.3 * parsec / 0.15 / AU
hill_over_Rp = 24
output_file = "final_grid_dict_toi1726_02.pkl"'''

#mass_loss_rates = 10**np.linspace(7, 14, 50)
mass_loss_rates = 10**np.linspace(7, 11, 50)
exosphere_temps = np.linspace(2500, 10000, 50)
wavenums = np.exp(np.arange(np.log(9230), np.log(9233), 1./res))

grid = np.zeros((len(mass_loss_rates), len(exosphere_temps), len(wavenums)))
for i, rate in enumerate(mass_loss_rates):
    for j, T0 in enumerate(exosphere_temps):
        #print(np.log10(rate), T0)
        transit_spectrum = predict_depths(wavenums, sys.argv[1], Mp, Rp, T0, rate, Rs, D_over_a, hill_over_Rp)
        grid[i,j] = transit_spectrum
        print("log10(rate), T0, max depth (ppm):", np.log10(rate), T0, np.max(transit_spectrum) * 1e6)

final_output = {"mass_loss_rates": mass_loss_rates,
                "exosphere_temps": exosphere_temps,
                "wavenums": wavenums,
                "depths_grid": grid}
#np.save("grid.npy", grid)
with open(output_file, "wb") as f:
    pickle.dump(final_output, f)
