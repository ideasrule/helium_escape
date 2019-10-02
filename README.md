This package models the excess absorption from escaping helium, following
the model of Oklopcic & Hirata 2018.  To run it, look at the examples within
rad_transfer.py.  The first argument (sys.argv[1]) must be a stellar spectrum
in the same format as the MUSCLES spectra (https://archive.stsci.edu/prepds/muscles/).  Namely, it must be a FITS file where hdul[1].data["WAVELENGTH"] is the
wavelength in angstroms, and hdul[1].data["FLUX"] is the flux in
erg/s/cm^2/Angstrom as measured on Earth (not on the exoplanet).