"""
Cosmic Birefringence Analysis Pipeline for SO LAT
==================================================

**cobi** is a comprehensive Python package for cosmic birefringence analysis
using data from the Simons Observatory Large Aperture Telescope (LAT) and 
Small Aperture Telescopes (SATs).

The package provides end-to-end functionality for:

- CMB simulation and foreground modeling (dust, synchrotron)
- Polarization map generation with realistic noise
- Power spectrum computation using NaMaster
- Calibration angle estimation via Maximum Likelihood
- Quadratic estimator reconstruction for cosmic birefringence
- Cross-correlation analysis between LAT and SAT

Main Modules
------------

calibration
    Calibration analysis for LAT-SAT cross-spectrum studies, including
    birefringence angle (β) fitting and amplitude parameter (A_EB) estimation.

data
    Data loading utilities for masks, spectra, and CAMB parameter files.

mle
    Maximum Likelihood Estimation for calibration parameter recovery.

quest
    Quadratic Estimator (QE) implementation for birefringence reconstruction.

simulation
    Simulation framework including:
    - CMB map generation with cosmic birefringence
    - Galactic foreground modeling (dust and synchrotron)
    - Realistic noise simulation
    - Sky model combination

spectra
    Power spectrum computation using pseudo-Cℓ methods with mode coupling
    correction via NaMaster.

sht
    Spherical harmonic transforms using DUCC0 for improved performance.

utils
    Utility functions for logging, coordinate transformations, and map operations.

mpi
    MPI support for parallel computation.

Example Usage
-------------

Calibration Analysis::

    from cobi.calibration import Sat4LatCross
    from cobi.spectra import SpectraCross
    
    # Set up cross-spectrum analysis
    spec_cross = SpectraCross(libdir, lat_sky, sat_sky)
    
    # Fit birefringence angle
    calib = Sat4LatCross(
        spec_cross=spec_cross,
        sat_err=0.1,
        beta_fid=0.0,
        lat_lrange=(30, 300)
    )
    
    # Run MCMC
    samples = calib.run_mcmc(nwalkers=32, nsamples=2000)

Quadratic Estimator::

    from cobi.quest import FilterEB, QE
    
    # Set up filtering
    filter_eb = FilterEB(lat_sky, mask, lmax=3000)
    
    # Compute quadratic estimator
    qe = QE(filter_eb, lmin=30, lmax=300, recon_lmax=100)
    qlm = qe.qlm(idx=0)

Authors
-------
Anto Idicherian Lonappan, P. Diego-Palazuelos

License
-------
MIT License

References
----------
For more details on the methodology, see the associated publications
and documentation at https://cobi.readthedocs.io/
"""
__version__ = "1.0"