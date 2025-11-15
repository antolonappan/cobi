"""
Simulation Subpackage
=====================

This subpackage provides comprehensive simulation capabilities for CMB
observations including signal, foreground, and noise components.

Modules
-------
cmb
    CMB power spectrum calculation and map generation with cosmic
    birefringence effects (isotropic and time-dependent models).

fg
    Galactic foreground simulation including thermal dust and
    synchrotron emission with realistic spatial and spectral properties.

mask
    Mask generation for observations including Galactic plane cuts,
    point source masking, CO emission masking, and apodization.

noise
    Realistic noise simulation for LAT and SAT including white noise
    and 1/f components with proper spatial correlations.

sky
    Complete sky simulation classes that combine CMB, foregrounds, and
    noise for LAT and SAT observations. Includes bandpass integration
    and component separation via harmonic ILC.

Classes
-------
CMB
    CMB simulation with cosmic birefringence
synfast_pol
    Fast polarization map generation
Foreground
    Foreground component simulation
BandpassInt
    Bandpass integration utilities  
HILC
    Harmonic Internal Linear Combination
Mask
    Observation mask generation
Noise
    Noise realization generation
NoiseSpectra
    Noise power spectra
LATsky
    LAT observation simulation
SATsky
    SAT observation simulation
LATskyC
    LAT calibration simulation
SATskyC
    SAT calibration simulation
"""

from .cmb import CMB, synfast_pol
from .fg import Foreground, BandpassInt, HILC
from .mask import Mask
from .noise import Noise, NoiseSpectra
from .sky import LATsky, SATsky,LATskyC, SATskyC