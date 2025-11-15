COBI: Cosmic Birefringence Analysis Pipeline
=============================================

**COBI** is a comprehensive Python package for cosmic birefringence analysis 
using observations from the Simons Observatory Large Aperture Telescope (LAT) 
and Small Aperture Telescopes (SATs).

Features
--------

* **CMB Simulation**: Generate realistic CMB polarization maps with cosmic birefringence
* **Foreground Modeling**: Thermal dust and synchrotron emission with realistic spectral properties
* **Noise Simulation**: White noise and 1/f components with proper spatial correlations
* **Power Spectrum Analysis**: Pseudo-Cℓ estimation with NaMaster
* **Calibration**: Maximum likelihood estimation of miscalibration angles
* **Quadratic Estimator**: Birefringence angle reconstruction from EB correlations
* **Cross-Correlation**: LAT×SAT analysis for enhanced constraints

Quick Start
-----------

Installation::

   git clone https://github.com/antolonappan/cobi.git
   cd cobi
   pip install -e .

Basic Usage::

   from cobi.calibration import Sat4LatCross
   from cobi.spectra import SpectraCross
   
   # Set up cross-spectrum analysis
   spec_cross = SpectraCross(libdir, lat_sky, sat_sky)
   
   # Fit cosmic birefringence angle
   calib = Sat4LatCross(
       spec_cross=spec_cross,
       sat_err=0.1,
       beta_fid=0.0,
       lat_lrange=(30, 300)
   )
   
   # Run MCMC
   samples = calib.run_mcmc(nwalkers=32, nsamples=2000)

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   
   api

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

About
-----

**Authors**: Anto Idicherian Lonappan, P. Diego-Palazuelos

**License**: MIT

**Repository**: https://github.com/antolonappan/cobi

**Documentation**: https://cobi.readthedocs.io/

Citation
--------

If you use COBI in your research, please cite::

   @software{cobi2024,
     author = {Lonappan, Anto Idicherian and Diego-Palazuelos, P.},
     title = {COBI: Cosmic Birefringence Analysis Pipeline},
     year = {2024},
     url = {https://github.com/antolonappan/cobi}
   }

.. note::

   This project is under active development. Features and API may change.
