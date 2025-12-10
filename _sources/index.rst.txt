COBI: Cosmic Birefringence Analysis Pipeline
=============================================

.. image:: https://readthedocs.org/projects/cobi/badge/?version=latest
   :target: https://cobi.readthedocs.io/
   :alt: Documentation Status

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

A comprehensive Python package for cosmic birefringence analysis using Simons Observatory Large Aperture Telescope (LAT) and Small Aperture Telescopes (SAT) data.

Overview
--------

**COBI** provides end-to-end functionality for detecting and characterizing cosmic birefringence—a parity-violating effect that rotates the polarization plane of CMB photons. This rotation can arise from physics beyond the Standard Model, including axion-like particles and Chern-Simons modifications to electromagnetism.

Key Features
------------

🌌 **Full-sky CMB Simulation**
   Generate realistic CMB maps with cosmic birefringence effects

🎨 **Foreground Modeling**
   Galactic dust and synchrotron emission with spatial templates

📊 **Power Spectrum Analysis**
   Pseudo-Cℓ computation with NaMaster mode-coupling correction

🔍 **Quadratic Estimator**
   Reconstruct birefringence angle maps from EB correlations

📈 **Maximum Likelihood Estimation**
   Fit calibration angles and birefringence amplitudes

🚀 **MPI Support**
   Parallel processing for large-scale simulations

🔗 **Cross-correlation**
   LAT-SAT joint analysis for systematic mitigation

Installation
------------

Prerequisites
^^^^^^^^^^^^^

COBI requires several dependencies with C extensions. We recommend using conda::

   # Clone the repository
   git clone https://github.com/antolonappan/cobi.git
   cd cobi
   
   # Create conda environment
   conda env create -f conda/environment_with_build.yml
   conda activate cobi
   
   # Install COBI
   pip install -e .

Quick Start Examples
--------------------

1. Simulate CMB with Birefringence
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cobi.simulation import CMB
   import matplotlib.pyplot as plt
   
   # Isotropic birefringence: constant rotation angle
   cmb = CMB(
       libdir='./sims',
       nside=2048,
       model='iso',
       beta=0.35,  # rotation angle in degrees
       lensing=True,
       verbose=True
   )
   
   # Visualize EB power spectrum (signature of birefringence)
   plt.loglog(cmb.get_cb_lensed_spectra(beta=0.35)['eb'], label='β=0.35°')
   plt.xlabel('$\\ell$')
   plt.ylabel('$C_\\ell^{EB}$')

2. Complete Sky Simulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cobi.simulation import LATsky
   
   # LAT observation simulation with systematics
   lat = LATsky(
       libdir='./sims',
       nside=2048,
       cb_model='iso',
       beta=0.35,
       alpha=[-0.1, -0.1, 0.2, 0.2, 0.15, 0.15],  # miscalibration per freq
       alpha_err=0.1,
       bandpass=True,
       noise_model='NC',
       verbose=True
   )
   
   # Get observed Q/U maps
   QU_27 = lat.obsQU(idx=0, band='27-1')

3. Calibration Analysis
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cobi.simulation import LATskyC, SATskyC
   from cobi.spectra import Spectra
   from cobi.calibration import Sat4Lat
   
   # Setup simulations
   lat = LATskyC(libdir='./sims', nside=2048, cb_model='iso', beta=0.35)
   sat = SATskyC(libdir='./sims', nside=2048, cb_model='iso', beta=0.35)
   
   # Compute spectra
   lat_spec = Spectra(lat, libdir='./spec', galcut=40, binwidth=5)
   sat_spec = Spectra(sat, libdir='./spec', galcut=40, binwidth=5)
   
   # Fit birefringence
   calib = Sat4Lat(
       libdir='./calib',
       lmin=100,
       lmax=3000,
       latlib=lat_spec,
       satlib=sat_spec,
       sat_err=0.1,
       beta=0.35
   )
   
   calib.plot_getdist(nwalkers=100, nsamples=2000)

4. Quadratic Estimator Reconstruction
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from cobi.quest import FilterEB, QE
   
   # Setup filtering
   filt = FilterEB(lat, mask, lmax=3000, sht_backend='ducc')
   
   # Quadratic estimator reconstruction
   qe = QE(filt, lmin=100, lmax=3000, recon_lmax=2048)
   
   # Reconstruct birefringence map
   beta_lm_recon = qe.qlm(idx=0) - qe.mean_field()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: API Documentation
   
   api

Project Information
-------------------

:Author: Anto I. Lonappan
:Contributors: Patricia Diego-Palazuelos, Carlos Hervías-Caimapo
:License: MIT
:Repository: https://github.com/antolonappan/cobi
:Documentation: https://cobi.readthedocs.io/

Citation
--------

If you use COBI in your research, please cite:

.. code-block:: bibtex

   @software{cobi2024,
     author = {Lonappan, Anto I.},
     title = {COBI: Cosmic Birefringence Analysis Pipeline},
     year = {2024},
     url = {https://github.com/antolonappan/cobi}
   }

Related Publications
--------------------

* Diego-Palazuelos, P., et al. (2023). "Cosmic Birefringence from the Planck Data Release 4"
* Minami, Y., & Komatsu, E. (2020). "New Extraction of the Cosmic Birefringence from the Planck 2018 Polarization Data"

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. note::

   This project is under active development. Features and API may change in future releases.

