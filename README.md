# COBI - Cosmic Birefringence Analysis Pipeline

[![Documentation Status](https://readthedocs.org/projects/cobi/badge/?version=latest)](https://cobi.readthedocs.io/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://antolonappan.github.io/cobi/)
[![Tests](https://github.com/antolonappan/cobi/workflows/Tests/badge.svg)](https://github.com/antolonappan/cobi/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A comprehensive Python package for cosmic birefringence analysis using Simons Observatory Large Aperture Telescope (LAT) and Small Aperture Telescopes (SAT) data.

## Overview

**COBI** provides end-to-end functionality for detecting and characterizing cosmic birefringence—a parity-violating effect that rotates the polarization plane of CMB photons. This rotation can arise from physics beyond the Standard Model, including axion-like particles and Chern-Simons modifications to electromagnetism.

### Key Features

- 🌌 **Full-sky CMB Simulation**: Generate realistic CMB maps with cosmic birefringence effects
- 🎨 **Foreground Modeling**: Galactic dust and synchrotron emission with spatial templates
- 📊 **Power Spectrum Analysis**: Pseudo-Cℓ computation with NaMaster mode-coupling correction
- 🔍 **Quadratic Estimator**: Reconstruct birefringence angle maps from EB correlations
- 📈 **Maximum Likelihood Estimation**: Fit calibration angles and birefringence amplitudes
- 🚀 **MPI Support**: Parallel processing for large-scale simulations
- 🔗 **Cross-correlation**: LAT-SAT joint analysis for systematic mitigation

## Installation

### Prerequisites

COBI requires several dependencies with C extensions. We recommend using conda:

```bash
# Clone the repository
git clone https://github.com/antolonappan/cobi.git
cd cobi

# Create conda environment
conda env create -f conda/environment_with_build.yml
conda activate cobi

# Install COBI
pip install -e .
```

### Core Dependencies

- Python ≥ 3.8
- `healpy` - HEALPix pixelization
- `pymaster` (NaMaster) - Power spectrum estimation
- `ducc0` - Fast spherical harmonic transforms
- `camb` - CMB power spectrum generation
- `pysm3` - Sky model templates
- `emcee` - MCMC sampling
- `mpi4py` - Parallel computing (optional)

## Quick Start

### 1. Simulate CMB with Birefringence

```python
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
plt.loglog(cmb.get_cb_lensed_spectra(beta=0.1)['eb'], label='β=0.1°')
plt.xlabel('$\\ell$')
plt.ylabel('$C_\\ell^{EB}$')
plt.legend()
```

**Anisotropic birefringence** (spatially-varying rotation):

```python
# Patchy reionization model with scale-invariant power spectrum
cmb_aniso = CMB(
    libdir='./sims',
    nside=1024,
    model='aniso',
    Acb=4.0e-6,  # amplitude parameter
    lensing=False,
    verbose=True
)

# Get birefringence angle map
beta_map = cmb_aniso.alpha_map(idx=0)
```

### 2. Complete Sky Simulation (CMB + Foregrounds + Noise)

```python
from cobi.simulation import LATsky
import numpy as np

# LAT observation simulation with systematics
nside = 2048
cb_model = "iso"
beta = 0.35
alpha = [-0.1, -0.1, 0.2, 0.2, 0.15, 0.15]  # miscalibration per freq
alpha_err = 0.1  # calibration uncertainty

lat = LATsky(
    libdir='./sims',
    nside=nside,
    cb_model=cb_model,
    beta=beta,
    alpha=alpha,
    alpha_err=alpha_err,
    bandpass=True,
    noise_model='NC',  # or 'TOD'
    verbose=True
)

# Get observed Q/U maps for specific frequency and split
QU_27 = lat.obsQU(idx=0, band='27-1')
```

### 3. Power Spectra Analysis

```python
from cobi.spectra import Spectra

# Compute all auto- and cross-spectra with NaMaster
spec = Spectra(lat, libdir='./spectra', cache=True, parallel=1)

# Get full covariance matrix of observed spectra
obs_spectra = spec.obs_x_obs(idx=0)  # Shape: (nfreqs, nfreqs, 4, nbins)
```

### 4. Calibration: SAT Calibrating LAT

```python
from cobi.simulation import LATskyC, SATskyC
from cobi.spectra import Spectra
from cobi.calibration import Sat4Lat

# Setup LAT and SAT simulations
alpha_lat = [0.2, 0.2]
alpha_lat_err = 0.2
alpha_sat_err = 0.1

lat = LATskyC(
    libdir='./sims',
    nside=2048,
    cb_model='iso',
    beta=0.35,
    alpha=alpha_lat,
    alpha_err=alpha_lat_err,
    nsplits=2,
    noise_model='TOD'
)

sat = SATskyC(
    libdir='./sims',
    nside=2048,
    cb_model='iso',
    beta=0.35,
    alpha_err=alpha_sat_err,
    nsplits=2,
    noise_model='TOD'
)

# Compute spectra with galactic cut
lat_spec = Spectra(lat, libdir='./spec', galcut=40, binwidth=5)
sat_spec = Spectra(sat, libdir='./spec', galcut=40, binwidth=5)

# Fit birefringence and calibration angles
calib = Sat4Lat(
    libdir='./calib',
    lmin=100,
    lmax=3000,
    latlib=lat_spec,
    satlib=sat_spec,
    sat_err=alpha_sat_err,
    beta=0.35
)

# Run MCMC and visualize posteriors
calib.plot_getdist(nwalkers=100, nsamples=2000, beta_only=True)
```

### 5. Maximum Likelihood Estimation (Minami-Komatsu Method)

```python
from cobi.mle import MLE

# Fit birefringence and calibration angles simultaneously
mle = MLE(
    libdir='./mle',
    spec=spec,
    fit="Ad + beta + alpha",  # dust amplitude + β + calibration
    alpha_per_split=False,
    rm_same_tube=True,
    binwidth=10,
    bmin=50,
    bmax=2000
)

# Estimate angles for realization
results = mle.estimate_angles(idx=1)
print(f"β = {results['beta']:.4f}°")
print(f"α_93 = {results['93']:.4f}°")
```

### 6. Quadratic Estimator Reconstruction (Anisotropic β)

```python
from cobi.simulation import LATsky
from cobi.simulation import Mask
from cobi.quest import FilterEB, QE
import healpy as hp

# Simulate anisotropic birefringence
lat = LATsky(
    libdir='./sims',
    nside=1024,
    cb_model='aniso',
    Acb=4.0e-6,
    AEcb=-1.0e-3,
    lensing=False,
    nsplits=1
)

# Setup mask
mask = Mask(lat.basedir, lat.nside, 'LAT+GAL', apo_scale=2, gal_cut=0.8)

# Step 1: Component separation (Harmonic ILC)
EB_alm, noise_cl = lat.HILC_obsEB(idx=0)

# Step 2: Filtering with C^-1 (pixel-domain)
filt = FilterEB(lat, mask, lmax=3000, sht_backend='ducc')
filt.plot_cinv(idx=0, lmax=2000)

# Step 3: Quadratic estimator reconstruction
qe = QE(filt, lmin=100, lmax=3000, recon_lmax=2048)

# Reconstruct birefringence power spectrum
qcl_list = []
for i in range(50):
    qcl_list.append(qe.qcl(i))
qcl_mean = np.mean(qcl_list, axis=0)

# Reconstruct birefringence map
beta_lm_recon = qe.qlm(idx=0) - qe.mean_field()
beta_map_recon = hp.alm2map(beta_lm_recon, nside=32, lmax=10)
```

## Documentation

📚 **Full documentation available at:**
- [ReadTheDocs](https://cobi.readthedocs.io/) (primary)
- [GitHub Pages](https://antolonappan.github.io/cobi/) (mirror)

Documentation includes:
- Detailed API reference for all modules
- Configuration guides for α-parameter analysis
- Tutorial notebooks with complete examples
- Mathematical background on quadratic estimators

## Project Structure

```
cobi/
├── calibration.py      # Calibration analysis and fitting
├── data.py            # Data loading utilities
├── mle.py             # Maximum likelihood estimation
├── quest.py           # Quadratic estimator implementation
├── spectra.py         # Power spectrum computation
├── sht.py             # Spherical harmonic transforms
├── utils.py           # Helper functions
├── mpi.py             # MPI parallelization support
└── simulation/        # Simulation framework
    ├── cmb.py         # CMB map generation
    ├── dust.py        # Galactic dust modeling
    ├── synchrotron.py # Synchrotron emission
    ├── noise.py       # Instrumental noise
    ├── mask.py        # Survey mask generation
    └── sky.py         # Combined sky model
```

## Citation

If you use COBI in your research, please cite:

```bibtex
@software{cobi2024,
  author = {Lonappan, Anto I.},
  title = {COBI: Cosmic Birefringence Analysis Pipeline},
  year = {2024},
  url = {https://github.com/antolonappan/cobi}
}
```

## Related Publications

- Diego-Palazuelos, P., et al. (2022). "Cosmic Birefringence from the Planck Data Release 4"
- Minami, Y., & Komatsu, E. (2020). "New Extraction of the Cosmic Birefringence from the Planck 2018 Polarization Data"

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

- **Anto I. Lonappan** - [antolonappan](https://github.com/antolonappan)

## Contributors

- **Patricia Diego-Palazuelos** - Max Planck Institute for Astrophysics
- **Carlos Hervías-Caimapo** - Pontificia Universidad Católica de Chile

## Acknowledgments

- Simons Observatory Collaboration
- CMB-S4 Collaboration
- NaMaster (pymaster) developers
- HEALPix and healpy maintainers

## Contact

For questions or support:
- 📧 Email: antolonappan@icloud.com
- 🐛 Issues: [GitHub Issues](https://github.com/antolonappan/cobi/issues)
- 📖 Docs: [cobi.readthedocs.io](https://cobi.readthedocs.io/)

---

**Note**: This package is under active development. Features and API may change in future releases.
