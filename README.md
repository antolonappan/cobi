# COBI - Cosmic Birefringence Analysis Pipeline

[![Documentation Status](https://readthedocs.org/projects/cobi/badge/?version=latest)](https://cobi.readthedocs.io/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://antolonappan.github.io/cobi/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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

# Generate CMB map with β = 0.3° isotropic rotation
cmb = CMB(
    nside=512,
    lmax=1500,
    beta=0.3,  # birefringence angle in degrees
    output_dir='./sims'
)

# Create realization
cmb_map = cmb.get_map(idx=0, apply_beta=True)
```

### 2. Compute Power Spectra

```python
from cobi.spectra import SpectraSingle
import healpy as hp

# Load mask
mask = hp.read_map('mask.fits')

# Compute auto-spectrum with NaMaster
spec = SpectraSingle(
    libdir='./spectra',
    sky=cmb_map,
    mask=mask,
    lmax=1500,
    beams=[1.4, 1.4]  # arcmin FWHM for T/P
)

# Get EB spectrum (signature of birefringence)
ell, eb_spectrum = spec.get_spectra('EB', idx=0)
```

### 3. Reconstruct Birefringence with Quadratic Estimator

```python
from cobi.quest import FilterEB, QE

# Set up EB filtering
filter_eb = FilterEB(
    sky=cmb_map,
    mask=mask,
    lmax=1500,
    libdir='./filter'
)

# Initialize quadratic estimator
qe = QE(
    filter_eb=filter_eb,
    lmin=30,
    lmax=300,
    recon_lmax=100
)

# Reconstruct birefringence map
beta_lm = qe.qlm(idx=0)
beta_map = hp.alm2map(beta_lm, nside=512)
```

### 4. Calibration Analysis (LAT-SAT Cross-correlation)

```python
from cobi.calibration import Sat4LatCross
from cobi.spectra import SpectraCross

# Cross-correlate LAT and SAT
spec_cross = SpectraCross(
    libdir='./cross_spec',
    lat_sky=lat_map,
    sat_sky=sat_map,
    lat_mask=lat_mask,
    sat_mask=sat_mask
)

# Fit for miscalibration angle
calib = Sat4LatCross(
    spec_cross=spec_cross,
    sat_err=0.1,  # SAT calibration uncertainty (degrees)
    beta_fid=0.0,
    lat_lrange=(30, 300)
)

# Run MCMC
samples = calib.run_mcmc(nwalkers=32, nsamples=2000)
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

- Diego-Palazuelos, P., et al. (2023). "Cosmic Birefringence from the Planck Data Release 4"
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
