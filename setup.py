"""
Setup configuration for COBI - Cosmic Birefringence Analysis Pipeline.
"""
from setuptools import setup, find_packages
import os

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read version from __init__.py
def get_version():
    """Extract version from cobi/__init__.py"""
    init_file = os.path.join("cobi", "__init__.py")
    with open(init_file, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

# Core dependencies
install_requires = [
    "numpy>=1.22.0",
    "scipy>=1.9.0",
    "matplotlib>=3.5.0",
    "astropy>=5.0",
    "healpy>=1.16.0",
    "h5py>=3.7.0",
    "tqdm>=4.64.0",
    "pandas>=1.5.0",
]

# Optional dependencies for advanced features
extras_require = {
    "full": [
        "pymaster>=2.0.0",      # NaMaster for power spectra
        "pysm3>=3.4.0",         # PySM for foregrounds
        "camb>=1.4.0",          # CAMB for CMB power spectra
        "ducc0>=0.30.0",        # DUCC for fast SHTs
        "emcee>=3.1.0",         # MCMC sampling
        "getdist>=1.4.0",       # MCMC analysis and plotting
        "mpi4py>=3.1.0",        # MPI parallelization
        "lenspyx>=1.0.0",       # CMB lensing
        "pixell>=0.20.0",       # Flat-sky utilities
    ],
    "docs": [
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "nbsphinx>=0.9.0",
        "sphinx-autodoc-typehints>=1.19.0",
    ],
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=0.990",
    ],
}

# Add 'all' option to install everything
extras_require["all"] = list(set(sum(extras_require.values(), [])))

setup(
    name="cobi",
    version=get_version(),
    author="Anto I. Lonappan",
    author_email="antolonappan@icloud.com",
    maintainer="Anto I. Lonappan",
    maintainer_email="antolonappan@icloud.com",
    description="Cosmic Birefringence Analysis Pipeline for Simons Observatory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/antolonappan/cobi",
    project_urls={
        "Documentation": "https://cobi.readthedocs.io/",
        "Source": "https://github.com/antolonappan/cobi",
        "Bug Reports": "https://github.com/antolonappan/cobi/issues",
    },
    packages=find_packages(include=["cobi", "cobi.*"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "cosmology",
        "cmb",
        "cosmic-birefringence",
        "simons-observatory",
        "polarization",
        "parity-violation",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            # Add command-line scripts here if needed
            # "cobi-sim=cobi.cli:simulate",
        ],
    },
    zip_safe=False,
)

 