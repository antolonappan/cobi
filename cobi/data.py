"""
Data Loading and Management Module
===================================

This module provides data classes and functions to download and manage
data files for the COBI package.

The module handles loading of:
- Binary masks (SAT, LAT, CO, GAL, PS)
- Bandpass profiles
- CAMB initialization files
- Pre-computed power spectra
- Time-dependent isotropic spectra

All data files are automatically downloaded from the GitHub release
repository if not present locally.

Classes
-------
Data
    Main data class that handles file downloading, caching, and loading
    for various file formats (FITS, pickle, INI).

Data Objects
------------
SAT_MASK : Data
    Binary mask for SAT observations
LAT_MASK : Data
    Binary mask for LAT observations
CO_MASK : Data
    Binary mask for CO emission regions
GAL_MASK : Data
    Binary mask for Galactic plane (parameterized by galcut)
PS_MASK : Data
    Binary mask for polarized point sources
BP_PROFILE : Data
    Bandpass profile data
CAMB_INI : Data
    CAMB initialization file for CMB spectrum calculation
SPECTRA : Data
    Pre-computed CMB power spectra
ISO_TD_SPECTRA : Data
    Time-dependent isotropic birefringence spectra

Example
-------
    from cobi.data import LAT_MASK
    
    # Set directory where data will be cached
    LAT_MASK.directory = '/path/to/cache'
    
    # Load the mask (downloads if not present)
    mask = LAT_MASK.data
"""

# General imports
import os
from pickle import load
from camb import read_ini
from healpy import read_map
from typing import Any, Optional
from dataclasses import dataclass,field
# Local imports
from cobi.utils import download_file
from cobi import mpi

@dataclass
class Data:
    filename: str
    _directory: Optional[str] = field(default=None, repr=False)
    _galcut: Optional[int] = field(default=0, repr=False)


    @property
    def directory(self) -> Optional[str]:
        return self._directory
    
    @property
    def galcut(self) -> Optional[int]:
        return self._galcut

    @directory.setter
    def directory(self, value: str) -> None:
        if not os.path.isdir(value):
            print(f"The directory {value} does not exist. I will try to create it.")
            try:
                os.makedirs(value, exist_ok=True)
            except Exception as e:
                raise ValueError(f"Failed to create the directory {value}: {e}")
        self._directory = value

    @galcut.setter
    def galcut(self, value: int) -> None:
        if value < 0:
            raise ValueError("The galcut value must be non-negative.")
        self._galcut = value

    def __dir__(self) -> str:
        assert self.directory is not None, 'Directory is not set.'
        return os.path.join(self.directory, 'Data')

    @property
    def fname(self) -> str:
        directory = self.__dir__()
        return os.path.join(directory, self.filename)

    @property
    def url(self) -> str:
        return f"https://github.com/antolonappan/cobi/releases/download/1.0/{self.filename}"
    
    

    def __load__(self, fname: str) -> Any:
        ext = fname.split('.')[-1]
        match ext:
            case 'fits':  
                return read_map(fname, field=self._galcut) # type: ignore
            case 'pkl':
                return load(open(fname, 'rb'))
            case 'ini':
                return read_ini(fname)
            case _:
                raise ValueError(f'Unknown file extension: {ext}')

    @property
    def data(self) -> Any:
        fname = self.fname
        if os.path.isfile(fname):
            return self.__load__(fname)
        else:
            if mpi.rank == 0:
                os.makedirs(self.__dir__(), exist_ok=True)
                download_file(self.url, fname)
            mpi.barrier()
            return self.__load__(fname)


SAT_MASK = Data('binary_SAT_mask_N1024.fits')
LAT_MASK = Data('binary_LAT_mask_N1024.fits')
CO_MASK = Data('binary_CO_mask_N1024.fits')
GAL_MASK = Data("binary_GAL_mask_N1024.fits")
PS_MASK = Data('binary_comb_PS_mask_N1024.fits')
BP_PROFILE = Data('bp_profile.pkl')
CAMB_INI = Data('cb.ini')
SPECTRA = Data('spectra.pkl')
ISO_TD_SPECTRA = Data('iso_time_dep.pkl')