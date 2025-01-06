# This file contains the data classes and functions to download the data files from the repository.

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
            raise ValueError(f"The directory {value} does not exist.")
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