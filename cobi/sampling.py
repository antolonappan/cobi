import emcee
import numpy as np
import matplotlib.pyplot as plt
from cobi.utils import Binner
from getdist import MCSamples, plots
import os
import pickle as pl


def sci_str(x):
    s = f"{x:.0e}"
    mantissa, exp = s.split('e')
    return f"{mantissa}e{int(exp)}"


class AcbLikelihood:
    def __init__(self, libdir, binner, qcl: np.ndarray,
                 lmin=2, lmax=50, mp=1.0,
                 fiducial=None):
        """
        Implements Hamimeche-Lewis (HL) likelihood for anisotropic birefringence.
        fiducial: fiducial bandpowers (default: MC mean)
        """
        self.libdir = libdir
        os.makedirs(libdir, exist_ok=True)
        self.lmin = lmin
        self.lmax = lmax
        self.qcl = qcl
        self.binner = binner
        self.b = binner.b
        self.sel = np.where((self.b >= lmin) & (self.b <= lmax))[0]
        # "Data" vector and MC stats on selected bins
        self.mean = qcl.mean(axis=0)[self.sel]
        self.cov = np.cov(qcl.T)[self.sel][:, self.sel]
        self.std = qcl.std(axis=0)[self.sel]
        self.icov = np.linalg.inv(self.cov)
        self.mp = mp
        # HL fiducial
        self.fiducial = fiducial[self.sel] if fiducial is not None else self.mean

    def theory(self, Acb):
        l = np.arange(1024 + 1)
        cl = Acb * 2 * np.pi / (l**2 + l + 1e-30) * self.mp
        cl[0], cl[1] = 0.0, 0.0
        return self.binner.bin_cell(cl)[self.sel]

    def plot(self, Acb):
        plt.figure(figsize=(6, 6))
        plt.errorbar(self.b[self.sel], self.mean, yerr=self.std, fmt='o')
        plt.loglog(self.b[self.sel], self.theory(Acb), label='Theory')
        plt.xlabel(r'$L$')
        plt.ylabel(r'$C_L^{\alpha\alpha}$')
        plt.grid()
        plt.legend()
        plt.show()

    def ln_prior(self, Acb):
        return 0.0 if (0 < Acb < 1e-5) else -np.inf

    # ---------- Likelihood ----------
    def _hl_transform(self, x):
        """
        HL g-function: sign(x-1) * sqrt(2*(x - ln x - 1)) for x > 0.
        Handles arrays; ignores invalid for x=1 (g=0).
        """
        with np.errstate(invalid='ignore'):
            return np.sign(x - 1) * np.sqrt(2 * (x - np.log(x) - 1))

    def ln_likelihood(self, Acb):
        model = self.theory(Acb)
        if np.any(model <= 0) or np.any(self.mean <= 0) or np.any(self.fiducial <= 0):
            return -np.inf
        # Ratios relative to fiducial
        ratio_data = self.mean / self.fiducial
        ratio_model = model / self.fiducial
        g_data = self._hl_transform(ratio_data)
        g_model = self._hl_transform(ratio_model)
        # Handle any NaNs (e.g., at x=1)
        g_data = np.nan_to_num(g_data, nan=0.0)
        g_model = np.nan_to_num(g_model, nan=0.0)
        delta = g_data - g_model
        return -0.5 * (delta @ self.icov @ delta)

    def ln_posterior(self, Acb):
        lp = self.ln_prior(Acb)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.ln_likelihood(Acb)
        return lp + ll

    def sampler(self, nwalkers=32, nsamples=1000, rerun=False):
        fname = os.path.join(
            self.libdir,
            f'samples_hl_nw{nwalkers}_ns{nsamples}_'
            f'lmin{self.lmin}_lmax{self.lmax}_bn{self.binner.n}_m{self.binner.method}.h5'
        )
        backend = emcee.backends.HDFBackend(fname)
        if os.path.isfile(fname):
            if rerun:
                backend.reset(nwalkers, 1)
            else:
                return backend
        pos = np.array([3.5e-6]) * (1 + 0.1 * np.random.randn(nwalkers, 1))
        sampler = emcee.EnsembleSampler(nwalkers, 1, self.ln_posterior, backend=backend)
        sampler.run_mcmc(pos, nsamples, progress=True)
        return sampler

    def samples(self, nwalkers=32, nsamples=1000, discard=200, getdist=False, rerun=False):
        sampler = self.sampler(nwalkers=nwalkers, nsamples=nsamples, rerun=rerun)
        samples = sampler.get_chain(discard=discard, thin=15, flat=True)
        samples = samples[samples > 0.0]
        if getdist:
            gdsamples = MCSamples(samples=samples, names=['acb'], labels=['acb'])
            return gdsamples
        return samples