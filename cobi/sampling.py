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


class Likelihood:
    def __init__(
        self, qe, binner, lmin=2, lmax=1024,
        rdn0: bool = False, fid: float = 1e-6, use_fid: bool = True,
        null: bool = False,
    ):
        basedir = os.path.join(qe.basedir, "likelihood")
        os.makedirs(basedir, exist_ok=True)

        self.binner = binner
        self.sel = (binner.b >= lmin) & (binner.b <= lmax)

        lhdir = (
            f"bn{binner.n}_bm{binner.method}_lmin{lmin}_lmax{lmax}_"
            f"{'rdn0' if rdn0 else 'mcn0'}"
            f"{'_null' if null else ''}"
        )
        self.basedir = os.path.join(basedir, lhdir)
        os.makedirs(self.basedir, exist_ok=True)

        self.lmax = int(qe.recon_lmax)
        self.rdn0 = bool(rdn0)
        self.use_fid = bool(use_fid)
        self.fid = float(fid)
        self.null = bool(null)

        # physically meaningful priors
        self.Amax = 1e-5

        # precompute theory shape in ell-space
        l = np.arange(self.lmax + 1, dtype=float)
        self._th_shape = 2.0 * np.pi / (l**2 + l + 1e-30)

        # "without bias" raw QE bandpowers (nsims, ell)
        qcl_wb = qe.qcl_stat(n0=None, mf=False, n1=False, nlens=False, binned=False)

        # "with all bias corrections" output directly from qe.qcl_stat
        if rdn0:
            qcl_wob = qe.qcl_stat(n0="rdn0", mf=True, n1=True, nlens=True, binned=False)
        else:
            qcl_wob = qe.qcl_stat(n0="mcn0", mf=True, n1=True, nlens=True, binned=False)

        # keep bias curve (ell-space) for theory-with-bias
        n0 = qe.MCN0("stat")
        n1 = qe.N1()
        nlens = qe.Nlens()
        mf = qe.mean_field_cl()
        self.bias = n0 + n1 + nlens + mf

        # precompute binned bias in selected bins
        self.bias_binned_sel = self.binner.bin(self.bias[: self.lmax + 1])[self.sel]

        # ---- helpers to build theory vectors in selected bins ----
        def _theory_sel(A):
            return self.binner.bin(float(A) * self._th_shape)[self.sel]

        def _theorywbias_sel(A):
            return _theory_sel(A) + self.bias_binned_sel

        # bin simulations (restrict to selected bins immediately)
        qcl_wb_binned_sel = binner.bin(qcl_wb)[:, self.sel]
        qcl_wob_binned_sel = binner.bin(qcl_wob)[:, self.sel]

        # choose "observed" vector (default: mean for pipeline tests)
        cl_obs_wb = qcl_wb_binned_sel.mean(axis=0)
        cl_obs_wob = qcl_wob_binned_sel.mean(axis=0)

        if self.null:
            # subtract fiducial theory from BOTH sims and obs before mean/cov
            thfid_wb = _theorywbias_sel(self.fid)
            thfid_wob = _theory_sel(self.fid)

            d_wb = qcl_wb_binned_sel - thfid_wb[None, :]
            d_wob = qcl_wob_binned_sel - thfid_wob[None, :]

            self.cl_obs_wb = cl_obs_wb - thfid_wb
            self.cl_obs_wob = cl_obs_wob - thfid_wob

            # mean/cov in residual space
            self.cl_fid_wb = d_wb.mean(axis=0)
            self.cl_fid_wob = d_wob.mean(axis=0)

            cov_wb = np.cov(d_wb.T)
            cov_wob = np.cov(d_wob.T)

            # store fid vectors for fast residual theory evaluation
            self._thfid_wb = thfid_wb
            self._thfid_wob = thfid_wob
        else:
            # standard (non-null) pipeline
            self.cl_obs_wb = cl_obs_wb.copy()
            self.cl_obs_wob = cl_obs_wob.copy()

            self.cl_fid_wb = qcl_wb_binned_sel.mean(axis=0)
            self.cl_fid_wob = qcl_wob_binned_sel.mean(axis=0)

            cov_wb = np.cov(qcl_wb_binned_sel.T)
            cov_wob = np.cov(qcl_wob_binned_sel.T)

            self._thfid_wb = None
            self._thfid_wob = None

        self.icov_wb = np.linalg.inv(cov_wb)
        self.icov_wob = np.linalg.inv(cov_wob)

    # --- theory vectors in selected bins ---
    def theory(self, Acb):
        return self.binner.bin(float(Acb) * self._th_shape)[self.sel]

    def theorywbias(self, Acb):
        th_binned_sel = self.binner.bin(float(Acb) * self._th_shape)[self.sel]
        return th_binned_sel + self.bias_binned_sel

    # --- residual theories used in null mode ---
    def _dtheory_wob(self, Acb):
        # theory(A) - theory(fid)
        return self.theory(Acb) - self._thfid_wob

    def _dtheory_wb(self, Acb):
        return self.theorywbias(Acb) - self._thfid_wb

    # -------- Gaussian bandpower likelihoods --------
    def log_likelihood_wb(self, theta):
        if self.null:
            dA = float(theta[0])
            Acb = self.fid + dA
            if (Acb < 0.0) or (Acb > self.Amax):
                return -np.inf
            cl_th = self._dtheory_wb(Acb)
        else:
            Acb = float(theta[0])
            if (Acb < 0.0) or (Acb > self.Amax):
                return -np.inf
            cl_th = self.theorywbias(Acb)

        r = self.cl_obs_wb - cl_th
        chi2 = r @ self.icov_wb @ r
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def log_likelihood_wob(self, theta):
        if self.null:
            dA = float(theta[0])
            Acb = self.fid + dA
            if (Acb < 0.0) or (Acb > self.Amax):
                return -np.inf
            cl_th = self._dtheory_wob(Acb)
        else:
            Acb = float(theta[0])
            if (Acb < 0.0) or (Acb > self.Amax):
                return -np.inf
            cl_th = self.theory(Acb)

        r = self.cl_obs_wob - cl_th
        chi2 = r @ self.icov_wob @ r
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def log_probability_wb(self, theta):
        return self.log_likelihood_wb(theta)

    def log_probability_wob(self, theta):
        return self.log_likelihood_wob(theta)

    def samples_wb(self, n_walkers, nsteps=2000, discard=200, thin=15):
        fname = os.path.join(self.basedir, f"samples_wb_{n_walkers}_{nsteps}.h5")
        if os.path.exists(fname):
            sampler = emcee.backends.HDFBackend(fname)
            return sampler.get_chain(discard=discard, thin=thin, flat=True)
        else:
            backend = emcee.backends.HDFBackend(fname)
            ndim = 1
            if self.null:
                # ΔA centered at 0
                scale = 0.1 * self.fid if self.fid != 0 else 1e-7
                initial = 0.0 + scale * np.random.randn(n_walkers, ndim)
            else:
                initial = np.abs(self.fid + 0.1 * self.fid * np.random.randn(n_walkers, ndim))

            sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_probability_wb, backend=backend)
            sampler.run_mcmc(initial, nsteps, progress=True)
            return sampler.get_chain(discard=discard, thin=thin, flat=True)

    def samples_wob(self, n_walkers, nsteps=2000, discard=200, thin=15):
        fname = os.path.join(self.basedir, f"samples_wob_{n_walkers}_{nsteps}.h5")
        if os.path.exists(fname):
            sampler = emcee.backends.HDFBackend(fname)
            return sampler.get_chain(discard=discard, thin=thin, flat=True)
        else:
            backend = emcee.backends.HDFBackend(fname)
            ndim = 1
            if self.null:
                scale = 0.1 * self.fid if self.fid != 0 else 1e-7
                initial = 0.0 + scale * np.random.randn(n_walkers, ndim)
            else:
                initial = np.abs(self.fid + 0.1 * self.fid * np.random.randn(n_walkers, ndim))

            sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_probability_wob, backend=backend)
            sampler.run_mcmc(initial, nsteps, progress=True)
            return sampler.get_chain(discard=discard, thin=thin, flat=True)