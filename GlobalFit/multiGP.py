import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from numpy.typing import NDArray
from scipy.linalg import cho_factor, cho_solve


@dataclass(frozen=True)
class QPHyperParams:
    """
    Quasi-periodic kernel hyperparameters.

    Kernel form used here:
        k(τ) = amp^2 * exp( -0.5*(τ^2/lam^2) - 2*sin^2(pi*τ/P)/w^2 )

    Parameters
    ----------
    amp : float
        Amplitude of the latent GP (same units as latent G).
    lam : float
        Exponential decay / evolution timescale.
    P : float
        Period (often rotation period).
    w : float
        Controls the smoothness/complexity within a period.
        Smaller w -> more structure within a period.
    """
    amp: float
    lam: float
    P: float
    w: float


def _qp_kernel_and_derivs_tau(
    tau: NDArray[np.floating],
    hp: QPHyperParams,
) -> Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """
    Compute k(τ), dk/dτ, d2k/dτ2 for the quasi-periodic kernel.

    Returns
    -------
    k : array
    dk_dtau : array
    d2k_dtau2 : array
    """
    amp, lam, P, w = hp.amp, hp.lam, hp.P, hp.w

    u = np.pi * tau / P
    sin_u = np.sin(u)
    sin2_u = np.sin(2.0 * u)
    cos2_u = np.cos(2.0 * u)

    # Exponent E(τ)
    E = -0.5 * (tau * tau) / (lam * lam) - (2.0 * sin_u * sin_u) / (w * w)

    k = (amp * amp) * np.exp(E)

    # dE/dτ
    dE_dtau = -(tau / (lam * lam)) - (2.0 * np.pi / (P * w * w)) * sin2_u

    # d2E/dτ2
    d2E_dtau2 = -(1.0 / (lam * lam)) - (4.0 * np.pi**2 / (P * P * w * w)) * cos2_u

    dk_dtau = k * dE_dtau
    d2k_dtau2 = k * (dE_dtau * dE_dtau + d2E_dtau2)

    return k, dk_dtau, d2k_dtau2


def lnlike_multidim_qp_gp(
    t_list: List[NDArray[np.floating]],
    r_list: List[NDArray[np.floating]],
    err_list: List[NDArray[np.floating]],
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    hp: QPHyperParams,
    jitter: Optional[NDArray[np.floating]] = None,
    add_diag_eps: float = 1e-12,
    return_cholesky: bool = False,
) -> Union[float,Tuple[float, Tuple[NDArray[np.floating], bool]]]:
    """
    Log-likelihood for the multi-output GP used in "latent GP + derivative" frameworks.

    Inputs are lists per observable (e.g. RV, BIS, FWHM, S-index):
      - t_list[m]: times for observable m
      - r_list[m]: residuals (data - deterministic model) for observable m
      - err_list[m]: measurement uncertainties for observable m

    The multi-output GP model is:
        y_m(t) = A_m * G(t) + B_m * dG/dt

    Covariances derive from k, dk/dτ, d2k/dτ2 with τ = t - t'.

    Parameters
    ----------
    A, B : arrays of shape (M,)
        Linear coefficients mapping latent process and its derivative into each observable.
    jitter : array of shape (M,), optional
        Extra white noise (added in quadrature) per observable.
    add_diag_eps : float
        Small diagonal term for numerical stability.
    return_cholesky : bool
        If True, returns (lnlike, (c_factor, lower)) where c_factor is from scipy cho_factor.

    Returns
    -------
    lnlike : float
        Gaussian log marginal likelihood.
    """
    M = len(t_list)
    if not (len(r_list) == len(err_list) == M):
        raise ValueError("t_list, r_list, err_list must have the same length (= number of observables).")

    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != (M,) or B.shape != (M,):
        raise ValueError(f"A and B must have shape (M,) with M={M}.")

    if jitter is None:
        jitter = np.zeros(M, dtype=float)
    else:
        jitter = np.asarray(jitter, dtype=float)
        if jitter.shape != (M,):
            raise ValueError(f"jitter must have shape (M,) with M={M}.")

    # Concatenate all observations into one big vector
    n_list = [len(t) for t in t_list]
    N = int(np.sum(n_list))
    y = np.concatenate([np.asarray(r, dtype=float) for r in r_list])
    if y.shape != (N,):
        raise ValueError("Residual vectors did not concatenate to 1-D length N.")

    # Build big covariance matrix K with block structure
    K = np.zeros((N, N), dtype=float)

    # Indices for blocks
    starts = np.cumsum([0] + n_list[:-1])
    slices = [slice(starts[m], starts[m] + n_list[m]) for m in range(M)]

    for m in range(M):
        tm = np.asarray(t_list[m], dtype=float)
        sm = slices[m]
        for n in range(M):
            tn = np.asarray(t_list[n], dtype=float)
            sn = slices[n]

            # pairwise τ = t_m - t_n
            tau = tm[:, None] - tn[None, :]

            k, dk_dtau, d2k_dtau2 = _qp_kernel_and_derivs_tau(tau, hp)

            # Using:
            # Cov[G(t), G(t')] = k
            # Cov[dG/dt, G(t')] = dk/dτ
            # Cov[G(t), dG/dt'] = -dk/dτ
            # Cov[dG/dt, dG/dt'] = -d2k/dτ2
            K_mn = (
                (A[m] * A[n]) * k
                + (B[m] * A[n] - A[m] * B[n]) * dk_dtau
                - (B[m] * B[n]) * d2k_dtau2
            )

            K[sm, sn] = K_mn

    # Add white noise to diagonal (measurement err + jitter per observable)
    diag = np.zeros(N, dtype=float)
    for m in range(M):
        sm = slices[m]
        em = np.asarray(err_list[m], dtype=float)
        if em.shape != (n_list[m],):
            raise ValueError(f"err_list[{m}] must have shape ({n_list[m]},)")
        diag[sm] = em * em + jitter[m] * jitter[m]

    K[np.diag_indices(N)] += diag + add_diag_eps

    # Cholesky + loglike
    try:
        c_fac = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError as e:
        # Not PD -> lnlike is effectively -inf
        if return_cholesky:
            return (-np.inf, (None, True))  # type: ignore
        return -np.inf

    alpha = cho_solve(c_fac, y, check_finite=False)
    logdet = 2.0 * np.sum(np.log(np.diag(c_fac[0])))
    lnlike = -0.5 * (y @ alpha + logdet + N * np.log(2.0 * np.pi))

    if return_cholesky:
        return lnlike, c_fac
    return lnlike


import numpy as np
from scipy.linalg import cho_factor, cho_solve

def _build_big_K(
    t_list, err_list, A, B, hp, jitter=None, add_diag_eps=1e-12
):
    """Build K_yy (training covariance) and concatenated y-index slices."""
    M = len(t_list)
    if jitter is None:
        jitter = np.zeros(M)
    n_list = [len(t) for t in t_list]
    N = int(np.sum(n_list))
    K = np.zeros((N, N), float)

    starts = np.cumsum([0] + n_list[:-1])
    slices = [slice(starts[m], starts[m] + n_list[m]) for m in range(M)]

    for m in range(M):
        tm = np.asarray(t_list[m], float)
        sm = slices[m]
        for n in range(M):
            tn = np.asarray(t_list[n], float)
            sn = slices[n]
            tau = tm[:, None] - tn[None, :]

            k, dk_dtau, d2k_dtau2 = _qp_kernel_and_derivs_tau(tau, hp)

            K_mn = (
                (A[m] * A[n]) * k
                + (B[m] * A[n] - A[m] * B[n]) * dk_dtau
                - (B[m] * B[n]) * d2k_dtau2
            )
            K[sm, sn] = K_mn

    # add white noise (measurement + jitter) on diagonal
    diag = np.zeros(N, float)
    for m in range(M):
        sm = slices[m]
        em = np.asarray(err_list[m], float)
        diag[sm] = em * em + jitter[m] * jitter[m]

    K[np.diag_indices(N)] += diag + add_diag_eps
    return K, slices


def _build_K_star_y(
    t_star_list, t_list, A, B, hp
):
    """
    Build cross-cov K_*y between test points (grouped by observable)
    and training points (grouped by observable).
    """
    M = len(t_list)
    n_star_list = [len(t) for t in t_star_list]
    n_list = [len(t) for t in t_list]
    Nstar = int(np.sum(n_star_list))
    N = int(np.sum(n_list))

    Ksy = np.zeros((Nstar, N), float)

    # slices for star and train
    s_starts = np.cumsum([0] + n_star_list[:-1])
    y_starts = np.cumsum([0] + n_list[:-1])
    s_slices = [slice(s_starts[m], s_starts[m] + n_star_list[m]) for m in range(M)]
    y_slices = [slice(y_starts[m], y_starts[m] + n_list[m]) for m in range(M)]

    for m in range(M):  # star observable
        ts = np.asarray(t_star_list[m], float)
        sm = s_slices[m]
        for n in range(M):  # train observable
            ty = np.asarray(t_list[n], float)
            sn = y_slices[n]
            tau = ts[:, None] - ty[None, :]

            k, dk_dtau, d2k_dtau2 = _qp_kernel_and_derivs_tau(tau, hp)

            K_mn = (
                (A[m] * A[n]) * k
                + (B[m] * A[n] - A[m] * B[n]) * dk_dtau
                - (B[m] * B[n]) * d2k_dtau2
            )
            Ksy[sm, sn] = K_mn

    return Ksy, s_slices


def _build_K_star_star(
    t_star_list, A, B, hp, add_diag_eps=0.0
):
    """Build K_** among test points (no measurement noise unless you add it separately)."""
    M = len(t_star_list)
    n_star_list = [len(t) for t in t_star_list]
    Nstar = int(np.sum(n_star_list))
    Kss = np.zeros((Nstar, Nstar), float)

    s_starts = np.cumsum([0] + n_star_list[:-1])
    s_slices = [slice(s_starts[m], s_starts[m] + n_star_list[m]) for m in range(M)]

    for m in range(M):
        tm = np.asarray(t_star_list[m], float)
        sm = s_slices[m]
        for n in range(M):
            tn = np.asarray(t_star_list[n], float)
            sn = s_slices[n]
            tau = tm[:, None] - tn[None, :]

            k, dk_dtau, d2k_dtau2 = _qp_kernel_and_derivs_tau(tau, hp)
            K_mn = (
                (A[m] * A[n]) * k
                + (B[m] * A[n] - A[m] * B[n]) * dk_dtau
                - (B[m] * B[n]) * d2k_dtau2
            )
            Kss[sm, sn] = K_mn

    if add_diag_eps:
        Kss[np.diag_indices(Nstar)] += add_diag_eps
    return Kss, s_slices


def gp_predict_multidim(
    t_list, r_list, err_list,
    t_star_list,
    A, B, hp,
    jitter=None,
    add_diag_eps=1e-12,
    include_obs_noise_in_pred=False
):
    """
    Predict GP component at t_star_list for each observable.

    Returns
    -------
    mu_list : list of arrays, mean GP at each star time for each observable
    sig_list: list of arrays, 1-sigma GP uncertainty at each star time for each observable
    """
    M = len(t_list)
    if jitter is None:
        jitter = np.zeros(M)

    # concat training residuals y
    y = np.concatenate([np.asarray(r, float) for r in r_list])

    # build Kyy and Cholesky
    Kyy, _ = _build_big_K(t_list, err_list, A, B, hp, jitter=jitter, add_diag_eps=add_diag_eps)
    c_fac = cho_factor(Kyy, lower=True, check_finite=False)

    # build cross-cov and test cov
    Ksy, s_slices = _build_K_star_y(t_star_list, t_list, A, B, hp)
    Kss, _ = _build_K_star_star(t_star_list, A, B, hp)

    # posterior mean
    alpha = cho_solve(c_fac, y, check_finite=False)
    mu_star = Ksy @ alpha  # shape (Nstar,)

    # posterior covariance diagonal (for error bands)
    # v = Kyy^{-1} Kys^T  (solve Kyy v = Kys^T)
    v = cho_solve(c_fac, Ksy.T, check_finite=False)  # shape (N, Nstar)
    cov_star = Kss - (Ksy @ v)  # shape (Nstar, Nstar)
    var_star = np.clip(np.diag(cov_star), 0.0, np.inf)

    if include_obs_noise_in_pred:
        # add per-observable jitter+meas noise at test points if you want predictive distribution of observed points
        # (often you *don't* want this when plotting the latent GP component)
        add = np.zeros_like(var_star)
        # NOTE: we don't know err at test points unless you supply it; this only adds jitter term.
        # If you want measurement noise too, pass err_star_list and add it similarly.
        for m in range(M):
            add[s_slices[m]] += jitter[m] * jitter[m]
        var_star = var_star + add

    # split back into per-observable arrays
    mu_list = [mu_star[s_slices[m]] for m in range(M)]
    sig_list = [np.sqrt(var_star[s_slices[m]]) for m in range(M)]
    return mu_list, sig_list



if __name__ == "__main__":


    import pandas
    df = pandas.read_csv("LHS6050.rv.csv")
    
    fwhm = df['fwhm']-np.nanmean(df['fwhm'])
    rv = df['selfrv']-np.nanmean(df['selfrv'])

    

    hp = QPHyperParams(amp=3.0, lam=40.0, P=6.6, w=0.5)
    A = np.array([10.0, 0.005])      # e.g. RV and BIS scale with G
    B = np.array([5.0, 0.0])      # e.g. RV also scales with dG/dt, BIS not
    jitter = np.array([5.0, 0.005]) # per observable

    ll = lnlike_multidim_qp_gp(
        t_list=[df['bjd'], df['bjd']],
        r_list=[rv,fwhm],
        err_list=[df['selfrv_err'], df['fwhm_err']],
        A=A, B=B, hp=hp, jitter=jitter
    )

    print(ll)









    import matplotlib.pyplot as plt

    # Example with 1 observable (RV). For multiple, put them in lists.
    t_list   = [df['bjd']]
    r_list   = [rv]     # residuals
    err_list = [df['selfrv_err']]
    t_rv = df['bjd']

    # Predict on a dense grid
    tgrid = np.linspace(t_rv.min(), t_rv.max(), 2000)
    t_star_list = [tgrid]

    mu_list, sig_list = gp_predict_multidim(
        t_list, r_list, err_list,
        t_star_list=t_star_list,
        A=np.array([A[0]]),
        B=np.array([B[0]]),
        hp=hp,
        jitter=np.array([jitter[0]]),
    )

    mu = mu_list[0]
    sig = sig_list[0]

    plt.figure()
    plt.errorbar(t_rv, rv, yerr=df['selfrv_err'], fmt='.', label='data')
    plt.plot(tgrid, mu, label='det + GP mean')
    #plt.fill_between(tgrid,  mu - sig, det_model_on_grid + mu + sig, alpha=0.2, label='±1σ (GP)')
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("RV")
    plt.show()
