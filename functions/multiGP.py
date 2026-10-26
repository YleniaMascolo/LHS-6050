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
    E = - (0.5 * (tau * tau) / (lam * lam)) - (0.5 * (sin_u * sin_u) / (w * w))

    k = (amp * amp) * np.exp(E)

    # dE/dτ
    dE_dtau = -(tau / (lam * lam)) - (np.pi / (2.0 * P * w * w)) * sin2_u

    # d2E/dτ2
    d2E_dtau2 = -(1.0 / (lam * lam)) - (np.pi**2 / (P * P * w * w)) * cos2_u

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



def compute_mgic_rv(
    t_list: List[NDArray[np.floating]],
    r_list: List[NDArray[np.floating]],
    err_list: List[NDArray[np.floating]],
    A: NDArray[np.floating],
    B: NDArray[np.floating],
    hp: QPHyperParams,
    jitter: NDArray[np.floating],
    k_pars: int,
    add_diag_eps: float = 1e-12
) -> float:
    """
    Compute the Modified Gaussian Process Information Criterion for Radial Velocities (MGPIC_rv)
    as described in Barragán (2026).

    This criterion assesses the quality of the RV fit conditioned on the information
    extracted from the stellar activity indicators, penalizing for the effective 
    degrees of freedom of the GP over the RVs.

    Assumes that the FIRST element of the input lists (index 0) always corresponds 
    to the Radial Velocity (RV) dataset.

    Parameters
    ----------
    t_list : list of NDArray
        Time arrays for each observable, starting with RVs at index 0.
    r_list : list of NDArray
        Residual arrays (data minus deterministic model) for each observable.
    err_list : list of NDArray
        Measurement uncertainties for each observable.
    A : NDArray
        Linear coefficients mapping the latent GP into each observable (shape (M,)).
    B : NDArray
        Linear coefficients mapping the derivative of the latent GP into each observable (shape (M,)).
    hp : QPHyperParams
        Hyperparameters of the quasi-periodic kernel.
    jitter : NDArray
        Instrumental/white noise jitter for each observable (shape (M,)).
    k_pars : int
        Number of free physical/deterministic parameters in the global model 
        (e.g., planet parameters, offsets, instrument jitters, GP hyperparameters).
    add_diag_eps : float, optional
        Small diagonal term added for numerical stability (default is 1e-12).

    Returns
    -------
    float
        The MGPIC_rv value. Returns -inf if any Cholesky factorization fails.
    """
    M = len(t_list)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    jitter = np.asarray(jitter, dtype=float)

    # Build the big covariance matrix K_big including measurement noise and jitter
    K_big, slices = _build_big_K(t_list, err_list, A, B, hp, jitter=jitter, add_diag_eps=add_diag_eps)
    N = K_big.shape[0]

    # Build the clean covariance matrix k_big (zero noise) to compute the smoothing matrix
    zero_errs = [np.zeros_like(err) for err in err_list]
    k_big, _ = _build_big_K(t_list, zero_errs, A, B, hp, jitter=np.zeros(M), add_diag_eps=0.0)

    # Invert K_big using Cholesky factorization
    try:
        c_fac = cho_factor(K_big, lower=True, check_finite=False)
        K_big_inv = cho_solve(c_fac, np.eye(N), check_finite=False)
    except np.linalg.LinAlgError:
        return -np.inf  # Return -inf if the matrix is not positive-definite

    # Create boolean masks to isolate RVs (at index 0) from Activity Indicators (AI)
    sl_rv = slices[0]
    is_rv = np.zeros(N, dtype=bool)
    is_rv[sl_rv] = True
    is_ai = ~is_rv

    # Isolate RV residuals and concatenate all AI residuals
    res_rv = np.asarray(r_list[0], dtype=float)
    res_ai = np.concatenate([np.asarray(r, dtype=float) for r in r_list[1:]]) if M > 1 else np.array([])

    # Extract the conditional covariance and compute conditional residuals for RVs
    K_rvrv_prime_inv = K_big_inv[is_rv][:, is_rv]
    
    try:
        c_fac_rv_prime = cho_factor(K_rvrv_prime_inv, lower=True, check_finite=False)
        # log_det of K_rvrv_prime is equivalent to -log_det of its inverse
        log_det_K_rvrv_prime = - 2.0 * np.sum(np.log(np.diag(c_fac_rv_prime[0])))
    except np.linalg.LinAlgError:
        return -np.inf

    if M > 1:
        K_rvai = K_big[is_rv][:, is_ai]
        K_aiai = K_big[is_ai][:, is_ai]
        try:
            c_fac_ai = cho_factor(K_aiai, lower=True, check_finite=False)
            res_conditional = res_rv - K_rvai @ cho_solve(c_fac_ai, res_ai, check_finite=False)
        except np.linalg.LinAlgError:
            return -np.inf
    else:
        res_conditional = res_rv

    # Compute the conditional log-likelihood of the RVs
    lrv = -0.5 * (
        (res_conditional.T @ (K_rvrv_prime_inv @ res_conditional)) 
        + log_det_K_rvrv_prime 
        + len(res_rv) * np.log(2.0 * np.pi)
    )

    # Compute the trace of the RV-specific smoothing matrix (effective GP degrees of freedom)
    S_big = k_big @ K_big_inv
    S_rv_diag = np.diag(S_big)[is_rv]
    trace_S_rv = np.sum(S_rv_diag)

    # Compute final MGPIC_rv
    k_rv = k_pars + trace_S_rv
    MGPIC_rv = -2.0 * lrv + 2.0 * k_rv

    return MGPIC_rv



