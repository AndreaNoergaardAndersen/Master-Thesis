# welfare_calc.py
# =================================================================
# Truncated-horizon consumption-equivalent welfare calculations for
# the I-HANK model, plus weighted distributional statistics.
#
# All quantities are computed POST-MODEL: we read the transition
# paths stored in model.path and the steady-state objects in
# model.ss, and never modify the model itself. GEModelTools is not
# touched.
#
# Notation matches household_problem.py:
#   - i_fix indexes the sector (0=HH, 1=HL, 2=LH, 3=LL, 4=NT)
#   - i_z   indexes the productivity state
#   - i_a   indexes the asset grid point
#   - u(c) = c^(1-sigma)/(1-sigma)  (sigma != 1 branch is used since
#                                    sigma=1.5 in the calibration)
#   - psi_avg = varphi_avg * (N_j/s_j)^(1+nu) / (1+nu)
#     i.e. the population-average labor disutility variant
#
# The truncated-H welfare object is
#   W^H_0(i,z,a) = E_0 sum_{t=0}^{H-1} beta^t (u(c_t) - psi_avg_t)
# computed by backward recursion on the stored transition path.
# The truncated CEV (constant flow consumption ce_bar over the same
# H periods that delivers the same W) is recovered from
#   W^H_0 = u(ce_bar) * (1 - beta^H)/(1 - beta)
# which mirrors the Lucas (1987) / household_problem.py convention,
# only with the finite-horizon annuity factor in place of 1/(1-beta).
# =================================================================
 
import numpy as np
from consav.linear_interp import interp_1d_vec
 
 
# -----------------------------------------------------------------
# Disutility helper
# -----------------------------------------------------------------
 
def psi_path(par, NHH, NHL, NLH, NLL, NNT, variant='avg'):
    """
    Build the per-period labor disutility for each sector along an
    input path, under the requested utility variant.
 
    Variants
    --------
    'nodis' : returns zeros — welfare measure ignores labor disutility,
              matches household_problem.py's v_nodis.
    'avg'   : varphi_avg (population mean) applied to all sectors,
              matches household_problem.py's v_avg.
    'sec'   : sector-specific varphi_<j>, matches v_sec.
 
    Inputs
    ------
    par : model.par
    NHH, NHL, NLH, NLL, NNT : (T,) arrays of sectoral employment
    variant : str in {'nodis', 'avg', 'sec'}
 
    Returns
    -------
    psi : (T, Nfix) array
        Per-period disutility flow for each sector along the path.
    """
    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL
    s = np.array([par.sHH, par.sHL, par.sLH, par.sLL, sNT])
    N = np.stack([NHH, NHL, NLH, NLL, NNT], axis=1)   # (T, 5)
    h = N / s[None, :]
    T = h.shape[0]
 
    if variant == 'nodis':
        return np.zeros((T, 5))
    if variant == 'avg':
        return par.varphi_avg * h**(1.0 + par.nu) / (1.0 + par.nu)
    if variant == 'sec':
        varphi_sec = np.array([par.varphiHH, par.varphiHL,
                               par.varphiLH, par.varphiLL,
                               par.varphiNT])
        return varphi_sec[None, :] * h**(1.0 + par.nu) / (1.0 + par.nu)
    raise ValueError(f"Unknown variant '{variant}'. "
                     f"Choose from 'nodis', 'avg', 'sec'.")
 
 
# Backward-compatible alias
def psi_avg_path(par, NHH, NHL, NLH, NLL, NNT):
    """Legacy alias for psi_path(..., variant='avg')."""
    return psi_path(par, NHH, NHL, NLH, NLL, NNT, variant='avg')
 
 
# -----------------------------------------------------------------
# Backward recursion for truncated welfare
# -----------------------------------------------------------------
 
def truncated_welfare(par, u_path, a_policy_path, psi_avg_path_arr,
                      z_trans, H):
    """
    Backward recursion that returns the beginning-of-period
    (pre-z-draw) truncated welfare W^H at t=0, on the household
    state grid (Nfix, Nz, Na).
 
    Inputs
    ------
    par              : model.par (only par.a_grid and par.Na are used)
    u_path           : (T, Nfix, Nz, Na) flow utility along the path
    a_policy_path    : (T, Nfix, Nz, Na) asset policy a'_t(i,z,a)
    psi_avg_path_arr : (T, Nfix) per-period sector-level disutility
    z_trans          : (Nfix, Nz, Nz) z-transition matrices
    H                : horizon length (here H=20)
 
    Returns
    -------
    Wbeg_0 : (Nfix, Nz, Na) array
        W^H_0(i, z, a) evaluated at the start of period 0, BEFORE
        the time-0 z-shock realises. This is the object consistent
        with weighting by ss.Dbeg.
 
    Recursion structure (mirrors household_problem.py timing)
    ---------------------------------------------------------
    Initialise:
        Wbeg_next(i, z, a) = 0          # terminal H = 0
 
    For t = H-1, H-2, ..., 0:
        # post-z value at time t
        W_t(i, z, a) = u_t(i,z,a) - psi_t(i)
                       + beta * Wbeg_next(i, z, a'_t(i,z,a))
        # integrate over z' (probability of next-period z given current z)
        Wbeg_t(i, z, a) = sum_z' z_trans(i, z, z') * W_t(i, z', a)
 
    Subtle point: the continuation Wbeg_next is interpolated at the
    next-period asset a'_t(i,z,a), exactly as in the household
    block's `interp_1d_vec(par.a_grid, vbeg_avg_plus[i_fix, i_z, :],
    a[i_fix, i_z, :], cont_avg)` call. We keep the same convention.
    """
    Nfix, Nz, Na = u_path.shape[1], u_path.shape[2], u_path.shape[3]
    beta = par.beta
 
    Wbeg_next = np.zeros((Nfix, Nz, Na))   # terminal condition
 
    # backward sweep
    for t in range(H - 1, -1, -1):
        W_t = np.empty((Nfix, Nz, Na))
        for i_fix in range(Nfix):
            # interpolate continuation at next-period asset a'_t
            cont = np.empty((Nz, Na))
            for i_z in range(Nz):
                interp_1d_vec(par.a_grid,
                              Wbeg_next[i_fix, i_z, :],
                              a_policy_path[t, i_fix, i_z, :],
                              cont[i_z, :])
 
            # Bellman step (post-z)
            W_t[i_fix] = (u_path[t, i_fix]
                          - psi_avg_path_arr[t, i_fix]
                          + beta * cont)
 
            # beginning-of-period expectation over z' | z
            Wbeg_next[i_fix] = z_trans[i_fix] @ W_t[i_fix]
 
    return Wbeg_next   # at this point, this is Wbeg at t=0
 
 
# -----------------------------------------------------------------
# Wrapper that runs the recursion for transition AND steady state
# -----------------------------------------------------------------
 
def compute_welfare(model, H=20, variant='avg'):
    """
    Run the truncated-H backward recursion under both the stored
    transition (model.path) and the steady state (model.ss broadcast
    to H periods). Return W^H_0 for each case and the corresponding
    household-level CEV ce_bar^H.
 
    Parameters
    ----------
    model : IHANKModelClass
    H : int   truncation horizon (default 20 = 5 years quarterly)
    variant : str in {'nodis', 'avg', 'sec'}
        Selects the labor-disutility convention used in the value
        function. 'avg' is the default (population-average disutility,
        matches model.path.v_avg). 'nodis' computes consumption-only
        welfare (matches v_nodis). 'sec' uses sector-specific
        varphi (matches v_sec).
 
    Returns
    -------
    out : dict with keys
        'W_trans'   : (Nfix, Nz, Na) W^H_0 under the transition
        'W_ss'      : (Nfix, Nz, Na) W^H_0 under SS prices/policies
        'ce_trans'  : (Nfix, Nz, Na) truncated CEV under transition
        'ce_ss'     : (Nfix, Nz, Na) truncated CEV at SS
        'loss'      : (Nfix, Nz, Na) percent CEV loss
        'annuity'   : scalar (1 - beta^H)/(1 - beta)
        'variant'   : echo of the variant used
    """
    par, ss, path = model.par, model.ss, model.path
    beta = par.beta
    sigma = par.sigma
 
    # ---- transition inputs ----
    psi_trans = psi_path(par,
                         path.NHH.flatten(), path.NHL.flatten(),
                         path.NLH.flatten(), path.NLL.flatten(),
                         path.NNT.flatten(), variant=variant)
 
    W_trans = truncated_welfare(par,
                                u_path=path.u,
                                a_policy_path=path.a,
                                psi_avg_path_arr=psi_trans,
                                z_trans=ss.z_trans,   # z_trans is constant
                                H=H)
 
    # ---- SS inputs (broadcast to H periods) ----
    # We replicate ss.u, ss.a, etc., across the first axis (time)
    # so the same recursion code applies. Result is the same W^H
    # we'd get from a household sitting at SS forever, evaluated at
    # the start of period 0.
    T_dummy = H
    NHH_ss = np.full(T_dummy, ss.NHH)
    NHL_ss = np.full(T_dummy, ss.NHL)
    NLH_ss = np.full(T_dummy, ss.NLH)
    NLL_ss = np.full(T_dummy, ss.NLL)
    NNT_ss = np.full(T_dummy, ss.NNT)
    psi_ss = psi_path(par, NHH_ss, NHL_ss, NLH_ss, NLL_ss, NNT_ss,
                      variant=variant)
 
    u_ss_path = np.broadcast_to(ss.u[None, ...],
                                (T_dummy,) + ss.u.shape).copy()
    a_ss_path = np.broadcast_to(ss.a[None, ...],
                                (T_dummy,) + ss.a.shape).copy()
 
    W_ss = truncated_welfare(par,
                             u_path=u_ss_path,
                             a_policy_path=a_ss_path,
                             psi_avg_path_arr=psi_ss,
                             z_trans=ss.z_trans,
                             H=H)
 
    # ---- truncated CEV inversion ----
    # W = u(ce_bar) * (1 - beta^H)/(1 - beta)
    # => ce_bar = [W * (1 - beta)/(1 - beta^H) * (1 - sigma)]^(1/(1-sigma))
    # for sigma != 1. With sigma = 1.5 (CRRA), u is negative, and
    # the inside expression is positive because (1 - sigma) < 0
    # multiplies a typically negative W. Mirror household_problem.py
    # ce_from_value safety floor of 1e-300.
    annuity = (1.0 - beta**H) / (1.0 - beta)
 
    def W_to_ce(W):
        if np.abs(sigma - 1.0) < 1e-10:
            # log utility
            return np.exp(W / annuity)
        inside = W / annuity * (1.0 - sigma)
        inside = np.maximum(inside, 1e-300)
        return inside ** (1.0 / (1.0 - sigma))
 
    ce_trans = W_to_ce(W_trans)
    ce_ss    = W_to_ce(W_ss)
    loss     = 1.0 - ce_trans / ce_ss
 
    return {'W_trans': W_trans, 'W_ss': W_ss,
            'ce_trans': ce_trans, 'ce_ss': ce_ss,
            'loss': loss, 'annuity': annuity,
            'variant': variant}
 
 
# -----------------------------------------------------------------
# Weighted distributional statistics
# -----------------------------------------------------------------
 
def weighted_quantiles(values, weights, qs):
    """
    Quantiles of `values` weighted by `weights`.
 
    Used for the percentile-loss bar chart (figure 9b).
 
    Inputs
    ------
    values  : 1-D array of values (e.g., CEV losses)
    weights : 1-D array of same length (e.g., ss.Dbeg flattened)
    qs      : iterable of quantile levels in (0,1)
 
    Returns
    -------
    out : array of weighted quantiles, one per q in qs
 
    Method
    ------
    Sort by value, accumulate weights, locate the smallest value at
    which the cumulative weight (normalised to 1) exceeds q. Standard
    discrete weighted-quantile definition; identical to numpy's
    quantile with method='inverted_cdf' on the empirical CDF.
    """
    v = np.asarray(values).ravel()
    w = np.asarray(weights).ravel()
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w) / np.sum(w)
    return np.array([v[np.searchsorted(cw, q)] for q in qs])
 
 
def weighted_mean(values, weights):
    """Population-weighted mean. Used for the dashed-line average."""
    v = np.asarray(values).ravel()
    w = np.asarray(weights).ravel()
    return (v * w).sum() / w.sum()
 
 
def lorenz_and_gini(values, weights):
    """
    Lorenz curve and Gini coefficient for `values` weighted by
    `weights`. Used for figures 9c (L1) and the L2 variant.
 
    Inputs
    ------
    values  : 1-D array (must be non-negative for a textbook Lorenz)
    weights : 1-D array of same length
 
    Returns
    -------
    pop_cum   : (N+1,) cumulative population share, starting at 0
    val_cum   : (N+1,) cumulative value share,      starting at 0
    gini      : scalar Gini coefficient
 
    Method
    ------
    Sort by value ascending, build cumulative population share on
    the x-axis and cumulative value share on the y-axis. The Gini
    is twice the area between the 45-degree line and the Lorenz
    curve, computed by the trapezoidal rule. This is the standard
    discrete Lorenz formula for weighted observations (Lambert,
    2001, "The Distribution and Redistribution of Income", Ch. 2).
 
    Note on negative values: if `values` contains negatives (e.g.,
    when applied to CEV LOSSES rather than levels), the Lorenz
    curve is still well-defined but its economic interpretation
    differs and the Gini can fall outside [0, 1]. We allow this so
    the function can compute the L2 "Lorenz of losses". The caller
    should be aware.
    """
    v = np.asarray(values).ravel().astype(float)
    w = np.asarray(weights).ravel().astype(float)
    order = np.argsort(v)
    v, w = v[order], w[order]
    pop_cum = np.concatenate(([0.0], np.cumsum(w) / w.sum()))
    val_cum = np.concatenate(([0.0], np.cumsum(v * w) / (v * w).sum()))
    # Gini via trapezoidal area under the Lorenz curve
    # np.trapezoid replaced np.trapz in newer numpy
    area_under = (np.trapezoid(val_cum, pop_cum)
                  if hasattr(np, 'trapezoid')
                  else np.trapz(val_cum, pop_cum))
    gini = 1.0 - 2.0 * area_under
    return pop_cum, val_cum, gini
 
 
def wealth_group_masks(par, ss, cutoffs_in_inc_units=(1.0, 4.0)):
    """
    Build boolean masks over the (Nfix, Nz, Na) household state grid
    for four wealth groups. Used by figures 11 and 12.
 
    Groups (income-scaled, anchored to mean post-tax labor income):
        HtM  : a == 0
        Low  : 0 < a <= c1 * inc_avg
        Mid  : c1 * inc_avg < a <= c2 * inc_avg
        High : a > c2 * inc_avg
    where (c1, c2) defaults to (1, 4) quarters of mean labor income.
 
    Returns
    -------
    masks : dict[label -> (Nfix, Nz, Na) bool ndarray]
        Insertion order matches the canonical group ordering.
    inc_avg : float
        Mean post-tax labor income per household (= sum of ss.inc_j
        since population mass = 1). Reported for transparency.
    """
    c1, c2 = cutoffs_in_inc_units
    inc_avg = (ss.inc_HH + ss.inc_HL + ss.inc_LH + ss.inc_LL + ss.inc_NT)
 
    a_grid = par.a_grid                                  # (Na,)
    a3d = a_grid[None, None, :]                          # broadcast
 
    masks = {
        'HtM' : np.broadcast_to(a3d == 0.0,             (par.Nfix, par.Nz, par.Na)),
        'Low' : np.broadcast_to((a3d > 0) & (a3d <= c1*inc_avg), (par.Nfix, par.Nz, par.Na)),
        'Mid' : np.broadcast_to((a3d > c1*inc_avg) & (a3d <= c2*inc_avg), (par.Nfix, par.Nz, par.Na)),
        'High': np.broadcast_to(a3d > c2*inc_avg,        (par.Nfix, par.Nz, par.Na)),
    }
    return masks, inc_avg
 
 
def population_cev_loss(W_trans, W_ss, Dbeg, beta, sigma, H):
    """
    Population-aggregate truncated CEV loss. This is the SCALAR
    headline number that should be annotated above the inequality
    main figure and above the aggregate-welfare main figure.
 
    The aggregation order matters:
        bar_W^pop_trans = sum Dbeg * W_trans
        bar_W^pop_ss    = sum Dbeg * W_ss
        ce^pop = invert each, then take 1 - ce_trans/ce_ss
 
    This is NOT the same as the average of the per-state losses,
    because the CEV inversion is nonlinear. The convention here
    (aggregate W, then invert) is the utilitarian-welfare convention
    of Lucas (1987) and matches what an aggregate welfare statement
    requires. Bhandari-Evans-Golosov-Sargent (2023) call this the
    "social welfare CEV" and contrast it with the average of
    individual CEVs (which is what your percentile bars show).
    """
    annuity = (1.0 - beta**H) / (1.0 - beta)
    W_pop_trans = (Dbeg * W_trans).sum()
    W_pop_ss    = (Dbeg * W_ss).sum()
 
    def W_to_ce(W):
        if np.abs(sigma - 1.0) < 1e-10:
            return np.exp(W / annuity)
        inside = W / annuity * (1.0 - sigma)
        inside = max(inside, 1e-300)
        return inside ** (1.0 / (1.0 - sigma))
 
    return 1.0 - W_to_ce(W_pop_trans) / W_to_ce(W_pop_ss)
 