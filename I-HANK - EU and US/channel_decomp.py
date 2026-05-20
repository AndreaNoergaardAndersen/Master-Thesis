
# =================================================================
# IRF-version channel decomposition using the verified approach
# from welfare_decomp_calc.py.
#
# Channels: P | ra | tau | W | N
#
# For P, tau, W: uses scenario_inc_sub() from welfare_decomp_calc.py
#   (the same construction that produces the verified bar chart).
# For ra: uses scenario_ra_only() from welfare_decomp_calc.py.
# For N: extends scenario_inc_sub(sub='N') to ALSO activate the
#   labour-disutility effect of N (path.N_j passed through).
#   When target='ce_nodis', disutility is zero anyway so it makes
#   no difference; when target='ce_avg' or 'ce_sec', this correctly
#   includes the disutility offset shown in the bottom panel of the
#   bar chart.
#
# Outcome extraction (time series, not scalar):
#   C_hh         → aggregate path.c with SS Dbeg → (H,) array
#   ce_nodis/avg/sec → aggregate path.<target> with SS Dbeg → (H,)
#
# Aggregating with the FIXED SS distribution (ss.Dbeg) is the
# partial-equilibrium convention consistent with welfare_calc.py
# and the utilitarian social welfare function of Lucas (1987).
# =================================================================
 
import numpy as np
from welfare_decomp_calc import (
    _factor_path, INCOME_SECTORS, INCOME_SUB_CHANNELS,
    scenario_ra_only,
)
 
CHANNELS = ['P', 'ra', 'tau', 'W', 'N']
 
 
# ──────────────────────────────────────────────────────────────────
# Scenario builders
# ──────────────────────────────────────────────────────────────────
 
def _scenario_inc_sub_irf(model_baseline, sub):
    """
    Income sub-channel scenario (P, tau, W, or N-income-only).
    Mirrors welfare_decomp_calc.scenario_inc_sub exactly, but
    returns the solved model_cf instead of computing scalar welfare.
    ra and N_j (disutility) are held at SS.
    """
    model_cf = model_baseline.copy()
    ss, path = model_cf.ss, model_cf.path
 
    factor = _factor_path(model_baseline, sub)
 
    for j in INCOME_SECTORS:
        inc_ss_j = getattr(ss, f'inc_{j}')
        f_jt = factor[j] if isinstance(factor, dict) else factor
        arr  = getattr(path, f'inc_{j}')
        arr[:] = (inc_ss_j * f_jt).reshape(arr.shape)
 
    path.ra[:] = ss.ra
    for j in INCOME_SECTORS:
        arr = getattr(path, f'N{j}')
        arr[:] = getattr(ss, f'N{j}')
 
    model_cf.solve_hh_path(do_print=False)
    return model_cf
 
 
def _scenario_N_full(model_baseline):
    """
    Combined N channel: income effect of N AND labour-disutility
    effect of N. ra held at SS; all other income factors at SS.
 
    This is used when the target includes labour disutility
    (ce_avg, ce_sec). It captures the full N contribution shown
    as both the 'Employment' bar and the bottom-panel disutility
    bar in welfare_decomp_plots.plot_stacked_by_group.
    """
    model_cf = model_baseline.copy()
    ss, path = model_cf.ss, model_cf.path
 
    factor_N = _factor_path(model_baseline, 'N')   # dict by sector
 
    for j in INCOME_SECTORS:
        inc_ss_j = getattr(ss, f'inc_{j}')
        f_jt = factor_N[j]
        # income: activates N through inc_j
        inc_arr = getattr(path, f'inc_{j}')
        inc_arr[:] = (inc_ss_j * f_jt).reshape(inc_arr.shape)
        # disutility: pass actual N_j path
        N_arr = getattr(path, f'N{j}')
        # N_arr already holds path.N{j} from the baseline copy
 
    path.ra[:] = ss.ra
 
    model_cf.solve_hh_path(do_print=False)
    return model_cf
 
 
def _scenario_ra_irf(model_baseline):
    """
    Interest (ra) channel. Mirrors scenario_ra_only from
    welfare_decomp_calc.py but returns the solved model_cf.
    """
    model_cf = model_baseline.copy()
    ss, path = model_cf.ss, model_cf.path
 
    # hold all income inputs at SS
    for j in INCOME_SECTORS:
        getattr(path, f'inc_{j}')[:] = getattr(ss, f'inc_{j}')
        getattr(path, f'N{j}')[:] = getattr(ss, f'N{j}')
    # ra is already at transition path in the copy — leave it
 
    model_cf.solve_hh_path(do_print=False)
    return model_cf
 
 
# ──────────────────────────────────────────────────────────────────
# Outcome aggregation
# ──────────────────────────────────────────────────────────────────
 
def _agg(model_cf, target, H, Dbeg):
    """
    Aggregate HH output ``target`` weighted by the SS initial
    distribution Dbeg for periods 0…H-1. Returns a (H,) array.
    """
    key = 'c' if target == 'C_hh' else target
    out = getattr(model_cf.path, key)   # (T, Nfix, Nz, Na)
    return np.einsum('txza,xza->t', out[:H], Dbeg)
 
 
def _ss_level(model, target):
    ss  = model.ss
    key = 'c' if target == 'C_hh' else target
    return float(np.sum(ss.Dbeg * getattr(ss, key)))
 
 
# ──────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────
 
def run_channel_decomp(model, target='C_hh', H=20, do_print=False):
    """
    Five-channel partial-equilibrium IRF decomposition of ``target``.
 
    Uses the verified _factor_path / scenario_inc_sub approach from
    welfare_decomp_calc.py for P, tau, W, N channels. The ra channel
    mirrors scenario_ra_only.
 
    For target in {'ce_avg', 'ce_sec'}, the N channel includes both
    the income effect and the labour-disutility effect, matching the
    combined resource + preference-flow contribution visible in the
    stacked bar chart (welfare_decomp_plots.plot_stacked_by_group).
    For target='ce_nodis', disutility is zero in the value function
    so the distinction is irrelevant.
 
    Parameters
    ----------
    model    : solved IHANKModelClass (or any compatible instance)
    target   : 'C_hh' | 'ce_nodis' | 'ce_avg' | 'ce_sec'
    H        : int, quarters to report (default 20 = 5 years)
    do_print : bool, progress output
 
    Returns
    -------
    irfs   : dict  {label: (H,) % deviation from SS}
             Keys: 'full' and one per channel in CHANNELS.
    ss_lev : float, SS level of target (for reference)
    """
    Dbeg   = model.ss.Dbeg
    ss_lev = _ss_level(model, target)
    n_dis  = target in ('ce_avg', 'ce_sec')  # N channel includes disutility?
 
    def _pct(arr):
        return (arr - ss_lev) / ss_lev * 100.0
 
    # full GE transition aggregated with SS Dbeg
    irfs = {'full': _pct(_agg(model, target, H, Dbeg))}
 
    channel_fns = {
        'P'  : lambda: _scenario_inc_sub_irf(model, 'P'),
        'ra' : lambda: _scenario_ra_irf(model),
        'tau': lambda: _scenario_inc_sub_irf(model, 'tau'),
        'W'  : lambda: _scenario_inc_sub_irf(model, 'W'),
        'N'  : (lambda: _scenario_N_full(model))
               if n_dis
               else (lambda: _scenario_inc_sub_irf(model, 'N')),
    }
 
    for ch in CHANNELS:
        if do_print:
            print(f'  [channel_decomp] channel={ch}  target={target}',
                  flush=True)
        mc = channel_fns[ch]()
        irfs[ch] = _pct(_agg(mc, target, H, Dbeg))
        del mc
 
    return irfs, ss_lev