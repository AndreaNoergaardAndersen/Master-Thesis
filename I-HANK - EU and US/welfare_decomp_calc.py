# welfare_decomp_calc.py
# =================================================================
# Channel decomposition of the 20-period CEV loss.
#
# Approach (Auclert 2019, "Monetary Policy and the Redistribution
# Channel", non-linear version): for each channel, hold all
# inputs_hh except the channel-of-interest at SS, re-solve the
# household block only (PARTIAL EQUILIBRIUM — firm/government/
# foreign blocks NOT re-equilibrated), and recompute the welfare
# loss. The difference between this scenario's loss and the full
# transition's loss tells us how much that channel contributes.
#
# The household block in this model takes inputs_hh:
#   beta, ra,
#   inc_HH, inc_HL, inc_LH, inc_LL, inc_NT,         (income channel)
#   NHH, NHL, NLH, NLL, NNT                          (disutility channel)
# We never perturb beta (it is the preference parameter, not a
# transition input). Channels:
#   'ra'   : real return on assets (wealth/interest channel)
#   'inc'  : sectoral real disposable income (income channel,
#            collapses tax / nominal wage / labor / price-level
#            into one — see sub_decomposition.py for splitting it)
#   'N'    : sectoral employment used in labor disutility
#
# Each scenario is run independently — call only the ones you need.
# Each takes one `solve_hh_path` call on a model copy, so memory
# and compute are bounded.
# =================================================================
 
import numpy as np
from welfare_calc import compute_welfare, population_cev_loss
 
 
# inputs_hh partitioned by channel
CHANNEL_INPUTS = {
    'ra'  : ['ra'],
    'inc' : ['inc_HH', 'inc_HL', 'inc_LH', 'inc_LL', 'inc_NT'],
    'N'   : ['NHH', 'NHL', 'NLH', 'NLL', 'NNT'],
}
ALL_PERTURBABLE_INPUTS = (CHANNEL_INPUTS['ra']
                          + CHANNEL_INPUTS['inc']
                          + CHANNEL_INPUTS['N'])
 
 
# -----------------------------------------------------------------
# Scenario runner
# -----------------------------------------------------------------
 
def run_channel_scenario(model_baseline, active_channels, H=20,
                         variant='avg', do_print=False):
    """
    Re-solve the household block on a copy of `model_baseline`
    with only the inputs in `active_channels` set to their
    transition paths; all other inputs_hh are held at SS.
 
    Parameters
    ----------
    model_baseline : IHANKModelClass (solved transition)
    active_channels : list[str] subset of {'ra','inc','N'}
    H : int
    variant : 'nodis' | 'avg' | 'sec'
        Welfare variant — see welfare_calc.compute_welfare.
    do_print : bool
    """
    model_cf = model_baseline.copy()
 
    active_inputs = set()
    for c in active_channels:
        if c not in CHANNEL_INPUTS:
            raise ValueError(f"Unknown channel '{c}'. "
                             f"Choose from {list(CHANNEL_INPUTS)}.")
        active_inputs.update(CHANNEL_INPUTS[c])
 
    for inp in ALL_PERTURBABLE_INPUTS:
        if inp in active_inputs:
            continue
        path_arr = getattr(model_cf.path, inp)
        ss_val   = getattr(model_cf.ss, inp)
        path_arr[:] = ss_val
 
    model_cf.solve_hh_path(do_print=do_print)
    out = compute_welfare(model_cf, H=H, variant=variant)
    return out, model_cf
 
 
def scenario_full(model_baseline, H=20, variant='avg'):
    return run_channel_scenario(model_baseline, ['ra','inc','N'],
                                H=H, variant=variant)
 
def scenario_no_shock(model_baseline, H=20, variant='avg'):
    return run_channel_scenario(model_baseline, [],
                                H=H, variant=variant)
 
def scenario_ra_only(model_baseline, H=20, variant='avg'):
    return run_channel_scenario(model_baseline, ['ra'],
                                H=H, variant=variant)
 
def scenario_inc_only(model_baseline, H=20, variant='avg'):
    return run_channel_scenario(model_baseline, ['inc'],
                                H=H, variant=variant)
 
def scenario_N_only(model_baseline, H=20, variant='avg'):
    return run_channel_scenario(model_baseline, ['N'],
                                H=H, variant=variant)
 
 
# -----------------------------------------------------------------
# Aggregation helper
# -----------------------------------------------------------------
 
def aggregate_population_losses(scenario_dict, Dbeg, beta, sigma, H):
    """
    Given a dict {scenario_name: welfare_out}, return a dict
    {scenario_name: population CEV loss (utilitarian, fraction)}.
 
    This is the input to the bar chart in welfare_decomp_plots.
    """
    return {
        name: population_cev_loss(o['W_trans'], o['W_ss'],
                                  Dbeg, beta, sigma, H)
        for name, o in scenario_dict.items()
    }
 
 
def group_losses_by_mask(scenario_dict, masks, Dbeg, weighting='ce'):
    """
    For each scenario and each wealth/sector group mask, return the
    group-mean CEV loss. Used for the heterogeneity bar chart.
 
    Parameters
    ----------
    scenario_dict : dict {name -> welfare_out from compute_welfare}
    masks : dict {group_label -> (Nfix, Nz, Na) bool ndarray}
    Dbeg : (Nfix, Nz, Na) SS distribution
    weighting : 'ce'   -> population-weighted mean of household loss
                          (= 1 - mean(ce_trans)/mean(ce_ss)? NO — see
                          note below). Use the simple weighted mean
                          of per-household losses for the bar chart;
                          this is the conventional reporting in
                          Bayer, Born, Luetticke, Müller (2024).
 
    Returns
    -------
    out : dict of dict — out[scenario_name][group_label] = mean loss
    """
    result = {}
    for name, o in scenario_dict.items():
        loss = o['loss']
        result[name] = {}
        for g, mk in masks.items():
            w = Dbeg * mk
            result[name][g] = (loss * w).sum() / w.sum() if w.sum() > 0 else np.nan
    return result
 
 
# =================================================================
# Sub-decomposition of the income channel into (tau, W, N, P)
# =================================================================
#
# Identity from blocks.py line 477:
#   inc_j_t = (1 - tau_t) * w_j_t * N_j_t,   w_j_t = W_j_t / P_t
# Multiplicative decomposition (exact in logs):
#   inc_j_t / inc_j_ss = f_tau(t) * f_W(j,t) * f_P(t) * f_N(j,t)
# where
#   f_tau(t)   = (1 - tau_t) / (1 - tau_ss)
#   f_W(j,t)   = W_j_t  / W_j_ss   = (w_j_t * P_t) / (w_j_ss * P_ss)
#   f_P(t)     = P_ss / P_t
#   f_N(j,t)   = N_j_t / N_j_ss
#
# Each sub-counterfactual sets ONE factor to its transition value
# and the other three to one (i.e., SS). The interest channel
# (ra) and the labour-disutility channel (N in psi) are switched
# off (set to SS) in all sub-scenarios so the sub-decomposition is
# clean inside the income channel.
# =================================================================
 
INCOME_SECTORS = ['HH', 'HL', 'LH', 'LL', 'NT']
INCOME_SUB_CHANNELS = ('tau', 'W', 'N', 'P')
 
 
def _factor_path(model_baseline, sub):
    """
    Build the (T,) factor f_sub(t) for the requested sub-channel.
    Returns either a scalar (T,) array (tau, P) or a dict by sector
    of (T,) arrays (W, N).
    """
    ss, path = model_baseline.ss, model_baseline.path
    if sub == 'tau':
        tau_t = np.asarray(path.tau).ravel()
        return (1.0 - tau_t) / (1.0 - ss.tau)
    if sub == 'P':
        P_t = np.asarray(path.P).ravel()
        return ss.P / P_t
    if sub == 'W':
        # W_j_t / W_j_ss  =  (w_j_t * P_t) / (w_j_ss * P_ss)
        P_t = np.asarray(path.P).ravel()
        out = {}
        for j in INCOME_SECTORS:
            w_t  = np.asarray(getattr(path, f'w{j}')).ravel()
            w_ss = getattr(ss, f'w{j}')
            out[j] = (w_t * P_t) / (w_ss * ss.P)
        return out
    if sub == 'N':
        out = {}
        for j in INCOME_SECTORS:
            N_t  = np.asarray(getattr(path, f'N{j}')).ravel()
            N_ss = getattr(ss, f'N{j}')
            out[j] = N_t / N_ss
        return out
    raise ValueError(f"Unknown sub-channel '{sub}'.")
 
 
def scenario_inc_sub(model_baseline, sub, H=20, variant='avg',
                     do_print=False):
    """
    Sub-decomposition of the income channel. Only the requested
    factor (tau, W, N, or P) is allowed to move in the construction
    of inc_j_t paths. Interest and labour-disutility channels are
    switched off.
 
    Parameters
    ----------
    model_baseline : IHANKModelClass (solved transition)
    sub : str   one of {'tau', 'W', 'N', 'P'}
    H : int
    variant : 'nodis' | 'avg' | 'sec'
    """
    if sub not in INCOME_SUB_CHANNELS:
        raise ValueError(f"sub must be in {INCOME_SUB_CHANNELS}")
 
    model_cf = model_baseline.copy()
    ss, path = model_cf.ss, model_cf.path
 
    factor = _factor_path(model_baseline, sub)
 
    for j in INCOME_SECTORS:
        inc_ss_j = getattr(ss, f'inc_{j}')
        f_jt = factor[j] if isinstance(factor, dict) else factor
        getattr(path, f'inc_{j}')[:] = (inc_ss_j * f_jt).reshape(-1, 1)
 
    path.ra[:] = ss.ra
    for j in INCOME_SECTORS:
        getattr(path, f'N{j}')[:] = getattr(ss, f'N{j}')
 
    model_cf.solve_hh_path(do_print=do_print)
 
    out = compute_welfare(model_cf, H=H, variant=variant)
    return out, model_cf
 
 
def run_income_sub_decomposition(model_baseline, H=20, variant='avg',
                                 do_print=False):
    """Run all four sub-channels under one variant. Returns dict."""
    out = {}
    for sub in INCOME_SUB_CHANNELS:
        if do_print:
            print(f'  ... solving inc sub-channel: {sub}')
        out[sub], _ = scenario_inc_sub(model_baseline, sub, H=H,
                                       variant=variant, do_print=False)
    return out