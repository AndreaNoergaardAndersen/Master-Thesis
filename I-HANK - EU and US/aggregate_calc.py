# aggregate_calc.py
# =================================================================
# Aggregate paths and 20-period cumulative DKK losses for the
# I-HANK model. Used by the main aggregate welfare figure (panels
# 1-4 of figure 9-equivalent on the aggregate side).
#
# Variables we extract from model.path / model.ss:
#   GDP    : real GDP (already includes tariff revenue, see
#            blocks.accounting line 619-624). Normalised to 1 at SS.
#   C_hh   : real aggregate household consumption.
#   P      : CPI / consumption price index. Normalised to 1 at SS.
#
# The function is written to work on ANY model object that has
# been through find_transition_path — pass m_tau_war, m_tau_m_only,
# m_tau_x_only, etc. interchangeably.
# =================================================================
 
import numpy as np
from welfare_calc import compute_welfare, population_cev_loss
 
 
# -----------------------------------------------------------------
# Main calculation
# -----------------------------------------------------------------
 
def compute_aggregate_paths(model, H=20, DKK_per_GDP_unit=780e9):
    """
    Pull the GDP / C_hh / P paths from a solved model, compute the
    20-period IRFs (% deviation from SS) and the 20-period
    cumulative deviations in DKK.
 
    Parameters
    ----------
    model : IHANKModelClass instance
        Must have been through find_transition_path before this is
        called. Reads model.path and model.ss.
    H : int
        Truncation horizon, in quarters. Defaults to 20 = 5 years.
    DKK_per_GDP_unit : float
        Scaling factor that maps one unit of model real GDP to DKK.
        Default: 780e9 DKK, based on Statistics Denmark Q4-2025
        nominal quarterly GDP (annual 2025 GDP ≈ 3090 bn DKK / 4).
        Override from the notebook for sensitivity. Source:
        Danmarks Statistik, Nøgletal for nationalregnskabet (BNP),
        https://www.dst.dk
 
    Returns
    -------
    out : dict
        't'              : (H,) array of 0..H-1 quarter indices
        'irf_GDP_pct'    : (H,) % deviation of GDP from SS
        'irf_C_pct'      : (H,) % deviation of C_hh from SS
        'irf_CPI_pct'    : (H,) % deviation of P from SS
        'cum_GDP_DKK'    : scalar, sum_{t=0..H-1} (GDP_t - GDP_ss)
                           * DKK_per_GDP_unit
        'cum_C_DKK'      : scalar, same for C_hh
        'cum_CPI_cost_DKK' : scalar, sum_{t=0..H-1} (P_t - 1) *
                             C_hh_ss * DKK_per_GDP_unit, i.e. the
                             excess cost over H quarters of buying
                             the SS consumption bundle at transition
                             prices (Laspeyres concept; will be
                             NEGATIVE if the transition is on net
                             disinflationary relative to SS).
        'DKK_per_GDP_unit' : echo of the scaling assumption
        'H'              : echo of the horizon
 
    Notes
    -----
    All "loss" magnitudes are signed: a cumulative GDP loss appears
    as a NEGATIVE number, which the bar chart then plots as a
    downward bar.
 
    GDP in this model includes tariff revenue (see blocks.accounting
    line 619-624). For the trade-war shock, tariff revenue is
    positive, so GDP including tariff revenue contracts less than
    the value-added measure alone. We report the model's GDP object
    directly for internal consistency with the IRFs.
    """
    ss   = model.ss
    path = model.path
 
    # path.GDP / path.C_hh / path.P are stored with shape (T, 1) in
    # GEModelTools; flatten to 1D for clean array arithmetic.
    GDP = np.asarray(path.GDP).ravel()
    C   = np.asarray(path.C_hh).ravel()
    P   = np.asarray(path.P).ravel()
 
    # ---- IRFs in % deviation from SS ----
    GDP_dev_pct = (GDP[:H] - ss.GDP) / ss.GDP * 100.0
    C_dev_pct   = (C[:H]   - ss.C_hh) / ss.C_hh * 100.0
    CPI_dev_pct = (P[:H]   - ss.P)   / ss.P   * 100.0
 
    # ---- Cumulative DKK deviations ----
    # In model units, GDP_ss = 1 and P_ss = 1 by normalisation.
    # The scaling factor DKK_per_GDP_unit converts one unit of
    # model real GDP into DKK.
    cum_GDP = (GDP[:H] - ss.GDP).sum() * DKK_per_GDP_unit
    cum_C   = (C[:H]   - ss.C_hh).sum() * DKK_per_GDP_unit
 
    # CPI bar: excess cost over H quarters of the SS bundle at
    # transition prices. Laspeyres concept.
    cum_CPI_cost = (P[:H] - ss.P).sum() * ss.C_hh * DKK_per_GDP_unit
 
    return {
        't'                : np.arange(H),
        'irf_GDP_pct'      : GDP_dev_pct,
        'irf_C_pct'        : C_dev_pct,
        'irf_CPI_pct'      : CPI_dev_pct,
        'cum_GDP_DKK'      : cum_GDP,
        'cum_C_DKK'        : cum_C,
        'cum_CPI_cost_DKK' : cum_CPI_cost,
        'DKK_per_GDP_unit' : DKK_per_GDP_unit,
        'H'                : H,
    }
 
 
def compute_aggregate_cev(model, H=20, variant='avg'):
    """
    Convenience wrapper: population truncated CEV loss over H
    periods under the chosen welfare variant.
 
    variant : 'nodis' | 'avg' | 'sec'  (see welfare_calc.compute_welfare)
    """
    out = compute_welfare(model, H=H, variant=variant)
    return population_cev_loss(out['W_trans'], out['W_ss'],
                               model.ss.Dbeg,
                               model.par.beta, model.par.sigma, H=H)