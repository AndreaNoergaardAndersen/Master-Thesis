# trade_plots.py
# =================================================================
# Figure 6 — explanatory trade-dynamics figure.
#
# 2x2 layout:
#   top-left  : nominal exchange rates E (DKK/EUR) and E_us (DKK/USD)
#   top-right : real exchange rates Q (vs EU) and Q_us (vs US)
#   bot-left  : Danish exports — to EU vs to US (aggregated across
#               the four traded sectors HH, HL, LH, LL)
#   bot-right : Danish imports — from EU vs from US (materials
#               aggregated across high/low intensity, plus the
#               household import basket CTF_eu / CTF_us)
#
# All series in % deviation from SS so they share an axis. Each
# function takes a model object so it can be reused on m_tau_war,
# m_tau_m_only, m_tau_x_only, etc.
# =================================================================

import numpy as np
import matplotlib.pyplot as plt

from thesis_graph_style import *

set_thesis_style(use_latex=False)
make_graph_folders()

def _pct_dev(path_arr, ss_val):
    """% deviation of a (T,1) path from SS, returned as (T,) ndarray."""
    return (np.asarray(path_arr).ravel() - ss_val) / ss_val * 100.0


def compute_trade_paths(model, H=20):
    """
    Pull the eight series needed for figure 6 and return them as
    H-period IRFs in % deviation from SS.

    Aggregates exports across the four traded sectors (HH, HL, LH,
    LL) by destination, and imports across high/low material
    classes by source, plus the household import baskets CTF_eu and
    CTF_us.

    Returns
    -------
    out : dict with keys
        't'              : np.arange(H)
        'E', 'E_us'      : (H,) % deviations of nominal exchange rates
        'Q', 'Q_us'      : (H,) % deviations of real exchange rates
        'X_eu', 'X_us'   : (H,) % deviations of Danish exports to EU/US
        'M_eu', 'M_us'   : (H,) % deviations of Danish imports from EU/US
    """
    ss, path = model.ss, model.path

    # ----- exchange rates -----
    E_dev    = _pct_dev(path.E,    ss.E)[:H]
    Eus_dev  = _pct_dev(path.E_us, ss.E_us)[:H]
    Q_dev    = _pct_dev(path.Q,    ss.Q)[:H]
    Qus_dev  = _pct_dev(path.Q_us, ss.Q_us)[:H]

    # ----- exports by destination, aggregated across sectors -----
    # CTH_<sector>_eu_s and CTH_<sector>_us_s are EU/US demand for
    # each Danish traded sector. Total exports = sum across sectors.
    X_eu_path = (np.asarray(path.CTH_HH_eu_s).ravel()
                 + np.asarray(path.CTH_HL_eu_s).ravel()
                 + np.asarray(path.CTH_LH_eu_s).ravel()
                 + np.asarray(path.CTH_LL_eu_s).ravel())
    X_eu_ss   = (ss.CTH_HH_eu_s + ss.CTH_HL_eu_s
                 + ss.CTH_LH_eu_s + ss.CTH_LL_eu_s)
    X_us_path = (np.asarray(path.CTH_HH_us_s).ravel()
                 + np.asarray(path.CTH_HL_us_s).ravel()
                 + np.asarray(path.CTH_LH_us_s).ravel()
                 + np.asarray(path.CTH_LL_us_s).ravel())
    X_us_ss   = (ss.CTH_HH_us_s + ss.CTH_HL_us_s
                 + ss.CTH_LH_us_s + ss.CTH_LL_us_s)
    X_eu_dev = (X_eu_path[:H] - X_eu_ss) / X_eu_ss * 100.0
    X_us_dev = (X_us_path[:H] - X_us_ss) / X_us_ss * 100.0

    # ----- imports by source, aggregated -----
    # Materials: M_dk_eu_h + M_dk_eu_l (EU) ; M_dk_us_h + M_dk_us_l (US)
    # Plus the M_dk_*_hx and M_dk_*_lx aggregator pieces are inputs
    # to the same production technology — sum them in.
    # Household imports: CTF_eu (from EU), CTF_us (from US).
    M_eu_path = (np.asarray(path.M_dk_eu_h).ravel()
                 + np.asarray(path.M_dk_eu_l).ravel()
                 + np.asarray(path.M_dk_eu_hx).ravel()
                 + np.asarray(path.M_dk_eu_lx).ravel()
                 + np.asarray(path.CTF_eu).ravel())
    M_eu_ss   = (ss.M_dk_eu_h + ss.M_dk_eu_l
                 + ss.M_dk_eu_hx + ss.M_dk_eu_lx + ss.CTF_eu)
    M_us_path = (np.asarray(path.M_dk_us_h).ravel()
                 + np.asarray(path.M_dk_us_l).ravel()
                 + np.asarray(path.M_dk_us_hx).ravel()
                 + np.asarray(path.M_dk_us_lx).ravel()
                 + np.asarray(path.CTF_us).ravel())
    M_us_ss   = (ss.M_dk_us_h + ss.M_dk_us_l
                 + ss.M_dk_us_hx + ss.M_dk_us_lx + ss.CTF_us)
    M_eu_dev = (M_eu_path[:H] - M_eu_ss) / M_eu_ss * 100.0
    M_us_dev = (M_us_path[:H] - M_us_ss) / M_us_ss * 100.0

    return {
        't': np.arange(H),
        'E': E_dev, 'E_us': Eus_dev,
        'Q': Q_dev, 'Q_us': Qus_dev,
        'X_eu': X_eu_dev, 'X_us': X_us_dev,
        'M_eu': M_eu_dev, 'M_us': M_us_dev,
    }


def plot_trade_main(trd, title=None, figsize=(12, 8)):
    """Render the 2x2 trade-dynamics figure from compute_trade_paths."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    t = trd['t']

    # Top-left: nominal exchange rates
    ax = axes[0, 0]
    ax.plot(t, trd['E'],    color='steelblue', lw=2, label='$E$ (DKK/EUR)')
    ax.plot(t, trd['E_us'], color='firebrick', lw=2, linestyle='--',
            label='$E_{us}$ (DKK/USD)')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('Nominal exchange rates')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')
    ax.legend(loc='best', fontsize=11)

    # Top-right: real exchange rates
    ax = axes[0, 1]
    ax.plot(t, trd['Q'],    color='steelblue', lw=2, label='$Q$ (vs EU)')
    ax.plot(t, trd['Q_us'], color='firebrick', lw=2, linestyle='--',
            label='$Q_{us}$ (vs US)')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('Real exchange rates')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')
    ax.legend(loc='best', fontsize=11)

    # Bot-left: exports by destination
    ax = axes[1, 0]
    ax.plot(t, trd['X_eu'], color='steelblue', lw=2, label='to EU')
    ax.plot(t, trd['X_us'], color='firebrick', lw=2, linestyle='--',
            label='to US')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('Danish exports')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')
    ax.legend(loc='best', fontsize=11)

    # Bot-right: imports by source
    ax = axes[1, 1]
    ax.plot(t, trd['M_eu'], color='steelblue', lw=2, label='from EU')
    ax.plot(t, trd['M_us'], color='firebrick', lw=2, linestyle='--',
            label='from US')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('Danish imports')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')
    ax.legend(loc='best', fontsize=11)

    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=15)
    plt.tight_layout()
    return fig
