# gdp_decomp_plots.py
# =================================================================
# Figure 7 — demand-side decomposition of GDP deviations.
#
# Model identity (from blocks.accounting):
#   GDP = C_hh + (PNT/P) * G + NX
#   GDP = value_added + tariff_rev   (tariff revenue is inside GDP)
#
# We split the contribution to GDP into four parts that sum to
# Delta GDP exactly:
#   1. Delta C_hh
#   2. Delta G_eff = Delta((PNT/P) * G)
#   3. Delta tariff_rev
#   4. Delta NX_trade = Delta NX - Delta tariff_rev
#      (the "true" trade-balance part of NX, excluding the
#       mechanical tariff-revenue contribution)
#
# Stacked area chart over t = 0..H-1, with a black line on top
# tracing Delta GDP. The shaded areas should hug the line.
# =================================================================

import numpy as np
import matplotlib.pyplot as plt


def compute_gdp_decomp(model, H=20):
    """
    Decompose GDP deviations into C_hh, G_eff, tariff_rev, and
    NX_trade contributions over H periods.

    Returns
    -------
    out : dict with keys
        't', 'dGDP', 'dC', 'dG_eff', 'dTariff', 'dNX_trade'
        all (H,) ndarrays of absolute deviations from SS (in model
        GDP units; recall GDP_ss = 1 by normalisation).
    """
    ss, path = model.ss, model.path

    GDP   = np.asarray(path.GDP).ravel()[:H]
    C     = np.asarray(path.C_hh).ravel()[:H]
    NX    = np.asarray(path.NX).ravel()[:H]
    G     = np.asarray(path.G).ravel()[:H]
    P     = np.asarray(path.P).ravel()[:H]
    PNT   = np.asarray(path.PNT).ravel()[:H]
    PMus  = np.asarray(path.PM_dk_us).ravel()[:H]
    PFus  = np.asarray(path.PF_us).ravel()[:H]
    CTFus = np.asarray(path.CTF_us).ravel()[:H]
    Mus_h  = np.asarray(path.M_dk_us_h).ravel()[:H]
    Mus_hx = np.asarray(path.M_dk_us_hx).ravel()[:H]
    Mus_l  = np.asarray(path.M_dk_us_l).ravel()[:H]
    Mus_lx = np.asarray(path.M_dk_us_lx).ravel()[:H]
    tau_m  = np.asarray(path.tau_m).ravel()[:H]

    # G_eff = (PNT/P) * G  (effective real G contribution to GDP)
    G_eff      = (PNT / P) * G
    G_eff_ss   = (ss.PNT / ss.P) * ss.G

    # tariff_rev follows the exact accounting-block formula
    M_us_total = Mus_h + Mus_hx + Mus_l + Mus_lx
    tariff_rev = (tau_m/(1+tau_m)) * (PMus/P) * M_us_total \
                 + (tau_m/(1+tau_m)) * (PFus/P) * CTFus
    # SS tariff_rev: tau_m_ss = 0, so tariff_rev_ss = 0
    tariff_rev_ss = 0.0

    # NX_trade: the part of NX that is NOT tariff revenue
    NX_trade    = NX - tariff_rev
    NX_trade_ss = ss.NX - tariff_rev_ss

    # Deviations
    dGDP      = GDP   - ss.GDP
    dC        = C     - ss.C_hh
    dG_eff    = G_eff - G_eff_ss
    dTariff   = tariff_rev - tariff_rev_ss
    dNX_trade = NX_trade   - NX_trade_ss

    # Consistency check: dGDP should equal sum of the four components
    # to within numerical tolerance.
    residual = dGDP - (dC + dG_eff + dTariff + dNX_trade)

    return {
        't': np.arange(H),
        'dGDP': dGDP, 'dC': dC, 'dG_eff': dG_eff,
        'dTariff': dTariff, 'dNX_trade': dNX_trade,
        'residual': residual,
    }


def plot_gdp_decomp(decomp, title=None, figsize=(10, 6)):
    """
    Stacked area chart of GDP deviation by demand-side component
    plus tariff revenue, with a black line tracking total Delta GDP.

    Positive contributions stack upward from zero; negative
    contributions stack downward. The four shaded areas always sum
    to the Delta GDP line.
    """
    t = decomp['t']
    components = [
        ('$\\Delta C_{hh}$',     decomp['dC'],        '#1f77b4'),
        ('$\\Delta G_{eff}$',    decomp['dG_eff'],    '#2ca02c'),
        ('$\\Delta\\, tariff\\_rev$', decomp['dTariff'], '#ff7f0e'),
        ('$\\Delta NX_{trade}$', decomp['dNX_trade'], '#d62728'),
    ]

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)

    # Stack positives upward and negatives downward separately so
    # both signs are visible.
    pos_cum = np.zeros_like(t, dtype=float)
    neg_cum = np.zeros_like(t, dtype=float)
    for label, vals, color in components:
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        ax.fill_between(t, pos_cum, pos_cum + pos,
                        color=color, alpha=0.85, label=label,
                        edgecolor='white', linewidth=0.5)
        ax.fill_between(t, neg_cum, neg_cum + neg,
                        color=color, alpha=0.85,
                        edgecolor='white', linewidth=0.5)
        pos_cum = pos_cum + pos
        neg_cum = neg_cum + neg

    ax.plot(t, decomp['dGDP'], color='black', lw=2.0,
            label='$\\Delta$GDP (total)')

    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('Deviation from SS (model GDP units)')
    ax.set_title(title if title else
                 'GDP decomposition: $\\Delta Y = \\Delta C + \\Delta G_{eff} + \\Delta\\, tariff\\_rev + \\Delta NX_{trade}$')
    ax.legend(loc='best', fontsize=11, ncol=2)
    plt.tight_layout()
    return fig
