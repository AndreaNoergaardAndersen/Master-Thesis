# aggregate_plots.py
# =================================================================
# Plotting for the aggregate-welfare main figure (a 2x2 panel).
#
# Layout:
#   top-left  : IRF of GDP (% deviation from SS)
#   top-right : IRF of household consumption C_hh (% dev)
#   bot-left  : IRF of CPI / P (% dev)
#   bot-right : Bar chart of cumulative 20-period DKK deviations
#               for GDP, C_hh, and the SS-basket excess cost.
#               Losses are SIGNED — losses point downward.
#
# The CEV number is not put on the figure (per the user choice);
# it is returned by the calc module and the driver prints it.
# =================================================================

import numpy as np
import matplotlib.pyplot as plt


def _fmt_billion_dkk(x):
    """Format a DKK number as bn DKK with sign and one decimal."""
    return f'{x/1e9:+.1f} bn DKK'


def plot_aggregate_main(agg, title=None, figsize=(12, 8)):
    """
    Render the 2x2 main aggregate figure from the dict produced by
    aggregate_calc.compute_aggregate_paths.

    Parameters
    ----------
    agg : dict
        Output of compute_aggregate_paths.
    title : str or None
        Optional figure suptitle (e.g. 'Trade war: tau_m + tau_x').
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure

    Notes
    -----
    The IRF panels share a horizontal zero line and use the model
    convention that GDP_ss = C_hh_ss = P_ss = 1 (so % deviation is
    directly comparable across panels).

    The bar panel includes a horizontal zero reference. Losses
    appear as downward bars; gains (or disinflation, in the CPI
    bar) appear as upward bars. Colour codes:
        red    = welfare-relevant loss
                 (GDP loss, C_hh loss, higher cost of SS bundle)
        teal   = welfare-relevant gain
                 (less common but possible — flagged when it occurs)
    The CPI cost bar uses the *cost* sign convention: positive = more
    expensive to consume the SS bundle = bad for households; negative
    = cheaper SS bundle (disinflation relative to SS).
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    t = agg['t']

    # ---- Panel 1: GDP IRF ----
    ax = axes[0, 0]
    ax.plot(t, agg['irf_GDP_pct'], color='steelblue', lw=2)
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('GDP')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')

    # ---- Panel 2: C_hh IRF ----
    ax = axes[0, 1]
    ax.plot(t, agg['irf_C_pct'], color='darkorange', lw=2)
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('Household consumption $C_{hh}$')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')

    # ---- Panel 3: CPI IRF ----
    ax = axes[1, 0]
    ax.plot(t, agg['irf_CPI_pct'], color='seagreen', lw=2)
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_title('CPI (price level $P$)')
    ax.set_xlabel('Quarters after shock')
    ax.set_ylabel('% deviation from SS')

    # ---- Panel 4: Cumulative DKK bars ----
    # Losses point DOWN: we plot the signed cumulative deviation
    # directly, so negative cumulative GDP / C_hh deviations become
    # negative bars. For the CPI bar we plot the SS-basket excess
    # cost — positive means more expensive (bad for households).
    ax = axes[1, 1]
    labels = ['GDP', '$C_{hh}$', 'SS-basket cost']
    values = [agg['cum_GDP_DKK'], agg['cum_C_DKK'],
              agg['cum_CPI_cost_DKK']]

    # Sign-aware colours.
    # For GDP and C_hh: negative = loss → red.
    # For SS-basket cost: positive = more expensive = welfare loss
    #   → red; negative = disinflation relative to SS → teal.
    def _color(label, value):
        if label == 'SS-basket cost':
            return 'firebrick' if value > 0 else 'teal'
        return 'firebrick' if value < 0 else 'teal'

    colours = [_color(l, v) for l, v in zip(labels, values)]

    xs = np.arange(len(labels))
    bars = ax.bar(xs, np.array(values)/1e9, color=colours,
                  edgecolor='white')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Cumulative deviation over 20 quarters\n(bn DKK)')
    ax.set_title('Cumulative DKK loss (5 years)')

    # annotate each bar with its DKK value
    for bar, v in zip(bars, values):
        height_bn = v / 1e9
        ax.annotate(_fmt_billion_dkk(v),
                    xy=(bar.get_x() + bar.get_width()/2, height_bn),
                    xytext=(0, 4 if height_bn >= 0 else -14),
                    textcoords='offset points',
                    ha='center', fontsize=10)

    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=15)

    plt.tight_layout()
    return fig
