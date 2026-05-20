# welfare_channel_graphs.py
# =================================================================
# Publication-quality IRF figures for the welfare channel analysis.
#
#   Graph 1 : Transmission mechanism — P, ra, tau, W̄, N̄
#   Graph 2 : Channel decomposition of a target outcome variable
#   Graph 3 : Sectoral nominal wages W_j and employment N_j
#
# All three functions accept `models` as either:
#   - a single solved model, or
#   - a dict {label: model}  (e.g. {'Trade war': m_war, 'tau_m': m_m})
# This makes it trivial to overlay multiple scenarios on one figure.
#
# Graph 2 additionally accepts `precomputed` to avoid re-running
# the five solve_hh_path() calls when iterating over target variables.
# =================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from channel_decomp import run_channel_decomp, CHANNELS

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────

_SECTORS = ['HH', 'HL', 'LH', 'LL', 'NT']

_SECTOR_LABELS = {
    'HH': 'High mat. / High US ($HH$)',
    'HL': 'High mat. / Low US ($HL$)',
    'LH': 'Low mat. / High US ($LH$)',
    'LL': 'Low mat. / Low US ($LL$)',
    'NT': 'Non-tradeable ($NT$)',
}

_CH_LABELS = {
    'P'   : r'Price level ($P$)',
    'ra'  : r'Real return ($r^a$)',
    'tau' : r'Income tax ($\tau$)',
    'W'   : r'Nominal wage ($\bar{W}$)',
    'N'   : r'Employment ($\bar{N}$)',
    'full': 'Full (PE, SS dist.)',
}

# Colour-blind-friendly palette (based on Okabe–Ito 2008)
_CH_COLORS = {
    'P'   : '#D55E00',   # vermillion
    'ra'  : '#0072B2',   # blue
    'tau' : '#009E73',   # green
    'W'   : '#CC79A7',   # pink
    'N'   : '#E69F00',   # amber
    'full': '#000000',   # black
}
_SECTOR_COLORS = ['#D55E00', '#0072B2', '#009E73', '#CC79A7', '#E69F00']

_TARGET_LABELS = {
    'C_hh'    : r'Aggregate consumption ($C_{hh}$)',
    'ce_nodis': r'CE welfare – no labour disutility ($\overline{ce}^{\,nodis}$)',
    'ce_avg'  : r'CE welfare – average disutility ($\overline{ce}^{\,avg}$)',
    'ce_sec'  : r'CE welfare – sector-specific disutility ($\overline{ce}^{\,sec}$)',
}

# Line styles for multi-model overlays (up to 4 scenarios)
_LINE_STYLES = ['-', '--', ':', '-.']


# ──────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────

def _normalise_models(models):
    """Accept a single model or a dict; always return a dict."""
    if isinstance(models, dict):
        return models
    return {'Model': models}


def _r(arr, H=None):
    """Flatten path variable to 1-D float64; optionally truncate."""
    out = np.asarray(arr, dtype=float).ravel()
    return out[:H] if H is not None else out


def _pct(path_arr, ss_val, H):
    """% deviation of path_arr from ss_val over H periods."""
    return (_r(path_arr, H) - ss_val) / ss_val * 100.0


def _sNT(par):
    return 1.0 - par.sHH - par.sHL - par.sLH - par.sLL


def _weighted_W(model, H):
    """
    Population-share-weighted average nominal wage, (H,) % deviation.
    Weights: fixed sector employment shares (par.sHH, …, sNT).
    """
    par, ss, path = model.par, model.ss, model.path
    sNT = _sNT(par)
    weights = [par.sHH, par.sHL, par.sLH, par.sLL, sNT]
    W_bar_t  = sum(w * _r(getattr(path, f'W{j}'), H)
                   for w, j in zip(weights, _SECTORS))
    W_bar_ss = sum(w * float(getattr(ss, f'W{j}'))
                   for w, j in zip(weights, _SECTORS))
    return (W_bar_t - W_bar_ss) / W_bar_ss * 100.0


def _total_N(model, H):
    """Aggregate employment (sum across all sectors), (H,) % deviation."""
    ss, path = model.ss, model.path
    N_bar_t  = sum(_r(getattr(path, f'N{j}'), H) for j in _SECTORS)
    N_bar_ss = sum(float(getattr(ss,  f'N{j}'))  for j in _SECTORS)
    return (N_bar_t - N_bar_ss) / N_bar_ss * 100.0


def _hline_and_style(ax):
    """Add a zero reference line and polish tick labels."""
    ax.axhline(0, color='k', linewidth=0.6, linestyle=':', zorder=1)
    ax.tick_params(labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))


# ──────────────────────────────────────────────────────────────────
# Graph 1 : Transmission mechanism IRFs
# ──────────────────────────────────────────────────────────────────

def plot_graph1(models, H=20, figsize=(14, 4), save_path=None):
    """
    Five-panel IRF: P, ra, tau, W̄ (population-weighted nominal wage),
    N̄ (aggregate employment). All shown as % deviation from SS over
    H quarters.

    Parameters
    ----------
    models    : solved model, or dict {label: model}
    H         : int, quarters to plot (≤ par.T)
    figsize   : tuple, (width, height) in inches
    save_path : str or None — if given, saves the figure to this path
    """
    models = _normalise_models(models)
    t = np.arange(H)

    # Variable definitions: (panel title, lambda model → (H,) % dev)
    var_specs = [
        (r'Price level ($P$)',
         lambda m: _pct(m.path.P,   float(m.ss.P),   H)),
        (r'Real return ($r^a$)',
         lambda m: _pct(m.path.ra,  float(m.ss.ra),  H)),
        (r'Income tax rate ($\tau$)',
         lambda m: _pct(m.path.tau, float(m.ss.tau), H)),
        (r'Avg. nominal wage ($\bar{W}$)',
         lambda m: _weighted_W(m, H)),
        (r'Aggregate employment ($\bar{N}$)',
         lambda m: _total_N(m, H)),
    ]

    fig, axes = plt.subplots(1, 5, figsize=figsize)
    fig.suptitle(
        'Transmission Mechanism — % Deviation from Steady State',
        fontsize=12, y=1.03)

    for ax, (title, fn) in zip(axes, var_specs):
        for (label, model), ls in zip(models.items(), _LINE_STYLES):
            ax.plot(t, fn(model), ls, linewidth=1.8, label=label)
        _hline_and_style(ax)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel('Quarters', fontsize=8.5)

    axes[0].set_ylabel('% deviation from SS', fontsize=8.5)

    # Legend: only if multiple scenarios
    if len(models) > 1:
        axes[2].legend(fontsize=8, loc='best')

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)
    return fig, axes


# ──────────────────────────────────────────────────────────────────
# Graph 2 : Channel decomposition
# ──────────────────────────────────────────────────────────────────

def plot_graph2(models, target='C_hh', H=20,
                precomputed=None,
                figsize=None, save_path=None,
                do_print=False):
    """
    Channel decomposition: response of ``target`` when only one
    HH-block input channel varies at a time (partial equilibrium,
    SS initial distribution weights).

    One sub-panel per model in ``models``.

    Parameters
    ----------
    models      : solved model, or dict {label: model}
    target      : 'C_hh' | 'ce_nodis' | 'ce_avg' | 'ce_sec'
    H           : int, quarters to plot
    precomputed : dict {label: (irfs, ss_lev)} — output of
                  run_channel_decomp() for each model label.
                  Pass to avoid re-computation when switching target.
    figsize     : tuple or None (auto-sized)
    save_path   : str or None
    do_print    : bool, progress output

    Returns
    -------
    fig, axes, decomp_results
        decomp_results : dict {label: (irfs, ss_lev)} — cache for
                         subsequent calls with a different target.
    """
    models = _normalise_models(models)
    n = len(models)
    t = np.arange(H)
    figsize = figsize or (5.0 * n, 4.5)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=(n > 1),
                             squeeze=False)
    axes = axes[0]   # shape (n,)

    fig.suptitle(
        f'Channel Decomposition  —  {_TARGET_LABELS.get(target, target)}\n'
        r'(Partial equilibrium · SS distribution weights)',
        fontsize=11, y=1.04)

    decomp_results = {}

    for ax, (label, model) in zip(axes, models.items()):

        # ── Compute or retrieve ──────────────────────────────────
        if precomputed and label in precomputed:
            irfs, ss_lev = precomputed[label]
        else:
            if do_print:
                print(f'  Computing channel decomp: "{label}" …', flush=True)
            irfs, ss_lev = run_channel_decomp(model, target=target,
                                               H=H, do_print=do_print)
        decomp_results[label] = (irfs, ss_lev)

        # ── Full PE reference line ───────────────────────────────
        ax.plot(t, irfs['full'],
                color=_CH_COLORS['full'], linewidth=2.0,
                linestyle='--', label=_CH_LABELS['full'], zorder=5)

        # ── Channel lines ────────────────────────────────────────
        for ch in CHANNELS:
            ax.plot(t, irfs[ch],
                    color=_CH_COLORS[ch], linewidth=1.7,
                    label=_CH_LABELS[ch])

        _hline_and_style(ax)
        ax.set_title(label, fontsize=10.5)
        ax.set_xlabel('Quarters', fontsize=8.5)
        ax.legend(fontsize=7.5, loc='best', framealpha=0.85)

    axes[0].set_ylabel('% deviation from SS', fontsize=8.5)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)

    return fig, axes, decomp_results


# ──────────────────────────────────────────────────────────────────
# Graph 3 : Sectoral nominal wages and employment
# ──────────────────────────────────────────────────────────────────

def plot_graph3(models, H=20, figsize=None, save_path=None):
    """
    Two rows × n-model columns:
      Row 1 : Sectoral nominal wage W_j (% deviation from SS)
      Row 2 : Sectoral employment N_j   (% deviation from SS)

    Five lines per panel — one per sector (HH, HL, LH, LL, NT).

    Parameters
    ----------
    models    : solved model, or dict {label: model}
    H         : int, quarters
    figsize   : tuple or None (auto)
    save_path : str or None
    """
    models = _normalise_models(models)
    n = len(models)
    t = np.arange(H)
    figsize = figsize or (5.5 * n, 7.5)

    fig, axes = plt.subplots(2, n, figsize=figsize,
                             sharey='row', sharex=True, squeeze=False)

    fig.suptitle(
        r'Sectoral Nominal Wages and Employment — % Deviation from SS',
        fontsize=12, y=1.01)

    for col, (label, model) in enumerate(models.items()):
        ss, path = model.ss, model.path
        ax_W = axes[0, col]
        ax_N = axes[1, col]

        for j, color in zip(_SECTORS, _SECTOR_COLORS):
            W_dev = _pct(getattr(path, f'W{j}'), float(getattr(ss, f'W{j}')), H)
            N_dev = _pct(getattr(path, f'N{j}'), float(getattr(ss, f'N{j}')), H)
            ax_W.plot(t, W_dev, color=color, linewidth=1.7,
                      label=_SECTOR_LABELS[j])
            ax_N.plot(t, N_dev, color=color, linewidth=1.7)

        for ax in (ax_W, ax_N):
            _hline_and_style(ax)

        ax_W.set_title(label, fontsize=11)
        ax_N.set_xlabel('Quarters', fontsize=8.5)

    # Y-axis labels on leftmost column
    axes[0, 0].set_ylabel(r'Nominal wage $W_j$' + '\n% dev. from SS',
                           fontsize=8.5)
    axes[1, 0].set_ylabel(r'Employment $N_j$' + '\n% dev. from SS',
                           fontsize=8.5)

    # Shared legend below the figure
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc='lower center', ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.04),
               framealpha=0.9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=200)

    return fig, axes