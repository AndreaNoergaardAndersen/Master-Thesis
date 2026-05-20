# inequality_plots.py
# =================================================================
# Plotting functions for the three candidate "inequality main"
# figures (item 9 in the figure list).
#
# All three functions take pre-computed arrays from welfare_calc.py
# and SS distribution weights. They never touch the model.
#
# Each function returns the Matplotlib Figure object so the caller
# can save, modify, or compose into a multi-panel layout.
# =================================================================
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from welfare_calc import (weighted_quantiles, weighted_mean,
                          lorenz_and_gini, wealth_group_masks)
 
 
# -----------------------------------------------------------------
# Figure 9a: Weighted histogram + CDF of CEV losses
# -----------------------------------------------------------------
 
def plot_loss_distribution(loss, weights, nbins=80, ax=None,
                           clip_quantiles=(0.005, 0.995)):
    """
    Histogram of household-level CEV losses, weighted by the SS
    beginning-of-period distribution. Adds a CDF overlay on a
    secondary y-axis so the reader sees both density and rank.
 
    Inputs
    ------
    loss    : (Nfix, Nz, Na) percent CEV losses (positive = welfare loss)
    weights : (Nfix, Nz, Na) same-shape ss.Dbeg
    nbins   : number of histogram bins
    ax      : optional Matplotlib axis
    clip_quantiles : (low, high) used to set x-axis limits without
                     discarding the underlying distribution. Avoids
                     axis-domination by extreme but tiny-mass states.
 
    Interpretation
    --------------
    The x-axis shows the percent of consumption a household would
    surrender over 20 quarters to avoid the trade conflict. The
    histogram bars are normalised so that mass weighted by Dbeg
    integrates to 1. The CDF overlay tells you what fraction of the
    population has loss <= x.
    """
    l = loss.ravel()
    w = weights.ravel()
    w = w / w.sum()                          # normalise
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
 
    # x-axis clipping (display only, not for histogram math)
    lo, hi = weighted_quantiles(l, w, clip_quantiles)
 
    # weighted histogram (density on left axis)
    ax.hist(l, bins=nbins, weights=w, range=(lo, hi),
            color='steelblue', edgecolor='white', alpha=0.85,
            label='Density (left)')
    ax.set_xlabel('20-period CEV loss')
    ax.set_ylabel('Density (population-weighted)')
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=2))
    ax.set_xlim(lo, hi)
 
    # mean and median lines
    mean_loss = weighted_mean(l, w)
    median_loss = weighted_quantiles(l, w, [0.5])[0]
    ax.axvline(mean_loss, color='black', linestyle='--', lw=1.2,
               label=f'Mean = {mean_loss*100:+.3f}%')
    ax.axvline(median_loss, color='black', linestyle=':', lw=1.2,
               label=f'Median = {median_loss*100:+.3f}%')
 
    # CDF overlay (right axis)
    ax2 = ax.twinx()
    order = np.argsort(l)
    cdf_x = l[order]
    cdf_y = np.cumsum(w[order])
    ax2.plot(cdf_x, cdf_y, color='firebrick', lw=2, alpha=0.8,
             label='CDF (right)')
    ax2.set_ylabel('Cumulative population share', color='firebrick')
    ax2.tick_params(axis='y', labelcolor='firebrick')
    ax2.set_ylim(0, 1)
    ax2.grid(False)
 
    # combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              fontsize=11)
    ax.set_title('Distribution of household 20-period CEV losses')
    plt.tight_layout()
    return fig
 
 
# -----------------------------------------------------------------
# Figure 9b: Percentile-loss bar chart
# -----------------------------------------------------------------
 
def plot_percentile_losses(loss, weights, ax=None,
                           percentiles=(1, 10, 25, 50, 75, 90, 99)):
    """
    Bar chart of CEV losses at selected percentiles of the loss
    distribution. The dashed horizontal line is the population-
    weighted mean loss.
 
    Inputs
    ------
    loss        : (Nfix, Nz, Na) CEV losses
    weights     : (Nfix, Nz, Na) ss.Dbeg
    percentiles : iterable of integers in 1..99
    ax          : optional axis
 
    Interpretation
    --------------
    Reads "x% of households experience a CEV loss at least as large
    as the bar at the px label." Larger right-tail bars => more
    concentration of losses in a small group.
    """
    l = loss.ravel(); w = weights.ravel()
    qs = [p / 100.0 for p in percentiles]
    vals = weighted_quantiles(l, w, qs)
    mean_loss = weighted_mean(l, w)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
 
    xs = np.arange(len(percentiles))
    colors = ['seagreen' if v < 0 else 'firebrick' for v in vals]
    bars = ax.bar(xs, vals * 100, color=colors, edgecolor='white')
    ax.axhline(mean_loss * 100, color='black', linestyle='--', lw=1.2,
               label=f'Mean = {mean_loss*100:+.3f}%')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'p{p:02d}' for p in percentiles])
    ax.set_ylabel('20-period CEV loss (%)')
    ax.set_xlabel('Loss percentile')
    ax.set_title('CEV loss by percentile of the loss distribution')
    ax.legend(loc='best', fontsize=11)
    for b, v in zip(bars, vals):
        ax.annotate(f'{v*100:+.2f}%', (b.get_x()+b.get_width()/2, v*100),
                    ha='center', va='bottom' if v >= 0 else 'top',
                    fontsize=10)
    plt.tight_layout()
    return fig
 
 
# -----------------------------------------------------------------
# Figure 9c: Lorenz curve comparison (L1) — and L2 variant
# -----------------------------------------------------------------
 
def plot_lorenz_comparison(ce_ss, ce_trans, weights, ax=None):
    """
    L1 — classic Lorenz of welfare LEVELS.
 
    Two Lorenz curves of the truncated CEV ce_bar^H, one under SS
    and one under the transition, both weighted by the SS Dbeg
    distribution. Captures how the welfare distribution shifts.
 
    Note
    ----
    For a 1% tariff the SS and transition Lorenz curves will sit
    almost on top of each other and the Gini change will be small
    in absolute magnitude. This is methodologically correct: SS
    heterogeneity is the dominant driver of welfare dispersion, and
    a 1% tariff over 20 quarters cannot move that much. The figure's
    job is to show this, not to manufacture a visual effect.
    """
    w = weights.ravel()
    pop_ss, val_ss, gini_ss = lorenz_and_gini(ce_ss.ravel(), w)
    pop_tr, val_tr, gini_tr = lorenz_and_gini(ce_trans.ravel(), w)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
 
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':',
            label='Equality (45°)')
    ax.plot(pop_ss, val_ss, color='steelblue', lw=2.2,
            label=f'Steady state (Gini = {gini_ss:.4f})')
    ax.plot(pop_tr, val_tr, color='firebrick', lw=2.2, linestyle='--',
            label=f'Transition (Gini = {gini_tr:.4f})')
    ax.set_xlabel('Cumulative population share')
    ax.set_ylabel('Cumulative share of truncated CEV')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Lorenz of 20-period CEV: SS vs transition (L1)')
    ax.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    return fig, gini_ss, gini_tr
 
 
def plot_loss_lorenz(loss, weights, ax=None):
    """
    L2 — Lorenz of CEV LOSSES (concentration of losses).
 
    Sorts households from smallest to largest loss and plots the
    cumulative share of total loss carried by the bottom-x% of
    losers. A curve far below the 45-degree line means losses are
    heavily concentrated; a curve close to the 45-degree line means
    losses are evenly spread.
 
    Caveat: if any households actually GAIN (negative loss), the
    Lorenz curve dips below zero before recovering. We handle this
    transparently — the curve is plotted as is, and the Gini number
    is reported but flagged as a "concentration index" rather than
    a textbook Gini.
    """
    l = loss.ravel(); w = weights.ravel()
    pop_cum, val_cum, gini = lorenz_and_gini(l, w)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.figure
 
    ax.plot([0, 1], [0, 1], color='black', lw=1, linestyle=':',
            label='Equal-concentration (45°)')
    ax.plot(pop_cum, val_cum, color='darkorange', lw=2.2,
            label=f'Loss concentration (index = {gini:.3f})')
    ax.set_xlabel('Cumulative population share (sorted by loss, ascending)')
    ax.set_ylabel('Cumulative share of total CEV loss')
    ax.set_xlim(0, 1)
    ax.set_aspect('equal', 'box')
    ax.set_title('Lorenz of CEV losses (L2)')
    ax.legend(loc='upper left', fontsize=11)
    plt.tight_layout()
    return fig, gini
 
 
# =================================================================
# Figures 10, 11, 12 — explanatory inequality figures
# =================================================================
 
# Canonical sector ordering and labels used across figures 10 and 12.
_SECTOR_LABELS = ['HH', 'HL', 'LH', 'LL', 'NT']
 
def _weighted_group_mean(loss, weights, mask):
    """Population-weighted mean of `loss` over the subset `mask`."""
    w = weights * mask
    return (loss * w).sum() / w.sum() if w.sum() > 0 else np.nan
 
 
# -----------------------------------------------------------------
# Figure 10: CEV loss by sector
# -----------------------------------------------------------------
 
def plot_sector_losses(loss, weights, ax=None):
    """
    Bar chart of 20-period CEV loss by sector (HH, HL, LH, LL, NT)
    with a dashed line at the population-weighted mean.
 
    The HL sector accounts for only 1% of employment in the model,
    so its bar is statistically noisy. Flag this in the caption.
    """
    n_sec = loss.shape[0]
    sec_losses = np.array([
        _weighted_group_mean(loss[i], weights[i], np.ones_like(loss[i], dtype=bool))
        for i in range(n_sec)
    ])
    sec_mass = np.array([weights[i].sum() for i in range(n_sec)])
    pop_mean = weighted_mean(loss, weights)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
 
    xs = np.arange(n_sec)
    colors = ['firebrick' if v > 0 else 'seagreen' for v in sec_losses]
    bars = ax.bar(xs, sec_losses * 100, color=colors, edgecolor='white')
    ax.axhline(pop_mean * 100, color='black', linestyle='--', lw=1.2,
               label=f'Population mean = {pop_mean*100:+.3f}%')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{lab}\n({m*100:.1f}%)'
                        for lab, m in zip(_SECTOR_LABELS, sec_mass)])
    ax.set_xlabel('Sector (population share in parentheses)')
    ax.set_ylabel('20-period CEV loss (%)')
    ax.set_title('CEV loss by sector')
    ax.legend(loc='best', fontsize=11)
    for b, v in zip(bars, sec_losses):
        ax.annotate(f'{v*100:+.3f}%',
                    (b.get_x()+b.get_width()/2, v*100),
                    ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=10)
    plt.tight_layout()
    return fig
 
 
# -----------------------------------------------------------------
# Figure 11: CEV loss by wealth group
# -----------------------------------------------------------------
 
def plot_wealth_group_losses(loss, weights, masks, ax=None):
    """
    Bar chart of 20-period CEV loss by wealth group with dashed
    population-mean line. `masks` is the dict returned by
    welfare_calc.wealth_group_masks.
    """
    labels = list(masks.keys())
    group_losses = np.array([
        _weighted_group_mean(loss, weights, masks[k]) for k in labels
    ])
    group_mass = np.array([(weights * masks[k]).sum() for k in labels])
    pop_mean = weighted_mean(loss, weights)
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
 
    xs = np.arange(len(labels))
    colors = ['firebrick' if v > 0 else 'seagreen' for v in group_losses]
    bars = ax.bar(xs, group_losses * 100, color=colors, edgecolor='white')
    ax.axhline(pop_mean * 100, color='black', linestyle='--', lw=1.2,
               label=f'Population mean = {pop_mean*100:+.3f}%')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{lab}\n({m*100:.1f}%)'
                        for lab, m in zip(labels, group_mass)])
    ax.set_xlabel('Wealth group (population share in parentheses)')
    ax.set_ylabel('20-period CEV loss (%)')
    ax.set_title('CEV loss by wealth group')
    ax.legend(loc='best', fontsize=11)
    for b, v in zip(bars, group_losses):
        ax.annotate(f'{v*100:+.3f}%',
                    (b.get_x()+b.get_width()/2, v*100),
                    ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=10)
    plt.tight_layout()
    return fig
 
 
# -----------------------------------------------------------------
# Figure 12: Sector x wealth heatmap
# -----------------------------------------------------------------
 
def plot_sector_wealth_heatmap(loss, weights, masks, ax=None,
                               cmap='RdBu_r', sector_order=None):
    """
    Heatmap of 20-period CEV loss on the sector x wealth-group grid.
    Each cell is annotated with the mean loss (top line) and the
    population mass in that cell (bottom line) so the reader can
    see which cells carry meaningful weight.
 
    Colour scale is symmetric around zero so positive losses (red)
    and negative gains (blue) are visually comparable.
    """
    wealth_labels = list(masks.keys())
    n_sec, n_wealth = loss.shape[0], len(wealth_labels)
 
    grid_loss = np.full((n_sec, n_wealth), np.nan)
    grid_mass = np.zeros((n_sec, n_wealth))
 
    for i in range(n_sec):
        for j, k in enumerate(wealth_labels):
            cell_mask = np.zeros_like(loss, dtype=bool)
            cell_mask[i] = masks[k][i]
            w_cell = weights * cell_mask
            m = w_cell.sum()
            grid_mass[i, j] = m
            if m > 0:
                grid_loss[i, j] = (loss * w_cell).sum() / m

    # reorder rows if requested
    if sector_order is not None:
        grid_loss = grid_loss[sector_order, :]
        grid_mass = grid_mass[sector_order, :]
        row_labels = [_SECTOR_LABELS[i] for i in sector_order]
    else:
        row_labels = _SECTOR_LABELS
 
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
 
    vmax = np.nanmax(np.abs(grid_loss)) * 100
    im = ax.imshow(grid_loss * 100, cmap=cmap, aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(n_wealth))
    ax.set_xticklabels(wealth_labels)
    ax.set_yticks(np.arange(n_sec))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel('Wealth group')
    ax.set_ylabel('Sector')
    ax.set_title('CEV loss by sector × wealth (% loss; pop. mass below)')
 
    # cell annotations: value (top), mass (bottom)
    for i in range(n_sec):
        for j in range(n_wealth):
            v = grid_loss[i, j]
            m = grid_mass[i, j]
            if not np.isnan(v):
                ax.text(j, i-0.15, f'{v*100:+.3f}%',
                        ha='center', va='center', fontsize=10,
                        color='black')
                ax.text(j, i+0.20, f'({m*100:.1f}%)',
                        ha='center', va='center', fontsize=9,
                        color='dimgray')
 
    cbar = fig.colorbar(im, ax=ax, label='CEV loss (%)')
    plt.tight_layout()
    return fig