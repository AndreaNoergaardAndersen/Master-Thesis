# welfare_decomp_plots.py
# =================================================================
# Bar-chart plotting for the channel-decomposition of welfare loss.
#
# Two figures:
#   1. Aggregate channel bars: one bar per channel + total + residual
#   2. Channel-by-group bars: grouped bars by wealth (or sector),
#      with one cluster per channel
#
# Interpretation note (important for the caption):
# Counterfactual channel contributions DO NOT add to the total
# exactly because of nonlinear interactions (the household problem
# is concave). We report a "residual" bar = total - sum(channels).
# A small residual (relative to total) means the linear decomposition
# is a good approximation; a large residual flags interaction effects.
# =================================================================
 
import numpy as np
import matplotlib.pyplot as plt
 
 
_CHANNEL_LABELS = {
    'ra'  : 'Interest (ra)',
    'inc' : 'Income (inc)',
    'N'   : 'Labor disutility (N)',
}
_CHANNEL_COLORS = {
    'ra'  : '#1f77b4',
    'inc' : '#d62728',
    'N'   : '#2ca02c',
}

from thesis_graph_style import *

set_thesis_style(use_latex=False)
make_graph_folders()
 
def plot_aggregate_decomposition(pop_losses, title=None, figsize=(9, 5.5)):
    """
    Bar chart of population CEV loss by channel.
 
    Parameters
    ----------
    pop_losses : dict
        Must contain keys: 'full', 'ra_only', 'inc_only', 'N_only'.
        Optional: 'no_shock' (for sanity check; should be ~0).
        All values are CEV losses as fractions (e.g. 0.008 = 0.80%).
 
    Output bars (left to right):
        Full transition
        Interest channel (ra_only)
        Income channel (inc_only)
        Labor disutility channel (N_only)
        Sum of channels (= ra + inc + N)
        Residual (= full - sum of channels)
    """
    bars = [
        ('Full transition',      pop_losses['full'],       'black'),
        ('Interest only (ra)',   pop_losses['ra_only'],    _CHANNEL_COLORS['ra']),
        ('Income only (inc)',    pop_losses['inc_only'],   _CHANNEL_COLORS['inc']),
        ('Labor only (N)',       pop_losses['N_only'],     _CHANNEL_COLORS['N']),
    ]
    channel_sum = (pop_losses['ra_only']
                   + pop_losses['inc_only']
                   + pop_losses['N_only'])
    residual    = pop_losses['full'] - channel_sum
    bars.append(('Sum of channels', channel_sum, 'gray'))
    bars.append(('Residual (interaction)', residual,  '#ff7f0e'))
 
    labels = [b[0] for b in bars]
    vals   = np.array([b[1] for b in bars]) * 100
    colors = [b[2] for b in bars]
 
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(labels))
    bar_artists = ax.bar(xs, vals, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Population CEV loss (%)')
    ax.set_title(title if title else
                 'Channel decomposition of 20-period CEV loss')
    for b, v in zip(bar_artists, vals):
        ax.annotate(f'{v:+.3f}%',
                    (b.get_x() + b.get_width()/2, v),
                    ha='center',
                    va='bottom' if v >= 0 else 'top',
                    fontsize=10)
    plt.tight_layout()
    return fig
 
 
def plot_group_decomposition(group_losses, group_order=None,
                             channels=('ra_only','inc_only','N_only','full'),
                             title=None, figsize=(11, 6)):
    """
    Grouped bar chart of CEV losses by (channel, group). Used for
    the wealth-group or sector decomposition figures.
 
    Parameters
    ----------
    group_losses : dict of dict
        group_losses[scenario_name][group_label] = mean loss (fraction)
        Output of welfare_decomp_calc.group_losses_by_mask.
    group_order : list[str] or None
        Order of groups along the x-axis. Defaults to natural dict order.
    channels : tuple of str
        Which scenarios to plot side-by-side per group. Include 'full'
        to show the full-transition baseline alongside the channels.
    """
    if group_order is None:
        first_scen = next(iter(group_losses))
        group_order = list(group_losses[first_scen].keys())
 
    n_groups   = len(group_order)
    n_channels = len(channels)
    bar_w = 0.8 / n_channels
 
    label_map = {'full': 'Full', 'ra_only': 'Interest',
                 'inc_only': 'Income', 'N_only': 'Labor'}
    color_map = {'full': 'black', 'ra_only': _CHANNEL_COLORS['ra'],
                 'inc_only': _CHANNEL_COLORS['inc'],
                 'N_only':  _CHANNEL_COLORS['N']}
 
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(n_groups)
    for k, scen in enumerate(channels):
        vals = np.array([group_losses[scen][g] for g in group_order]) * 100
        offset = (k - (n_channels-1)/2) * bar_w
        ax.bar(xs + offset, vals, width=bar_w,
               label=label_map.get(scen, scen),
               color=color_map.get(scen, 'gray'),
               edgecolor='white')
 
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(group_order)
    ax.set_ylabel('CEV loss (%)')
    ax.set_title(title if title else 'CEV loss by group and channel')
    ax.legend(loc='best', fontsize=11)
    plt.tight_layout()
    return fig
 
 
# -----------------------------------------------------------------
# Sub-decomposition bar chart for the income channel
# -----------------------------------------------------------------
 
_SUB_LABELS = {
    'tau': 'Tax ($\\tau$)',
    'W'  : 'Nominal wage ($W$)',
    'N'  : 'Employment ($N$)',
    'P'  : 'Price level ($P$)',
}
_SUB_COLORS = {
    'tau': '#9467bd',
    'W'  : '#e377c2',
    'N'  : '#8c564b',
    'P'  : '#17becf',
}
 
 
def plot_income_sub_decomposition(pop_inc_total, pop_sub_losses,
                                  title=None, figsize=(10, 5.5)):
    """
    Bar chart showing the income channel split into (tau, W, N, P).
 
    Parameters
    ----------
    pop_inc_total : float
        Population CEV loss under the full inc-only scenario
        (the +1.049% bar from the top-level decomposition).
    pop_sub_losses : dict {sub -> CEV loss as fraction}
        Output of aggregate_population_losses applied to the dict
        returned by run_income_sub_decomposition.
    """
    bars = [('Income only (total)', pop_inc_total, '#d62728')]
    for sub in ('tau', 'W', 'N', 'P'):
        bars.append((_SUB_LABELS[sub], pop_sub_losses[sub],
                     _SUB_COLORS[sub]))
    sub_sum  = sum(pop_sub_losses[s] for s in ('tau', 'W', 'N', 'P'))
    residual = pop_inc_total - sub_sum
    bars.append(('Sum of sub-channels', sub_sum, 'gray'))
    bars.append(('Residual', residual, '#ff7f0e'))
 
    labels = [b[0] for b in bars]
    vals   = np.array([b[1] for b in bars]) * 100
    colors = [b[2] for b in bars]
 
    fig, ax = plt.subplots(figsize=figsize)
    xs = np.arange(len(labels))
    bar_artists = ax.bar(xs, vals, color=colors, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha='right')
    ax.set_ylabel('Population CEV loss (%)')
    ax.set_title(title if title else
                 'Sub-decomposition of the income channel')
    for b, v in zip(bar_artists, vals):
        ax.annotate(f'{v:+.3f}%',
                    (b.get_x() + b.get_width()/2, v),
                    ha='center',
                    va='bottom' if v >= 0 else 'top', fontsize=10)
    plt.tight_layout()
    return fig
 
 
# =================================================================
# Stacked-diverging decomposition by group (sector or wealth)
# =================================================================
 
# Five resource-side segment colors (ra + 4 income sub-channels)
_RESOURCE_SEG = [
    ('ra'  , 'Interest ($r^a$)',  '#1f77b4'),
    ('tau' , 'Tax ($\\tau$)',     '#9467bd'),
    ('W'   , 'Nominal wage ($W$)','#e377c2'),
    ('N'   , 'Employment ($N$)',  '#8c564b'),
    ('P'   , 'Price level ($P$)', '#17becf'),
]
 
 
def plot_stacked_by_group(group_loss_full, group_loss_segments,
                          group_loss_disutility=None,
                          group_order=None, segments=_RESOURCE_SEG,
                          title=None, figsize=(11, 7),
                          variant_label=None):
    """
    Two-panel diverging stacked-bar decomposition by group.
 
    Top panel: resource-side decomposition (5 segments). Positive
    contributions stack upward from zero, negative downward, with
    a black dot marking the resource-side total per group.
 
    Bottom panel: labor-disutility offset by group, plotted as a
    separate bar. None of these segments enter the resource stack.
 
    Parameters
    ----------
    group_loss_full : dict {group_label -> full-transition loss (fraction)}
        For the dot markers showing the headline welfare loss.
    group_loss_segments : dict
        group_loss_segments[segment_key][group_label] = mean loss
        Keys must include 'ra','tau','W','N','P'. Each is the mean
        CEV loss under the scenario where ONLY that input moves.
    group_loss_disutility : dict {group_label -> labor-disutility loss}
        or None. If provided, drawn in the bottom panel.
    group_order : list of group labels for x-axis ordering.
    segments : list of (key, label, color) — defaults to the five
        resource segments.
    title : str
    variant_label : str   appended to the title to indicate
                          'nodis' / 'avg' / 'sec' welfare variant.
    """
    if group_order is None:
        group_order = list(group_loss_full.keys())
    n_groups = len(group_order)
 
    if group_loss_disutility is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize,
                                       gridspec_kw={'height_ratios':[3,1]})
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None
 
    xs = np.arange(n_groups)
 
    # ---- top panel: diverging stack of resource-side segments ----
    pos_cum = np.zeros(n_groups)
    neg_cum = np.zeros(n_groups)
    legend_handles = []
    for key, label, color in segments:
        vals = np.array([group_loss_segments[key][g] for g in group_order]) * 100
        pos = np.where(vals > 0, vals, 0.0)
        neg = np.where(vals < 0, vals, 0.0)
        b1 = ax1.bar(xs, pos, bottom=pos_cum, color=color,
                     edgecolor='white', label=label)
        ax1.bar(xs, neg, bottom=neg_cum, color=color, edgecolor='white')
        pos_cum = pos_cum + pos
        neg_cum = neg_cum + neg
        legend_handles.append(b1)
 
    # Resource-side total = sum of segments (= pos_cum + neg_cum)
    res_total = pos_cum + neg_cum
    ax1.scatter(xs, res_total, color='black', s=60, zorder=5,
                marker='_', linewidths=2.5, label='Resource total (sum)')
 
    # Full-transition total dot
    full_total = np.array([group_loss_full[g] for g in group_order]) * 100
    ax1.scatter(xs, full_total, color='black', s=70, zorder=6,
                marker='o', label='Full-transition total')
 
    ax1.axhline(0, color='black', lw=0.8, alpha=0.6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(group_order)
    ax1.set_ylabel('Resource-side CEV loss (%)')
    base_title = title if title else 'Resource decomposition by group'
    if variant_label is not None:
        base_title = f'{base_title}  [variant: {variant_label}]'
    ax1.set_title(base_title)
    ax1.legend(loc='best', fontsize=9, ncol=2)
 
    # ---- bottom panel: labor-disutility offset ----
    if ax2 is not None:
        dis_vals = np.array([group_loss_disutility[g] for g in group_order]) * 100
        colors = ['seagreen' if v < 0 else 'firebrick' for v in dis_vals]
        ax2.bar(xs, dis_vals, color=colors, edgecolor='white')
        ax2.axhline(0, color='black', lw=0.8, alpha=0.6)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(group_order)
        ax2.set_ylabel('Labour-disutility (%)')
        ax2.set_title('Preference-flow offset (separate from resource decomposition)')
        for x, v in zip(xs, dis_vals):
            ax2.annotate(f'{v:+.3f}%', (x, v), ha='center',
                         va='bottom' if v >= 0 else 'top', fontsize=9)
 
    plt.tight_layout()
    return fig