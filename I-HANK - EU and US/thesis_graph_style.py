# thesis_graph_style.py

from pathlib import Path
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 1. Paths
# ============================================================

from pathlib import Path


# ============================================================
# 1. Paths
# ============================================================

def find_project_root(start=None, markers=("README.md", ".git")):
    """
    Find the local VS Code project/repository folder by walking upwards.

    This does NOT commit anything to GitHub.
    It only helps save graphs inside your local project folder.
    """
    start = Path.cwd() if start is None else Path(start)
    start = start.resolve()

    for path in [start] + list(start.parents):
        if any((path / marker).exists() for marker in markers):
            return path

    # Fallback: save relative to the current notebook/script location
    return start


PROJECT_ROOT = find_project_root()
FINAL_GRAPHS = PROJECT_ROOT / "Final graphs"


GRAPH_FOLDERS = {
    "shocks": FINAL_GRAPHS / "Shocks",
    "macroeconomic_effects": FINAL_GRAPHS / "Macroeconomic effects",
    "aggregate_welfare": FINAL_GRAPHS / "Aggregate welfare",
    "inequality": FINAL_GRAPHS / "Inequality",
    "sensitivity_analysis": FINAL_GRAPHS / "Sensitivity analysis",
    "appendix": FINAL_GRAPHS / "Appendix",
}


def make_graph_folders():
    """Create all main graph folders locally."""
    for folder in GRAPH_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def graph_dir(section, *subfolders):
    """
    Return the folder path for a graph section and create it if needed.

    Example:
        graph_dir("shocks")
        graph_dir("sensitivity_analysis", "eta_I_f")
    """
    if section not in GRAPH_FOLDERS:
        valid = ", ".join(GRAPH_FOLDERS.keys())
        raise ValueError(f"Unknown graph section '{section}'. Valid sections are: {valid}")

    folder = GRAPH_FOLDERS[section]

    for subfolder in subfolders:
        folder = folder / subfolder

    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_fig(
    fig,
    section,
    filename,
    *subfolders,
    formats=("pdf", "png"),
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
):
    """
    Save figure locally inside:
        <VS Code project>/Final graphs/<section>/<subfolders>/

    This does not add, commit, or push anything to Git.
    """
    folder = graph_dir(section, *subfolders)

    filename = Path(filename)
    stem = filename.stem

    saved_paths = []

    for fmt in formats:
        outpath = folder / f"{stem}.{fmt}"
        fig.savefig(
            outpath,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
        )
        saved_paths.append(outpath)

    print("Saved figure to:")
    for path in saved_paths:
        print(f"  {path}")

    return saved_paths


# ============================================================
# 2. Thesis colour palette
# ============================================================

BLUE1 = "#1c3897"
BLUE2 = "#2040ac"
BLUE3 = "#5373df"
BLUE4 = "#a9b9ef"

RED1 = "#f5796d"
RED2 = "#b81118"

BLACK = "#000000"
GREY = "#7a7a7a"
LIGHT_GREY = "#d9d9d9"

COLORS = {
    "blue1": BLUE1,
    "blue2": BLUE2,
    "blue3": BLUE3,
    "blue4": BLUE4,
    "red1": RED1,
    "red2": RED2,
    "black": BLACK,
    "grey": GREY,
    "light_grey": LIGHT_GREY,
}

SHOCK_COLORS = {
    "tau_x": RED2,
    "tau_m": BLUE2,
    "tau_war": BLACK,
    r"$\tau^x$": RED2,
    r"$\tau^m$": BLUE2,
    r"$\tau^m + \tau^x$": BLACK,
}

HOUSEHOLD_INPUT_COLORS = {
    "interest_rate": BLUE2,
    "tax_rate": BLUE3,
    "real_wage": RED1,
    "employment": RED2,
}

SECTOR_COLORS = {
    "HH": RED2,
    "LH": RED1,
    "HL": BLUE2,
    "LL": BLUE3,
    "NT": BLUE4,
}

SENSITIVITY_BASE_COLORS = [BLUE4, BLUE3, BLUE2, BLUE1]


def sensitivity_palette(n, reverse=False):
    """
    Blue gradient for sensitivity analyses.
    Default: low values = light blue, high values = dark blue.
    """
    cmap = LinearSegmentedColormap.from_list(
        "thesis_blue_gradient",
        [BLUE4, BLUE3, BLUE2, BLUE1],
        N=max(n, 2),
    )
    colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

    if reverse:
        colors = colors[::-1]

    return colors


CEV_LOSS_CMAP = LinearSegmentedColormap.from_list(
    "cev_loss_cmap",
    [BLUE4, RED1, RED2],
    N=256,
)


# ============================================================
# 3. Matplotlib thesis style
# ============================================================

SPINE_WIDTH = 0.8
BASELINE_WIDTH = 0.8
LINE_WIDTH = 1.7


def set_thesis_style(use_latex=False):
    """
    Apply common thesis graph style.

    use_latex=False is safer in Jupyter/VS Code.
    use_latex=True matches LaTeX more closely, but requires a working LaTeX installation.
    """
    mpl.rcParams.update({
        # Font
        "font.family": "serif",
        "font.serif": [
            "Computer Modern Roman",
            "Latin Modern Roman",
            "CMU Serif",
            "STIXGeneral",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "cm",
        "text.usetex": use_latex,

        # No grid
        "axes.grid": False,

        # Axis frame
        "axes.linewidth": SPINE_WIDTH,
        "axes.edgecolor": BLACK,

        # Text sizes
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9,

        # Lines
        "lines.linewidth": LINE_WIDTH,

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": SPINE_WIDTH,
        "ytick.major.width": SPINE_WIDTH,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        # Legend
        "legend.frameon": False,

        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })

    if use_latex:
        mpl.rcParams.update({
            "text.latex.preamble": r"\usepackage{lmodern}",
        })


# Standard figure sizes
FIGSIZE_SINGLE = (5.2, 3.1)
FIGSIZE_1X2 = (7.0, 2.9)
FIGSIZE_1X3 = (9.5, 2.8)
FIGSIZE_2X2 = (7.0, 5.4)
FIGSIZE_HEATMAP = (6.6, 4.2)


# ============================================================
# 4. Axis formatting helpers
# ============================================================

def years_grid(T, periods_per_year=4):
    """Convert model periods to years."""
    return np.arange(T) / periods_per_year


def pct_dev(path, ss):
    """Percentage deviation from steady state."""
    return 100 * (np.asarray(path) / ss - 1)


def pp_dev(path, ss):
    """
    Percentage-point deviation from steady state.
    Use for variables already measured as rates in decimal units.
    """
    return 100 * (np.asarray(path) - ss)


def format_axis(
    ax,
    ylabel=None,
    xlabel="Years",
    baseline=0.0,
    xlim=(0, 5),
    xticks=(1, 2, 3, 4, 5),
    spine_width=SPINE_WIDTH,
):
    """
    Apply thesis axis layout:
    - no grid
    - one baseline / steady-state line
    - x-axis in years
    - consistent axis edge thickness
    - no title
    """
    ax.grid(False)
    ax.set_title("")

    if baseline is not None:
        ax.axhline(
            baseline,
            color=GREY,
            linewidth=BASELINE_WIDTH,
            linestyle="-",
            zorder=0,
        )

    ax.set_xlim(*xlim)
    ax.set_xticks(list(xticks))
    ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)
        spine.set_color(BLACK)

    return ax


def format_axes(axes, **kwargs):
    """Apply format_axis to one or many axes."""
    axes = np.ravel(axes)
    for ax in axes:
        format_axis(ax, **kwargs)


def common_ylim_from_arrays(*arrays, pad=0.08, include_zero=True, symmetric=False):
    """
    Compute a common y-axis range from arrays.
    Useful when multiple panels should have the same y-axis range.
    """
    vals = []
    for arr in arrays:
        arr = np.asarray(arr).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)

    if not vals:
        return None

    vals = np.concatenate(vals)

    if include_zero:
        vals = np.concatenate([vals, np.array([0.0])])

    ymin, ymax = np.min(vals), np.max(vals)

    if symmetric:
        m = max(abs(ymin), abs(ymax))
        ymin, ymax = -m, m

    span = ymax - ymin
    if np.isclose(span, 0):
        span = 1.0 if np.isclose(ymax, 0) else abs(ymax)

    return ymin - pad * span, ymax + pad * span


def set_common_ylim(axes, ylim):
    """Set same y-limit for one or many axes."""
    if ylim is None:
        return

    for ax in np.ravel(axes):
        ax.set_ylim(*ylim)


# ============================================================
# 5. Legend helpers
# ============================================================

def collect_legend_entries(axes):
    """Collect unique legend entries across axes."""
    handles, labels = [], []
    seen = set()

    for ax in np.ravel(axes):
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and not label.startswith("_") and label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)

    return handles, labels


def add_bottom_legend(fig, axes, ncol=None, y=-0.02):
    """
    Add one shared, wide legend below the graph/panels.
    Best for multi-panel figures with repeated lines.
    """
    handles, labels = collect_legend_entries(axes)

    if not handles:
        return None

    if ncol is None:
        ncol = len(labels)

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
    )

    # Make room for legend
    fig.subplots_adjust(bottom=0.20)

    return legend


def remove_legends(axes):
    """Remove legends from individual panels."""
    for ax in np.ravel(axes):
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


# ============================================================
# 6. Plot helper
# ============================================================

def plot_line(
    ax,
    x,
    y,
    label=None,
    color=BLACK,
    linestyle="-",
    linewidth=LINE_WIDTH,
    alpha=1.0,
    **kwargs,
):
    """Small wrapper around ax.plot with thesis defaults."""
    return ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        **kwargs,
    )