import numpy as np

def weighted_gini(y, w):
    """
    Weighted Gini for non-negative y.
    """
    y = np.asarray(y).ravel()
    w = np.asarray(w).ravel()

    mask = np.isfinite(y) & np.isfinite(w) & (w > 0)
    y = y[mask]
    w = w[mask]

    if y.size == 0 or np.sum(w) == 0 or np.sum(w * y) <= 0:
        return np.nan

    order = np.argsort(y)
    y = y[order]
    w = w[order]

    w = w / np.sum(w)
    yw = y * w

    cumw = np.insert(np.cumsum(w), 0, 0)
    cumy = np.insert(np.cumsum(yw) / np.sum(yw), 0, 0)

    return 1 - 2 * np.trapz(cumy, cumw)


def crra_u(c, sigma):
    c = np.maximum(c, 1e-12)
    if sigma == 1:
        return np.log(c)
    else:
        return c**(1-sigma) / (1-sigma)


def crra_u_inv(u, sigma):
    if sigma == 1:
        return np.exp(u)
    else:
        return ((1-sigma) * u)**(1/(1-sigma))


def atkinson_index(c, w, sigma):
    """
    Utility-based inequality measure.
    Uses equally distributed equivalent consumption.
    """
    c = np.asarray(c).ravel()
    w = np.asarray(w).ravel()

    mask = np.isfinite(c) & np.isfinite(w) & (w > 0) & (c > 0)
    c = c[mask]
    w = w[mask]
    w = w / np.sum(w)

    mean_c = np.sum(w * c)
    mean_u = np.sum(w * crra_u(c, sigma))
    ede_c = crra_u_inv(mean_u, sigma)

    return 1 - ede_c / mean_c

def compute_distribution_stats(model, path=None, T_max=50):
    """
    Computes consumption Gini and Atkinson welfare inequality over time.
    """
    if path is None:
        path = model.path

    gini_c = np.empty(T_max)
    atkinson_c = np.empty(T_max)

    for t in range(T_max):
        c_t = path.c[t]
        D_t = path.D[t]

        gini_c[t] = weighted_gini(c_t, D_t)
        atkinson_c[t] = atkinson_index(c_t, D_t, model.par.sigma)

    return gini_c, atkinson_c

"""
welfare_gini.py
===============

Cross-sectional inequality measures for the welfare outputs of the
5-sector I-HANK model:

    v_nodis, v_sec, v_avg   — lifetime utility (three variants)
    ce_nodis, ce_sec, ce_avg — consumption-equivalent welfare (three variants)
    c, a                     — for benchmark consumption / wealth Ginis

Designed for objects produced by GEModelTools where, after find_ss /
find_transition_path:

    model.ss.<var>     has shape (Nfix, Nz, Na)
    model.path.<var>   has shape (T, Nfix, Nz, Na)
    model.ss.D         has shape (Nfix, Nz, Na)
    model.path.D       has shape (T, Nfix, Nz, Na)

Caveats
-------
- The standard Gini is only defined on a non-negative variable. With
  sigma > 1 the lifetime value V is negative everywhere; we still report
  a Gini for V via |V| (which equals the Gini of (-V) on the positive
  reals), but the economically clean object is the Gini of CE. See
  Lucas (1987), Floden (2001, JET), Bhandari, Evans, Golosov & Sargent
  (2021, Econometrica).
- Within-sector dispersion of V is independent of the disutility
  variant because psi_{j,t} is uniform within (j,t) under union
  rationing (Auclert, Rognlie & Straub 2024, JPE). All cross-variant
  differences in the Gini come from the *between*-sector component.
"""

#from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# core Gini routine
# ----------------------------------------------------------------------

def weighted_gini(x: np.ndarray, w: np.ndarray) -> float:
    """ Population-weighted Gini for a non-negative variable x with
    weights w. Uses the Lorenz-curve definition

        G = 1 - 2 * integral_0^1 L(p) dp,

    computed via the trapezoid rule on the sorted cumulative
    distribution. Negative inputs are flagged via NaN.
    """
    x = np.asarray(x, dtype=float).ravel()
    w = np.asarray(w, dtype=float).ravel()

    if x.size != w.size:
        raise ValueError("x and w must have the same flattened size")

    # drop zero-weight cells (they don't contribute and break sorting)
    keep = w > 0.0
    x, w = x[keep], w[keep]
    if x.size == 0:
        return np.nan
    if np.any(x < 0.0):
        return np.nan  # caller decides how to interpret

    order = np.argsort(x)
    x, w = x[order], w[order]

    cw = np.cumsum(w)
    cw /= cw[-1]                      # cumulative population
    cwx = np.cumsum(w * x)
    if cwx[-1] <= 0.0:
        return np.nan
    cwx /= cwx[-1]                    # Lorenz curve L(p)

    # prepend (0,0) so the trapezoid hits the origin
    cw = np.concatenate(([0.0], cw))
    cwx = np.concatenate(([0.0], cwx))

    return float(1.0 - 2.0 * np.trapz(cwx, cw))


def gini_of_neg(x: np.ndarray, w: np.ndarray) -> float:
    """ Gini computed on -x. Useful when x is everywhere negative
    (e.g. lifetime value with sigma>1). Equals the Gini of |x| in
    that case.
    """
    return weighted_gini(-np.asarray(x), w)


# ----------------------------------------------------------------------
# accessors that work for both ss and path
# ----------------------------------------------------------------------

def _get_ss(model, name: str) -> np.ndarray:
    arr = getattr(model.ss, name, None)
    if arr is None:
        raise AttributeError(f"model.ss has no attribute {name!r}")
    return np.asarray(arr)


def _get_path(model, name: str) -> np.ndarray:
    arr = getattr(model.path, name, None)
    if arr is None:
        raise AttributeError(
            f"model.path has no attribute {name!r}. "
            f"Did you run model.find_transition_path()?"
        )
    return np.asarray(arr)


# ----------------------------------------------------------------------
# user-facing functions
# ----------------------------------------------------------------------

WELFARE_VARS = ("v_nodis", "v_sec", "v_avg",
                "ce_nodis", "ce_sec", "ce_avg")


def utility_gini_ss(model, var: str = "ce_nodis") -> float:
    """ Steady-state Gini of `var`. """
    x = _get_ss(model, var)
    D = _get_ss(model, "D")
    if var.startswith("v_"):
        return gini_of_neg(x, D)   # lifetime value: sigma>1 makes it negative
    return weighted_gini(x, D)


def utility_gini_path(model, var: str = "ce_nodis", T: int | None = None) -> np.ndarray:
    """ Gini of `var` at each t along the transition path.

    Returns array of shape (T,).
    """
    x_path = _get_path(model, var)            # (T, Nfix, Nz, Na)
    D_path = _get_path(model, "D")            # (T, Nfix, Nz, Na)

    if x_path.shape != D_path.shape:
        raise ValueError(
            f"shape mismatch: {var}={x_path.shape}, D={D_path.shape}"
        )

    Tmax = x_path.shape[0]
    if T is None:
        T = Tmax
    T = min(T, Tmax)

    fn = gini_of_neg if var.startswith("v_") else weighted_gini
    return np.array([fn(x_path[t], D_path[t]) for t in range(T)])


def gini_table(model,
               vars: tuple[str, ...] = WELFARE_VARS,
               T: int | None = None,
               include_ss: bool = True,
               relative_to_ss: bool = False) -> pd.DataFrame:
    """ One-stop shop: returns a DataFrame indexed by time t with
    one column per variable in `vars`.

    Parameters
    ----------
    vars : tuple of str
        Variable names. Must be in `outputs_hh`.
    T : int or None
        Restrict to the first T periods. None = full path.
    include_ss : bool
        If True, also adds the steady-state Gini as the first row
        (label "ss") and the very last row of the path.
    relative_to_ss : bool
        If True, return Gini_t - Gini_ss (i.e. deviations).
    """
    cols = {}
    ss_row = {}

    for v in vars:
        cols[v]   = utility_gini_path(model, v, T=T)
        ss_row[v] = utility_gini_ss(model, v)

    df = pd.DataFrame(cols)
    df.index.name = "t"

    if relative_to_ss:
        df = df - pd.Series(ss_row)

    if include_ss and not relative_to_ss:
        df = pd.concat([pd.DataFrame(ss_row, index=["ss"]), df])

    return df


# ----------------------------------------------------------------------
# convenience plotting
# ----------------------------------------------------------------------

def plot_gini_path(model,
                   vars: tuple[str, ...] = ("ce_nodis", "ce_sec", "ce_avg"),
                   T: int | None = 60,
                   relative_to_ss: bool = True,
                   ax: plt.Axes | None = None,
                   title: str | None = None) -> plt.Axes:
    """ Plot the Gini of each variable along the transition path.

    By default plots the deviation from the steady-state Gini for
    the three CE variants over the first 60 periods.
    """
    df = gini_table(model, vars=vars, T=T,
                    include_ss=False,
                    relative_to_ss=relative_to_ss)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    for v in df.columns:
        label = {
            "ce_nodis": "CE — no disutility",
            "ce_sec":   "CE — sector-specific disutility",
            "ce_avg":   "CE — average disutility",
            "v_nodis":  "V — no disutility",
            "v_sec":    "V — sector-specific disutility",
            "v_avg":    "V — average disutility",
            "c":        "consumption",
            "a":        "wealth",
        }.get(v, v)
        ax.plot(df.index, df[v], label=label, linewidth=1.6)

    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("quarters since shock")
    ax.set_ylabel(("Gini deviation from ss"
                   if relative_to_ss else "Gini"))
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# ----------------------------------------------------------------------
# decomposition: between-sector vs within-sector
# ----------------------------------------------------------------------

def gini_decomposition(model,
                       var: str = "ce_sec",
                       use_path: bool = False,
                       t: int = 0) -> dict:
    """ Pyatt (1976) / Lambert & Aronson (1993) decomposition of the
    Gini into between-group, within-group, and overlap terms, where
    groups are the five sectors.

    For variants where psi_{j,t} is uniform within sector, the
    within-sector component is identical across variants — so this
    function is the cleanest way to attribute differences between
    `ce_sec`, `ce_avg`, `ce_nodis` to between-sector heterogeneity.
    """
    if use_path:
        x = _get_path(model, var)[t]
        D = _get_path(model, "D")[t]
    else:
        x = _get_ss(model, var)
        D = _get_ss(model, "D")

    Nfix = x.shape[0]
    fn = gini_of_neg if var.startswith("v_") else weighted_gini

    # population shares and within-group means
    pop = D.sum(axis=(1, 2))                     # (Nfix,)
    pop = np.where(pop > 0, pop, 1e-300)

    # within-group means (handle sign for V)
    if var.startswith("v_"):
        means_signed = np.array([
            (x[j] * D[j]).sum() / pop[j] for j in range(Nfix)
        ])
        means = -means_signed
    else:
        means = np.array([
            (x[j] * D[j]).sum() / pop[j] for j in range(Nfix)
        ])

    # within-group Ginis
    G_within = np.zeros(Nfix)
    for j in range(Nfix):
        if pop[j] > 0:
            G_within[j] = fn(x[j], D[j])

    # between-group Gini: treat each group as having mass pop[j]
    # at its group mean
    pop_norm = pop / pop.sum()
    G_between = weighted_gini(means, pop_norm)

    # overall and the residual "overlap" (Pyatt term)
    G_total = fn(x, D)
    grand_mean = (means * pop_norm).sum()
    if grand_mean > 0:
        income_share = pop_norm * means / grand_mean
        G_within_total = float((pop_norm * income_share * G_within).sum())
    else:
        G_within_total = np.nan
    G_overlap = G_total - G_between - G_within_total

    return {
        "var": var,
        "G_total":   G_total,
        "G_between": G_between,
        "G_within":  G_within_total,
        "G_overlap": G_overlap,
        "G_within_by_sector": G_within,
        "pop_shares": pop_norm,
        "group_means": means,
    }

"""
welfare_decomposition.py
========================

Partial-equilibrium "shut-off" decomposition of the welfare / utility
Ginis along the transition path. For each macro channel
(interest rate, tax, real wage, price level, hours) we

  1. override the household-block input path with its steady-state
     value while leaving all other channels at their actual path;
  2. re-solve the household block (backward iteration + forward
     simulation), so that the policies, the distribution, and hence
     the welfare arrays react only to the remaining channels;
  3. compute the Gini path of the chosen welfare variable;
  4. compare to the baseline Gini.

Difference (baseline - shutoff) = contribution of that channel to
the cross-sectional welfare Gini at each t. This is the standard
PE decomposition logic of Kaplan & Violante (2014, ECMA) and
Auclert (2019, AER), applied to the welfare object instead of to
aggregate consumption.

The decomposition is *not* additive: the sum of contributions need
not equal the total Gini change, because the household policy
function is non-linear in the inputs. The residual is reported.

Channels currently implemented:

    'ra'          : real interest rate
    'tax'         : labor income tax rate tau
    'realwage'    : real sectoral wages w_j = W_j/P (one shutoff for all sectors)
    'pricelevel'  : CPI level P (acts on real wages via w_j = W_j/P)
    'nominalwage' : nominal sectoral wages W_j (one shutoff for all sectors)
    'hours'       : sectoral hours N_j (acts on income AND on disutility)
    'income'      : full sector incomes inc_j (lump everything that enters
                    income into one channel, useful as a sanity check)

Sector list is hard-coded for the 5-sector model: HH, HL, LH, LL, NT.

Requires
--------
- A solved transition path (model.find_transition_path was called).
- welfare_gini.py (the module from the previous step) for the
  Gini routines.
"""

#from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import welfare_gini as wg

SECTORS = ("HH", "HL", "LH", "LL", "NT")
DEFAULT_CHANNELS = ("ra", "tax", "realwage", "pricelevel", "hours")


# ----------------------------------------------------------------------
# low-level helpers
# ----------------------------------------------------------------------

def _path(model, name: str) -> np.ndarray:
    arr = getattr(model.path, name, None)
    if arr is None:
        raise AttributeError(
            f"model.path has no attribute {name!r}. "
            f"Did you run model.find_transition_path()?"
        )
    return arr


def _ss(model, name: str):
    val = getattr(model.ss, name, None)
    if val is None:
        raise AttributeError(f"model.ss has no attribute {name!r}")
    return val


def _override_paths(model, overrides: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """ Apply overrides in-place; return originals so they can be
    restored later. """
    saved = {}
    for name, new_val in overrides.items():
        cur = _path(model, name)
        saved[name] = cur.copy()
        cur[:] = new_val
    return saved


def _restore_paths(model, saved: dict[str, np.ndarray]) -> None:
    for name, orig in saved.items():
        _path(model, name)[:] = orig


def _resolve_household(model) -> None:
    """ Re-run the household block: backward-iterate the policies and
    forward-simulate the distribution. Method names match the
    GEModelTools convention; if your version uses a leading underscore
    (e.g. _solve_hh_path), adjust here.
    """
    if hasattr(model, "solve_hh_path"):
        model.solve_hh_path(do_print=False)
    elif hasattr(model, "_solve_hh_path"):
        model._solve_hh_path(do_print=False)
    else:
        raise RuntimeError("Cannot find solve_hh_path on model")

    if hasattr(model, "simulate_hh_path"):
        model.simulate_hh_path(do_print=False)
    elif hasattr(model, "_simulate_hh_path"):
        model._simulate_hh_path(do_print=False)
    else:
        raise RuntimeError("Cannot find simulate_hh_path on model")


# ----------------------------------------------------------------------
# constructing channel shut-offs
# ----------------------------------------------------------------------

def _shutoff_overrides(model, channel: str) -> dict[str, np.ndarray]:
    """ Build the dict of {input_name: counterfactual_path} to feed
    into _override_paths for a given channel.

    All paths returned have the same shape as model.path.<input>.
    """
    par, ss = model.par, model.ss
    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL
    s = {"HH": par.sHH, "HL": par.sHL, "LH": par.sLH, "LL": par.sLL, "NT": sNT}

    if channel == "ra":
        ra = _path(model, "ra")
        return {"ra": np.full_like(ra, ss.ra)}

    if channel == "hours":
        # holding sectoral hours at SS affects:
        #   (i)  N_j as direct input to disutility
        #   (ii) the labor part of inc_j = (1-tau)*w_j*N_j
        out = {}
        for j in SECTORS:
            N_actual = _path(model, f"N{j}")
            N_ss     = _ss(model,   f"N{j}")
            inc_actual = _path(model, f"inc_{j}")
            # rescale income by N_ss / N_actual (linear in N_j)
            scale = N_ss / N_actual
            out[f"N{j}"]    = np.full_like(N_actual, N_ss)
            out[f"inc_{j}"] = inc_actual * scale
        return out

    if channel == "tax":
        tau_actual = _path(model, "tau")
        tau_ss     = _ss(model,   "tau")
        out = {}
        for j in SECTORS:
            inc_actual = _path(model, f"inc_{j}")
            scale = (1.0 - tau_ss) / (1.0 - tau_actual)
            out[f"inc_{j}"] = inc_actual * scale
        return out

    if channel == "realwage":
        # shut off real wages w_j = W_j/P at SS values, simultaneously
        # for all sectors. Income scales by w_ss/w_actual.
        out = {}
        for j in SECTORS:
            w_actual = _path(model, f"w{j}")
            w_ss     = _ss(model,   f"w{j}")
            inc_actual = _path(model, f"inc_{j}")
            scale = w_ss / w_actual
            out[f"inc_{j}"] = inc_actual * scale
        return out

    if channel == "pricelevel":
        # holding P at ss while leaving W_j actual: real wage becomes
        # w_j_cf = W_j / P_ss = w_j_actual * (P_actual / P_ss).
        # Hence inc_cf = inc_actual * (P_actual / P_ss).
        P_actual = _path(model, "P")
        P_ss     = _ss(model,   "P")
        scale = P_actual / P_ss
        out = {}
        for j in SECTORS:
            inc_actual = _path(model, f"inc_{j}")
            out[f"inc_{j}"] = inc_actual * scale
        return out

    if channel == "nominalwage":
        # symmetric to realwage but moves through W_j: hold W_j at SS,
        # P at actual. real wage w_j_cf = W_j_ss / P_actual.
        P_actual = _path(model, "P")
        P_ss     = _ss(model,   "P")
        out = {}
        for j in SECTORS:
            W_actual = _path(model, f"W{j}")
            W_ss     = _ss(model,   f"W{j}")
            inc_actual = _path(model, f"inc_{j}")
            # w_actual = W_actual/P_actual, w_cf = W_ss/P_actual
            scale = W_ss / W_actual
            out[f"inc_{j}"] = inc_actual * scale
        return out

    if channel == "income":
        out = {}
        for j in SECTORS:
            inc_actual = _path(model, f"inc_{j}")
            out[f"inc_{j}"] = np.full_like(inc_actual, _ss(model, f"inc_{j}"))
        return out

    raise ValueError(f"Unknown channel: {channel!r}. "
                     f"Available: {DEFAULT_CHANNELS + ('income','nominalwage')}")


# ----------------------------------------------------------------------
# main API
# ----------------------------------------------------------------------

def gini_under_shutoff(model,
                       var: str,
                       channel: str,
                       T: int | None = None) -> np.ndarray:
    """ Re-solve the household block with `channel` set to its SS value
    and return the resulting Gini path of `var`.
    """
    overrides = _shutoff_overrides(model, channel)
    saved = _override_paths(model, overrides)
    try:
        _resolve_household(model)
        gpath = utility_gini_path(model, var, T=T)
    finally:
        _restore_paths(model, saved)
        # ensure the model is back in the baseline state
        _resolve_household(model)
    return gpath


def decompose_gini(model,
                   var: str,
                   channels: tuple[str, ...] = DEFAULT_CHANNELS,
                   T: int | None = None) -> pd.DataFrame:
    """ Build the full PE decomposition table for one welfare variable.

    Columns
    -------
    'baseline'      : actual Gini path
    '<channel>_off' : Gini path with that channel held at SS
    '<channel>_contrib' : baseline - <channel>_off (contribution at each t)

    The contribution at time t is positive when the channel adds to
    inequality (shutting it off lowers the Gini) and negative when it
    smooths inequality.
    """
    # baseline first — done without any override, just re-solving to be safe
    _resolve_household(model)
    base = utility_gini_path(model, var, T=T)

    cols = {"baseline": base}
    for c in channels:
        gp = gini_under_shutoff(model, var, c, T=T)
        cols[f"{c}_off"] = gp

    df = pd.DataFrame(cols)
    df.index.name = "t"

    for c in channels:
        df[f"{c}_contrib"] = df["baseline"] - df[f"{c}_off"]

    # residual: total deviation from ss not accounted for by channels
    g_ss = utility_gini_ss(model, var)
    total_dev = df["baseline"] - g_ss
    summed_contrib = df[[f"{c}_contrib" for c in channels]].sum(axis=1)
    df["residual"] = total_dev - summed_contrib
    df["total_deviation"] = total_dev

    return df


def decomposition_snapshot(model,
                           var: str,
                           t: int,
                           channels: tuple[str, ...] = DEFAULT_CHANNELS) -> pd.Series:
    """ Single-period decomposition at time t. Useful to make a bar
    chart of "what drives the welfare gini at the peak of the
    response".
    """
    df = decompose_gini(model, var, channels=channels, T=t + 1)
    contribs = df.iloc[t][[f"{c}_contrib" for c in channels]]
    contribs.index = [c.replace("_contrib", "") for c in contribs.index]
    contribs["residual"] = df.iloc[t]["residual"]
    contribs["total_deviation"] = df.iloc[t]["total_deviation"]
    return contribs


def decompose_all_welfare_ginis(model,
                                channels: tuple[str, ...] = DEFAULT_CHANNELS,
                                T: int | None = None,
                                t_snapshot: int | None = None) -> dict:
    """ Run decompose_gini for the full set of welfare variants
    (v_nodis, v_sec, v_avg, ce_nodis, ce_sec, ce_avg).

    Returns dict of {var: DataFrame}. If t_snapshot is given, also
    returns a 'snapshot' DataFrame: rows = welfare variants,
    columns = channels, values = contributions at t_snapshot.
    """
    out = {"paths": {}}
    for v in WELFARE_VARS:
        out["paths"][v] = decompose_gini(model, v, channels=channels, T=T)

    if t_snapshot is not None:
        rows = {}
        for v in WELFARE_VARS:
            rows[v] = decomposition_snapshot(model, v,
                                             t=t_snapshot,
                                             channels=channels)
        out["snapshot"] = pd.DataFrame(rows).T
        out["snapshot"].index.name = "welfare variant"

    return out


# ----------------------------------------------------------------------
# plotting helpers
# ----------------------------------------------------------------------

def plot_decomposition_path(df: pd.DataFrame,
                            channels: tuple[str, ...] = DEFAULT_CHANNELS,
                            ax: plt.Axes | None = None,
                            title: str | None = None) -> plt.Axes:
    """ Stacked-area plot of channel contributions to the Gini deviation
    over time, with the baseline deviation overlaid as a line.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 4.5))

    ts = df.index.values
    contribs = np.array([df[f"{c}_contrib"].values for c in channels])

    # split positive and negative contributions for cleaner stacking
    pos = np.clip(contribs, 0, None)
    neg = np.clip(contribs, None, 0)
    ax.stackplot(ts, pos, labels=channels, alpha=0.85)
    ax.stackplot(ts, neg, alpha=0.85)

    ax.plot(ts, df["total_deviation"], color="black",
            linewidth=1.8, label="total Gini deviation")

    ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--")
    ax.set_xlabel("quarters since shock")
    ax.set_ylabel("Gini deviation from steady state")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def plot_snapshot_bars(snapshot_row: pd.Series,
                       channels: tuple[str, ...] = DEFAULT_CHANNELS,
                       ax: plt.Axes | None = None,
                       title: str | None = None) -> plt.Axes:
    """ Bar chart of channel contributions at one point in time. """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    vals = [snapshot_row[c] for c in channels] + [snapshot_row["residual"]]
    labels = list(channels) + ["residual"]

    colors = ["#3b6ea5"] * len(channels) + ["#a0a0a0"]
    ax.bar(labels, vals, color=colors)
    ax.axhline(0, color="0.3", linewidth=0.7)
    ax.set_ylabel("contribution to Gini deviation")
    ax.set_title(title or "Decomposition snapshot")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# ----------------------------------------------------------------------
# usage sketch
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # from IHANKModel import IHANKModelClass
    # import welfare_decomposition as wd
    #
    # model = IHANKModelClass()
    # model.find_ss(do_print=True)
    # model.compute_jacs()
    # model.find_transition_path(shocks=["Z_us"])
    #
    # # 1) full path decomposition for the sector-specific CE Gini:
    # df = wd.decompose_gini(model, "ce_sec", T=60)
    # print(df.round(5).head(10))
    #
    # # 2) snapshot at peak (e.g. t=8):
    # snap = wd.decomposition_snapshot(model, "ce_sec", t=8)
    # print(snap)
    #
    # # 3) all six welfare variants at once:
    # out = wd.decompose_all_welfare_ginis(model, T=60, t_snapshot=8)
    # print(out["snapshot"].round(5))   # rows: variants, cols: channels
    #
    # # 4) plots:
    # wd.plot_decomposition_path(df, title="CE_sec Gini decomposition")
    # plt.show()
    # wd.plot_snapshot_bars(snap, title="CE_sec Gini at t=8")
    # plt.show()
    pass