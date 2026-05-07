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