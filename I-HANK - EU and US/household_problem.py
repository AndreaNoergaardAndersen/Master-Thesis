# solving the household problem
#
# Adds, on top of the EGM consumption-saving problem:
#   - flow utility u(c)
#   - lifetime value functions V (three variants)
#       v_nodis : no labor disutility
#       v_sec   : sector-specific varphi_j (heterogeneous parameter and hours)
#       v_avg   : population-weighted average varphi_avg (homogeneous parameter, heterogeneous hours)
#   - constant-consumption-equivalent (CE) lifetime welfare for each variant
#
# Hours per worker within sector j follow the union-rationing convention of
# Auclert, Rognlie & Straub (2024, JPE): h_j_t = N_j_t / s_j, common to all
# (z,a). Per-period disutility is GHH/KPR-style:
#       psi_j_t = varphi_j * (N_j_t/s_j)^(1+nu) / (1+nu).
# This is uniform within sector j at time t, so it does not affect the
# household policies; it only enters the value functions.

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec


# ----------------------------------------------------------------------
# small numba helpers
# ----------------------------------------------------------------------

@nb.njit
def utility(c, sigma):
    """ flow utility from consumption (CRRA, log if sigma=1) """
    c_safe = np.maximum(c, 1e-14)
    if np.abs(sigma - 1.0) < 1e-10:
        return np.log(c_safe)
    else:
        return c_safe ** (1.0 - sigma) / (1.0 - sigma)


@nb.njit
def ce_from_value(v, beta, sigma):
    """ constant-consumption-equivalent welfare:
            v = u(ce) / (1-beta)   =>   ce
        Note: with sigma>1 and v<0, inside is positive.
    """
    if np.abs(sigma - 1.0) < 1e-10:
        return np.exp((1.0 - beta) * v)
    else:
        inside = (1.0 - beta) * (1.0 - sigma) * v
        inside = np.maximum(inside, 1e-300)
        return inside ** (1.0 / (1.0 - sigma))


# ----------------------------------------------------------------------
# main household block
# ----------------------------------------------------------------------

@nb.njit
def solve_hh_backwards(par, z_trans, beta, ra,
                       inc_HH, inc_HL, inc_LH, inc_LL, inc_NT,
                       NHH, NHL, NLH, NLL, NNT,
                       vbeg_a_plus,
                       vbeg_nodis_plus, vbeg_sec_plus, vbeg_avg_plus,
                       vbeg_a,
                       vbeg_nodis, vbeg_sec, vbeg_avg,
                       a, c,
                       uc_HH, uc_HL, uc_LH, uc_LL, uc_NT,
                       c_HH, c_HL, c_LH, c_LL, c_NT,
                       u,
                       v_nodis, v_sec, v_avg,
                       ce_nodis, ce_sec, ce_avg):
    """
    Backward step.

    Sectors:
        i_fix=0 -> HH    1 -> HL    2 -> LH    3 -> LL    4 -> NT
    """

    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL

    # ==================================================================
    # 1. EGM: policy functions a', c, and vbeg_a
    # ==================================================================

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            if i_fix == 0:
                inc = inc_HH / par.sHH
            elif i_fix == 1:
                inc = inc_HL / par.sHL
            elif i_fix == 2:
                inc = inc_LH / par.sLH
            elif i_fix == 3:
                inc = inc_LL / par.sLL
            else:
                inc = inc_NT / sNT

            z = par.z_grid[i_z]

            # endogenous grid step
            c_endo = (beta * vbeg_a_plus[i_fix, i_z]) ** (-1.0 / par.sigma)
            m_endo = c_endo + par.a_grid

            # interpolation onto exogenous grid
            m = (1.0 + ra) * par.a_grid + inc * z
            interp_1d_vec(m_endo, par.a_grid, m, a[i_fix, i_z])
            a[i_fix, i_z, :] = np.fmax(a[i_fix, i_z, :], 0.0)
            c[i_fix, i_z, :] = m - a[i_fix, i_z, :]

        # marginal value at the start of next period (expectation over z')
        v_a = (1.0 + ra) * c[i_fix] ** (-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix] @ v_a

    # ==================================================================
    # 2. Lifetime value functions and CE welfare
    # ==================================================================

    # 2.a per-period disutility shifters psi_j (one number per sector)
    # h_j = N_j / s_j is the union-rationed hours per worker
    h0 = NHH / par.sHH
    h1 = NHL / par.sHL
    h2 = NLH / par.sLH
    h3 = NLL / par.sLL
    h4 = NNT / sNT

    psi_sec = np.empty(par.Nfix)
    psi_sec[0] = par.varphiHH * h0 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_sec[1] = par.varphiHL * h1 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_sec[2] = par.varphiLH * h2 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_sec[3] = par.varphiLL * h3 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_sec[4] = par.varphiNT * h4 ** (1.0 + par.nu) / (1.0 + par.nu)

    psi_avg = np.empty(par.Nfix)
    psi_avg[0] = par.varphi_avg * h0 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_avg[1] = par.varphi_avg * h1 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_avg[2] = par.varphi_avg * h2 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_avg[3] = par.varphi_avg * h3 ** (1.0 + par.nu) / (1.0 + par.nu)
    psi_avg[4] = par.varphi_avg * h4 ** (1.0 + par.nu) / (1.0 + par.nu)

    # 2.b Bellman steps
    cont_nodis = np.empty(par.Na)
    cont_sec   = np.empty(par.Na)
    cont_avg   = np.empty(par.Na)

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            # continuation values evaluated at chosen a'
            interp_1d_vec(par.a_grid, vbeg_nodis_plus[i_fix, i_z, :],
                          a[i_fix, i_z, :], cont_nodis)
            interp_1d_vec(par.a_grid, vbeg_sec_plus[i_fix, i_z, :],
                          a[i_fix, i_z, :], cont_sec)
            interp_1d_vec(par.a_grid, vbeg_avg_plus[i_fix, i_z, :],
                          a[i_fix, i_z, :], cont_avg)

            # current flow utility from consumption
            u_now = utility(c[i_fix, i_z, :], par.sigma)
            u[i_fix, i_z, :] = u_now

            # lifetime values (Bellman): three variants
            v_nodis[i_fix, i_z, :] = u_now                  + beta * cont_nodis
            v_sec[i_fix, i_z, :]   = u_now - psi_sec[i_fix] + beta * cont_sec
            v_avg[i_fix, i_z, :]   = u_now - psi_avg[i_fix] + beta * cont_avg

            # consumption-equivalent welfare for each variant
            ce_nodis[i_fix, i_z, :] = ce_from_value(
                v_nodis[i_fix, i_z, :], beta, par.sigma)
            ce_sec[i_fix, i_z, :] = ce_from_value(
                v_sec[i_fix, i_z, :], beta, par.sigma)
            ce_avg[i_fix, i_z, :] = ce_from_value(
                v_avg[i_fix, i_z, :], beta, par.sigma)

        # expectation over z' for the start of next period
        vbeg_nodis[i_fix] = z_trans[i_fix] @ v_nodis[i_fix]
        vbeg_sec[i_fix]   = z_trans[i_fix] @ v_sec[i_fix]
        vbeg_avg[i_fix]   = z_trans[i_fix] @ v_avg[i_fix]

    # ==================================================================
    # 3. Sector-specific consumption / marginal utility outputs
    # ==================================================================

    uc_HH[:] = 0.0
    uc_HL[:] = 0.0
    uc_LH[:] = 0.0
    uc_LL[:] = 0.0
    uc_NT[:] = 0.0

    c_HH[:] = 0.0
    c_HL[:] = 0.0
    c_LH[:] = 0.0
    c_LL[:] = 0.0
    c_NT[:] = 0.0

    for i_z in range(par.Nz):
        uc_HH[0, i_z, :] = c[0, i_z, :] ** (-par.sigma) * par.z_grid[i_z]
        uc_HL[1, i_z, :] = c[1, i_z, :] ** (-par.sigma) * par.z_grid[i_z]
        uc_LH[2, i_z, :] = c[2, i_z, :] ** (-par.sigma) * par.z_grid[i_z]
        uc_LL[3, i_z, :] = c[3, i_z, :] ** (-par.sigma) * par.z_grid[i_z]
        uc_NT[4, i_z, :] = c[4, i_z, :] ** (-par.sigma) * par.z_grid[i_z]

        c_HH[0, i_z, :] = c[0, i_z, :]
        c_HL[1, i_z, :] = c[1, i_z, :]
        c_LH[2, i_z, :] = c[2, i_z, :]
        c_LL[3, i_z, :] = c[3, i_z, :]
        c_NT[4, i_z, :] = c[4, i_z, :]

    uc_HH[:] /= par.sHH
    uc_HL[:] /= par.sHL
    uc_LH[:] /= par.sLH
    uc_LL[:] /= par.sLL
    uc_NT[:] /= sNT

    c_HH[:] /= par.sHH
    c_HL[:] /= par.sHL
    c_LH[:] /= par.sLH
    c_LL[:] /= par.sLL
    c_NT[:] /= sNT
