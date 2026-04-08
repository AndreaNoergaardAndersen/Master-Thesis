# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,beta,ra,inc_HH,inc_HL,inc_NT,
                       vbeg_a_plus,vbeg_a,a,c,
                       uc_HH,uc_HL,uc_NT,c_HH,c_HL,c_NT):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus)

    Sectors:  i_fix=0 → TH-High,  i_fix=1 → TH-Low,  i_fix=2 → NT
    """

    sNT = 1.0 - par.sHH - par.sHL

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            # income per worker in this sector (total sector income / sector size)
            if i_fix == 0:
                inc = inc_HH / par.sHH
            elif i_fix == 1:
                inc = inc_HL / par.sHL
            else:
                inc = inc_NT / sNT

            z = par.z_grid[i_z]

            # EGM
            c_endo = (beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            m_endo = c_endo + par.a_grid

            # interpolate onto fixed grid
            m = (1+ra)*par.a_grid + inc*z
            interp_1d_vec(m_endo,par.a_grid,m,a[i_fix,i_z])
            a[i_fix,i_z,:] = np.fmax(a[i_fix,i_z,:],0.0)
            c[i_fix,i_z] = m - a[i_fix,i_z]

        # expectation step
        v_a = (1+ra)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a

    # ---- extra outputs (sector-specific MU of consumption and consumption) ----
    # Zero out all, then fill only the relevant i_fix slice for each output.
    # After distribution aggregation, UC_HH_hh = E[uc_HH] over all households
    # (non-zero only for i_fix=0), divided by par.sHH to get per-worker average.

    uc_HH[:] = 0.0
    uc_HL[:] = 0.0
    uc_NT[:] = 0.0
    c_HH[:] = 0.0
    c_HL[:] = 0.0
    c_NT[:] = 0.0

    for i_z in range(par.Nz):
        uc_HH[0,i_z,:] = c[0,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_HL[1,i_z,:] = c[1,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_NT[2,i_z,:] = c[2,i_z,:]**(-par.sigma)*par.z_grid[i_z]

        c_HH[0,i_z,:] = c[0,i_z,:]
        c_HL[1,i_z,:] = c[1,i_z,:]
        c_NT[2,i_z,:] = c[2,i_z,:]

    # normalise by sector size → per-worker averages used in NKW Phillips curves
    uc_HH[:] /= par.sHH
    uc_HL[:] /= par.sHL
    uc_NT[:] /= sNT

    c_HH[:] /= par.sHH
    c_HL[:] /= par.sHL
    c_NT[:] /= sNT
