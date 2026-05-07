# solving the household problem

import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit
def solve_hh_backwards(par,z_trans,beta,ra,inc_HH,inc_HL,inc_LH,inc_LL,inc_NT,
                       vbeg_a_plus,vbeg_a,a,c,
                       uc_HH,uc_HL,uc_LH,uc_LL,uc_NT,c_HH,c_HL,c_LH,c_LL,c_NT):
    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus)

    Sectors:  i_fix=0 -> HH (high mat, high US exp)
              i_fix=1 -> HL (high mat, low US exp)
              i_fix=2 -> LH (low mat, high US exp)
              i_fix=3 -> LL (low mat, low US exp)
              i_fix=4 -> NT
    """

    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL

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

    uc_HH[:] = 0.0
    uc_HL[:] = 0.0
    uc_LH[:] = 0.0
    uc_LL[:] = 0.0
    uc_NT[:] = 0.0
    c_HH[:]  = 0.0
    c_HL[:]  = 0.0
    c_LH[:]  = 0.0
    c_LL[:]  = 0.0
    c_NT[:]  = 0.0

    for i_z in range(par.Nz):
        uc_HH[0,i_z,:] = c[0,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_HL[1,i_z,:] = c[1,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_LH[2,i_z,:] = c[2,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_LL[3,i_z,:] = c[3,i_z,:]**(-par.sigma)*par.z_grid[i_z]
        uc_NT[4,i_z,:] = c[4,i_z,:]**(-par.sigma)*par.z_grid[i_z]

        c_HH[0,i_z,:] = c[0,i_z,:]
        c_HL[1,i_z,:] = c[1,i_z,:]
        c_LH[2,i_z,:] = c[2,i_z,:]
        c_LL[3,i_z,:] = c[3,i_z,:]
        c_NT[4,i_z,:] = c[4,i_z,:]

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
