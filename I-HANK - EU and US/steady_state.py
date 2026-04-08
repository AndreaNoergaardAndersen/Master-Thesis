# find steady state
import time
import numpy as np
from scipy import optimize

import blocks
from consav import elapsed

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst

def prepare_hh_ss(model):
    """ prepare the household block for finding the steady state """

    par = model.par
    ss = model.ss

    sNT = 1.0 - par.sHH - par.sHL

    ##################################
    # 1. grids and transition matrix #
    ##################################

    par.a_grid[:] = equilogspace(par.a_min, par.a_max, par.Na)
    par.z_grid[:], ss.z_trans[:,:,:], e_ergodic, _, _ = log_rouwenhorst(par.rho_z, par.sigma_psi, n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################

    for i_fix in range(par.Nfix):
        if i_fix == 0:    # TH-High
            ss.Dbeg[i_fix,:,0] = e_ergodic * par.sHH
        elif i_fix == 1:  # TH-Low
            ss.Dbeg[i_fix,:,0] = e_ergodic * par.sHL
        else:             # NT
            ss.Dbeg[i_fix,:,0] = e_ergodic * sNT
        ss.Dbeg[i_fix,:,1:] = 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix, par.Nz, par.Na))

    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):
            z = par.z_grid[i_z]
            if i_fix == 0:
                inc = ss.inc_HH / par.sHH * z
            elif i_fix == 1:
                inc = ss.inc_HL / par.sHL * z
            else:
                inc = ss.inc_NT / sNT * z
            c = (1+ss.ra) * par.a_grid + inc
            v_a[i_fix,i_z,:] = c**(-par.sigma)
        ss.vbeg_a[i_fix] = ss.z_trans[i_fix] @ v_a[i_fix]


def evaluate_ss(model, do_print=False):
    """ evaluate steady state """

    par = model.par
    ss = model.ss

    sNT = 1.0 - par.sHH - par.sHL

    ss.beta = par.beta

    # EU NK steady state
    ss.C_eu = 1.0
    ss.N_eu = 1.0
    ss.mc_eu = 1.0
    ss.i_eu = par.i_eu_ss
    ss.Z_eu = 1.0
    ss.rF_eu = par.i_eu_ss

    ss.eu_Euler_res = 0.0
    ss.eu_LS_res = 0.0
    ss.eu_NKPC_res = 0.0
    ss.eu_TR_res = 0.0
    ss.eu_RC_res = 0.0
    ss.i_shock_eu = 0.0

    # US NK steady state
    ss.C_us = 1.0
    ss.N_us = 1.0
    ss.mc_us = 1.0
    ss.i_us = par.i_us_ss
    ss.Z_us = 1.0
    ss.rF_us = par.i_us_ss

    ss.us_Euler_res = 0.0
    ss.us_LS_res = 0.0
    ss.us_NKPC_res = 0.0
    ss.us_TR_res = 0.0
    ss.us_RC_res = 0.0
    ss.i_shock_us = 0.0

    # normalize prices/exchange rates to 1 in SS
    for varname in ['PF_eu_s', 'E', 'PTH_eu_s', 'Q', 'PF_eu',
                    'Q_us', 'PF_us_s', 'E_us', 'PF_us', 'PTH_us_s', 'CB_us',
                    'PF_TF', 'PTH', 'PT', 'PNT', 'P',
                    # sector output prices and nominal wages (= 1 in SS)
                    'PHH', 'PHL', 'WHH', 'WHL', 'WNT',
                    # EU material prices
                    'PM_eu_eu', 'PM_eu_us', 'PM_eu', 'M_eu', 'M_eu_eu', 'M_eu_us',
                    # US material prices
                    'PM_us_us', 'PM_us_eu', 'PM_us', 'M_us', 'M_us_eu', 'M_us_us',
                    # DK shared material price components (PM_dk_h, PM_dk_l = 1 when both = 1)
                    'PM_dk_eu', 'PM_dk_us', 'PM_dk_h', 'PM_dk_l',
                    ]:
        ss.__dict__[varname] = 1.0

    # zero inflation in SS
    for varname in ['pi_F_eu_s', 'pi_F_eu', 'pi_TH_eu_s', 'pi_eu', 'piM_eu_eu',
                    'pi_F_us_s', 'pi_F_us', 'pi_TH_us_s', 'pi_us', 'piM_us_us',
                    'pi_FF', 'pi_TH', 'pi_T', 'pi_NT', 'pi',
                    'pi_PHH', 'pi_PHL',
                    'piWHH', 'piWHL', 'piWNT',
                    ]:
        ss.__dict__[varname] = 0.0

    # real and nominal interest rates equal to EU steady-state rate
    ss.ra = ss.r = ss.i = ss.rF_eu = par.i_eu_ss
    ss.rF_us = par.i_us_ss
    ss.UIP_res = 0.0
    ss.UIP_res_us = 0.0
    ss.i_shock = 0.0

    # tariffs = 0 in SS
    ss.tau_m = 0.0
    ss.tau_x = 0.0

    # ---- EU materials steady state ----
    ss.PM_eu_us = ss.PM_us_us * ss.E_us / ss.E
    ss.PM_eu = blocks.price_index(ss.PM_eu_us, ss.PM_eu_eu, par.eta_M_eu, par.alpha_M_eu_us)
    ss.W_eu = ss.PF_eu_s
    ss.M_eu = ss.N_eu * (par.beta_M_eu / (1.0 - par.beta_M_eu)) * ((ss.W_eu/ss.PF_eu_s) / (ss.PM_eu/ss.PF_eu_s))**par.eta_VA_eu
    ss.M_eu_us = par.alpha_M_eu_us * (ss.PM_eu_us / ss.PM_eu)**(-par.eta_M_eu) * ss.M_eu
    ss.M_eu_eu = (1.0 - par.alpha_M_eu_us) * (ss.PM_eu_eu / ss.PM_eu)**(-par.eta_M_eu) * ss.M_eu
    rho_eu = (par.eta_VA_eu - 1.0) / par.eta_VA_eu
    ss.Y_eu = ss.Z_eu * (((1.0 - par.beta_M_eu)**(1.0/par.eta_VA_eu) * ss.N_eu**rho_eu
                          + par.beta_M_eu**(1.0/par.eta_VA_eu) * ss.M_eu**rho_eu) ** (1.0 / rho_eu))
    ss.C_eu = ss.Y_eu - (ss.PM_eu / ss.PF_eu_s) * ss.M_eu
    par.varphi_eu = (ss.W_eu / ss.PF_eu_s) * ss.C_eu**(-par.sigma_eu) / (ss.N_eu**par.nu_eu)

    # ---- US materials steady state ----
    ss.PM_us_eu = ss.PM_eu_eu * ss.E / ss.E_us
    ss.PM_us = blocks.price_index(ss.PM_us_us, ss.PM_us_eu, par.eta_M_us, par.alpha_M_us_us)
    ss.W_us = ss.PF_us_s
    ss.M_us = ss.N_us * (par.beta_M_us / (1.0 - par.beta_M_us)) * ((ss.W_us/ss.PF_us_s) / (ss.PM_us/ss.PF_us_s))**par.eta_VA_us
    ss.M_us_us = par.alpha_M_us_us * (ss.PM_us_us / ss.PM_us)**(-par.eta_M_us) * ss.M_us
    ss.M_us_eu = (1.0 - par.alpha_M_us_us) * (ss.PM_us_eu / ss.PM_us)**(-par.eta_M_us) * ss.M_us
    rho_us = (par.eta_VA_us - 1.0) / par.eta_VA_us
    ss.Y_us = ss.Z_us * (((1.0 - par.beta_M_us)**(1.0/par.eta_VA_us) * ss.N_us**rho_us
                          + par.beta_M_us**(1.0/par.eta_VA_us) * ss.M_us**rho_us) ** (1.0 / rho_us))
    ss.C_us = ss.Y_us - (ss.PM_us / ss.PF_us_s) * ss.M_us
    par.varphi_us = (ss.W_us / ss.PF_us_s) * ss.C_us**(-par.sigma_us) / (ss.N_us**par.nu_us)

    # ---- DK production: NT sector ----
    ss.ZNT = 1.0
    ss.NNT = sNT
    ss.YNT = ss.ZNT * ss.NNT
    ss.WNT = ss.PNT * ss.ZNT     # = 1 in SS
    ss.wNT = ss.WNT / ss.P       # = 1 in SS

    # shared CES exponents for DK tradeable sectors
    pow_dk = 1.0 - par.eta_VA_dk
    rho_dk = (par.eta_VA_dk - 1.0) / par.eta_VA_dk

    # ---- DK production: TH-High sector ----
    ss.ZTH = 1.0
    ss.NHH = par.sHH

    ss.PM_dk_h = blocks.price_index(ss.PM_dk_us, ss.PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_h)

    # nominal wage from unit-cost inversion (PHH = ZTH = PM_dk_h = 1 → WHH = 1)
    rhs_h = ((ss.PHH * ss.ZTH)**pow_dk - par.beta_M_dk_h * ss.PM_dk_h**pow_dk) / (1.0 - par.beta_M_dk_h)
    ss.WHH = rhs_h ** (1.0 / pow_dk)
    ss.wHH = ss.WHH / ss.P

    ss.M_dk_h = ss.NHH * (par.beta_M_dk_h / (1.0 - par.beta_M_dk_h)) * (ss.WHH / ss.PM_dk_h)**par.eta_VA_dk
    ss.M_dk_us_h = par.alpha_M_dk_us_h * (ss.PM_dk_us / ss.PM_dk_h)**(-par.eta_M_dk) * ss.M_dk_h
    ss.M_dk_eu_h = (1.0 - par.alpha_M_dk_us_h) * (ss.PM_dk_eu / ss.PM_dk_h)**(-par.eta_M_dk) * ss.M_dk_h

    ss.YHH = ss.ZTH * (((1.0 - par.beta_M_dk_h)**(1.0/par.eta_VA_dk) * ss.NHH**rho_dk
                        + par.beta_M_dk_h**(1.0/par.eta_VA_dk) * ss.M_dk_h**rho_dk) ** (1.0 / rho_dk))

    # ---- DK production: TH-Low sector ----
    ss.NHL = par.sHL

    ss.PM_dk_l = blocks.price_index(ss.PM_dk_us, ss.PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_l)

    rhs_l = ((ss.PHL * ss.ZTH)**pow_dk - par.beta_M_dk_l * ss.PM_dk_l**pow_dk) / (1.0 - par.beta_M_dk_l)
    ss.WHL = rhs_l ** (1.0 / pow_dk)
    ss.wHL = ss.WHL / ss.P

    ss.M_dk_l = ss.NHL * (par.beta_M_dk_l / (1.0 - par.beta_M_dk_l)) * (ss.WHL / ss.PM_dk_l)**par.eta_VA_dk
    ss.M_dk_us_l = par.alpha_M_dk_us_l * (ss.PM_dk_us / ss.PM_dk_l)**(-par.eta_M_dk) * ss.M_dk_l
    ss.M_dk_eu_l = (1.0 - par.alpha_M_dk_us_l) * (ss.PM_dk_eu / ss.PM_dk_l)**(-par.eta_M_dk) * ss.M_dk_l

    ss.YHL = ss.ZTH * (((1.0 - par.beta_M_dk_l)**(1.0/par.eta_VA_dk) * ss.NHL**rho_dk
                        + par.beta_M_dk_l**(1.0/par.eta_VA_dk) * ss.M_dk_l**rho_dk) ** (1.0 / rho_dk))

    # ---- Calibrate inner flat-CES weight from SS gross outputs ----
    par.omega_TH_H = ss.YHH / (ss.YHH + ss.YHL)

    # ---- Household income ----
    ss.tau = par.tau_ss
    ss.inc_HH = (1.0 - ss.tau) * ss.wHH * ss.NHH
    ss.inc_HL = (1.0 - ss.tau) * ss.wHL * ss.NHL
    ss.inc_NT = (1.0 - ss.tau) * ss.wNT * ss.NNT

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # ---- Government ----
    ss.B = ss.A_hh
    ss.G = ss.tau * (ss.wHH*ss.NHH + ss.wHL*ss.NHL + ss.wNT*ss.NNT) - ss.r * ss.B

    # Monetary policy
    if par.float:
        ss.CB = ss.E
    else:
        ss.CB = ss.i
    ss.CB_us = ss.E_us

    # ---- Consumption ----
    # outer nest: calibrate alphaT from NT market clearing
    par.alphaT = 1.0 - (ss.YNT - ss.G) / ss.C_hh

    ss.CT  = par.alphaT * ss.C_hh
    ss.CNT = (1.0 - par.alphaT) * ss.C_hh

    # middle nest: home vs foreign
    ss.CTH = (1.0 - par.alphaF) * ss.CT
    ss.CTF = par.alphaF * ss.CT

    # inner flat-CES: TH-High vs TH-Low (PHH = PHL = PTH = 1 in SS)
    ss.CTH_H = par.omega_TH_H * ss.CTH
    ss.CTH_L = (1.0 - par.omega_TH_H) * ss.CTH

    # foreign split: EU vs US
    ss.CTF_eu = (1.0 - par.alpha_us) * ss.CTF
    ss.CTF_us = par.alpha_us * ss.CTF

    # total exports and split across EU and US
    X_tot = ss.YHH + ss.YHL - ss.CTH
    ss.CTH_eu_s = (1.0 - par.share_X_us) * X_tot
    ss.CTH_us_s = par.share_X_us * X_tot

    # sector-specific export splits (same flat-CES shares as domestic, PTH=PHH=PHL=1)
    ss.CTH_HH_eu_s = par.omega_TH_H * ss.CTH_eu_s
    ss.CTH_HL_eu_s = (1.0 - par.omega_TH_H) * ss.CTH_eu_s
    ss.CTH_HH_us_s = par.omega_TH_H * ss.CTH_us_s
    ss.CTH_HL_us_s = (1.0 - par.omega_TH_H) * ss.CTH_us_s

    ss.M_eu_s = ss.CTH_eu_s
    ss.M_us_s = ss.CTH_us_s
    par.M_eu_s_ss = ss.M_eu_s
    par.M_us_s_ss = ss.M_us_s

    # ---- Market clearing (should be zero in SS) ----
    ss.clearing_YHH = ss.YHH - ss.CTH_H - ss.CTH_HH_eu_s - ss.CTH_HH_us_s
    ss.clearing_YHL = ss.YHL - ss.CTH_L - ss.CTH_HL_eu_s - ss.CTH_HL_us_s
    ss.clearing_YNT = ss.YNT - ss.CNT - ss.G

    # ---- Accounting ----
    ss.GDP = (ss.PHH*ss.YHH - ss.PM_dk_h*ss.M_dk_h
              + ss.PHL*ss.YHL - ss.PM_dk_l*ss.M_dk_l
              + ss.PNT*ss.YNT) / ss.P
    ss.NX  = ss.GDP - ss.C_hh - ss.G
    ss.NFA = ss.A_hh - ss.B
    ss.CA  = ss.NX + ss.ra * ss.NFA
    ss.Walras = ss.CA

    # ---- Labor disutility parameters from SS NKWCs ----
    par.varphiHH = (1.0/par.muw) * (1.0-ss.tau) * ss.wHH * ss.UC_HH_hh / ((ss.NHH/par.sHH)**par.nu)
    par.varphiHL = (1.0/par.muw) * (1.0-ss.tau) * ss.wHL * ss.UC_HL_hh / ((ss.NHL/par.sHL)**par.nu)
    par.varphiNT = (1.0/par.muw) * (1.0-ss.tau) * ss.wNT * ss.UC_NT_hh / ((ss.NNT/sNT)**par.nu)

    ss.NKWCHH_res = 0.0
    ss.NKWCHL_res = 0.0
    ss.NKWCNT_res = 0.0


def find_ss(model, do_print=False):
    """ find the steady state """

    par = model.par
    ss = model.ss

    t0 = time.time()
    evaluate_ss(model, do_print=do_print)

    if do_print:
        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.inc_HH = :.3f}')
        print(f'{ss.inc_HL = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.alphaT = :.3f}')
        print(f'{par.alphaF = :.3f}')
        print(f'{par.omega_TH_H = :.3f}')
        print(f'{par.varphiHH = :.3f}')
        print(f'{par.varphiHL = :.3f}')
        print(f'{par.varphiNT = :.3f}')
        print(f'{ss.YHH = :.3f}')
        print(f'{ss.YHL = :.3f}')
        print(f'{ss.YNT = :.3f}')
        print(f'{ss.M_eu_s = :.3f}')
        print(f'{ss.M_us_s = :.3f}')
        print(f'{ss.clearing_YHH = :12.8f}')
        print(f'{ss.clearing_YHL = :12.8f}')
        print(f'{ss.clearing_YNT = :12.8f}')
        print(f'{ss.G = :.3f}')
        print(f'{ss.NFA = :.3f}')
        print(f'{ss.CB = :.3f}')
        print(f'{ss.CB_us = :.3f}')
