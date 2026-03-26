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

    ##################################
    # 1. grids and transition matrix #
    ##################################

    # b. a
    par.a_grid[:] = equilogspace(par.a_min,par.a_max,par.Na)

    # c. z
    par.z_grid[:],ss.z_trans[:,:,:],e_ergodic,_,_ = log_rouwenhorst(par.rho_z,par.sigma_psi,n=par.Nz)

    ###########################
    # 2. initial distribution #
    ###########################
    
    for i_fix in range(par.Nfix):
        
        if i_fix == 0:
            ss.Dbeg[i_fix,:,0] = e_ergodic*par.sT
        elif i_fix == 1:
            ss.Dbeg[i_fix,:,0] = e_ergodic*(1-par.sT)
        else:
            raise NotImplementedError('i_fix must be 0 or 1')
        
        ss.Dbeg[i_fix,:,1:] = 0.0    

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    v_a = np.zeros((par.Nfix,par.Nz,par.Na))
    
    for i_fix in range(par.Nfix):
        for i_z in range(par.Nz):

            z = par.z_grid[i_z]

            if i_fix == 0:
                inc = ss.inc_TH/par.sT*z
            elif i_fix == 1:
                inc = ss.inc_NT/(1-par.sT)*z

            c = (1+ss.ra)*par.a_grid + inc
            v_a[i_fix,i_z,:] = c**(-par.sigma)

            ss.vbeg_a[i_fix] = ss.z_trans[i_fix]@v_a[i_fix]
        
def evaluate_ss(model,do_print=False):
    """ evaluate steady state"""

    par = model.par
    ss = model.ss

    ss.beta = par.beta
    
    #EU NK steady state
    ss.C_eu = 1.0
    ss.N_eu = 1.0
    ss.mc_eu = 1.0
    ss.i_eu = par.i_eu_ss
    ss.Z_eu = 1.0
    ss.rF_eu = par.i_eu_ss
    
    #EU NK residuals in SS
    ss.eu_Euler_res=0.0
    ss.eu_LS_res=0.0
    ss.eu_NKPC_res=0.0
    ss.eu_TR_res=0.0
    ss.eu_RC_res=0.0

    ss.i_shock_eu = 0.0

    # US NK steady state
    ss.C_us = 1.0
    ss.N_us = 1.0
    ss.mc_us = 1.0
    ss.i_us = par.i_us_ss
    ss.Z_us = 1.0
    ss.rF_us = par.i_us_ss

    #US NK residuals in SS
    ss.us_Euler_res=0.0
    ss.us_LS_res=0.0
    ss.us_NKPC_res=0.0
    ss.us_TR_res=0.0
    ss.us_RC_res=0.0

    ss.i_shock_us = 0.0

    # normalzied to 1
#    for varname in ['PF_eu_s', 'E','PTH_eu_s','Q', 'PF_eu', #EU
#                'Q_us', 'PF_us_s','E_us','PF_us','PTH_us_s', 'CB_us', #US
#                'PF_TF', 'PTH','PT','PNT','P', #General
#                ]:
#        ss.__dict__[varname] = 1.0
    for varname in ['PF_eu_s', 'E','PTH_eu_s','Q', 'PF_eu',#EU
                'Q_us', 'PF_us_s','E_us','PF_us','PTH_us_s', 'CB_us',  #US
                'PF_TF', 'PTH','PT','PNT','P', #General
                'PM_eu_eu', 'PM_eu_us','PM_eu','M_eu','M_eu_eu','M_eu_us', #EU materials
                'PM_us_us', 'PM_us_eu','PM_us','M_us','M_us_eu','M_us_us', #US materials
                'PM_dk_eu', 'PM_dk_us','PM_dk','M_dk','M_dk_eu','M_dk_us'  #DK materials
                ]:
        ss.__dict__[varname] = 1.0
    
    # zero inflation
    for varname in ['pi_F_eu_s','pi_F_eu','pi_TH_eu_s','pi_eu', 'piM_eu_eu', #EU
                    'pi_F_us_s','pi_F_us','pi_TH_us_s','pi_us', 'piM_us_us', #US
                    'pi_FF','pi_TH','pi_T','pi_NT','pi','piWTH','piWNT' #General
                    ]:
        ss.__dict__[varname] = 0.0
    

    # real+nominal interest rates are equal to foreign interest rate
    ss.ra = ss.r  = ss.i = ss.rF_eu = par.i_eu_ss
    ss.rF_us = par.i_us_ss
    ss.UIP_res = 0.0
    ss.UIP_res_us = 0.0
    # domestic interes rate shock:
    ss.i_shock = 0.0

    # EU materials steady state (nested CES in production)
    ss.PM_eu_us = ss.PM_us_us * ss.E_us / ss.E
    ss.PM_eu = blocks.price_index(ss.PM_eu_us, ss.PM_eu_eu, par.eta_M_eu, par.alpha_M_eu_us)
    ss.W_eu = ss.PF_eu_s
    ss.M_eu = ss.N_eu * (par.beta_M_eu / (1.0 - par.beta_M_eu)) * ((ss.W_eu/ss.PF_eu_s) / (ss.PM_eu/ss.PF_eu_s))**par.eta_VA_eu
    ss.M_eu_us = par.alpha_M_eu_us * (ss.PM_eu_us / ss.PM_eu)**(-par.eta_M_eu) * ss.M_eu
    ss.M_eu_eu = (1.0 - par.alpha_M_eu_us) * (ss.PM_eu_eu / ss.PM_eu)**(-par.eta_M_eu) * ss.M_eu
    rho_eu = (par.eta_VA_eu - 1.0) / par.eta_VA_eu
    ss.Y_eu = ss.Z_eu * (((1.0 - par.beta_M_eu) * ss.N_eu**rho_eu + par.beta_M_eu * ss.M_eu**rho_eu) ** (1.0 / rho_eu))
    ss.C_eu = ss.Y_eu - (ss.PM_eu / ss.PF_eu_s) * ss.M_eu
    par.varphi_eu = (ss.W_eu / ss.PF_eu_s) * ss.C_eu**(-par.sigma_eu) / (ss.N_eu**par.nu_eu)

    # US materials steady state (nested CES in production)
    ss.PM_us_eu = ss.PM_eu_eu * ss.E / ss.E_us
    ss.PM_us = blocks.price_index(ss.PM_us_us, ss.PM_us_eu, par.eta_M_us, par.alpha_M_us_us)
    ss.W_us = ss.PF_us_s
    ss.M_us = ss.N_us * (par.beta_M_us / (1.0 - par.beta_M_us)) * ((ss.W_us/ss.PF_us_s) / (ss.PM_us/ss.PF_us_s))**par.eta_VA_us
    ss.M_us_us = par.alpha_M_us_us * (ss.PM_us_us / ss.PM_us)**(-par.eta_M_us) * ss.M_us
    ss.M_us_eu = (1.0 - par.alpha_M_us_us) * (ss.PM_us_eu / ss.PM_us)**(-par.eta_M_us) * ss.M_us
    rho_us = (par.eta_VA_us - 1.0) / par.eta_VA_us
    ss.Y_us = ss.Z_us * (((1.0 - par.beta_M_us) * ss.N_us**rho_us + par.beta_M_us * ss.M_us**rho_us) ** (1.0 / rho_us))
    ss.C_us = ss.Y_us - (ss.PM_us / ss.PF_us_s) * ss.M_us
    par.varphi_us = (ss.W_us / ss.PF_us_s) * ss.C_us**(-par.sigma_us) / (ss.N_us**par.nu_us)

    # b. production

    #Non-tradable sector
    ss.ZNT = 1.0
    ss.NNT = 1.0*(1-par.sT)
    ss.YNT = ss.ZNT*ss.NNT
    ss.wNT = ss.WNT = ss.PNT*ss.ZNT

    # normalize TFP and labor
    ss.ZTH = 1.0
    ss.NTH = 1.0*par.sT
    # production
    #ss.YTH = ss.ZTH*ss.NTH
    #ss.wTH = ss.WTH = ss.PTH*ss.ZTH
    
    #DK materials steady state (nested CES in production)
    ss.PM_dk_eu = ss.PM_eu_eu * ss.E
    ss.PM_dk_us = ss.PM_us_us * ss.E_us
    ss.PM_dk = blocks.price_index(ss.PM_dk_us, ss.PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us)

    # real = nominal wages = value of mpl
    pow_dk = 1.0 - par.eta_VA_dk
    rhs = ((ss.PTH * ss.ZTH)**pow_dk - par.beta_M_dk * ss.PM_dk**pow_dk) / (1.0 - par.beta_M_dk)
    ss.WTH = rhs ** (1.0 / pow_dk)        # nominal wage from unit-cost inversion
    ss.wTH = ss.WTH / ss.P                # real wage (= WTH since P = 1 in SS)

    ss.M_dk = ss.NTH * (par.beta_M_dk / (1.0 - par.beta_M_dk)) * ((ss.WTH/ss.PTH) / (ss.PM_dk/ss.PTH))**par.eta_VA_dk
    ss.M_dk_us = par.alpha_M_dk_us * (ss.PM_dk_us / ss.PM_dk)**(-par.eta_M_dk) * ss.M_dk
    ss.M_dk_eu = (1.0 - par.alpha_M_dk_us) * (ss.PM_dk_eu / ss.PM_dk)**(-par.eta_M_dk) * ss.M_dk

    rho_dk = (par.eta_VA_dk - 1.0) / par.eta_VA_dk
    ss.YTH = ss.ZTH * (((1.0 - par.beta_M_dk) * ss.NTH**rho_dk + par.beta_M_dk * ss.M_dk**rho_dk) ** (1.0 / rho_dk))

    # c. household 
    ss.tau = par.tau_ss
    ss.inc_TH = (1-ss.tau)*ss.wTH*ss.NTH
    ss.inc_NT = (1-ss.tau)*ss.wNT*ss.NNT

    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # d. government
    ss.B = ss.A_hh
    ss.G = ss.tau*(ss.wTH*ss.NTH+ss.wNT*ss.NNT)-ss.r*ss.B

    #Monetary policy
    if par.float == True:
        ss.CB = ss.E 
    else:
        ss.CB = ss.i
    
    ss.CB_us=ss.E_us
    
    # e. consumption

    # tradeables vs. non-tradeables
    par.alphaT = 1-(ss.YNT-ss.G)/ss.C_hh # clearing_NT

    ss.CT = par.alphaT*ss.C_hh 
    ss.CNT = (1-par.alphaT)*ss.C_hh

    # home vs. foreign
    ss.CTH = (1-par.alphaF)*ss.CT
    ss.CTF = par.alphaF*ss.CT
    
    #SS import split
    ss.CTF_us = par.alpha_us * ss.CTF
    ss.CTF_eu = (1-par.alpha_us) * ss.CTF

    # size of foreign market
    # Total exports of DK tradable good
    X_tot = ss.YTH - ss.CTH

    # Export split across EU and US
    ss.CTH_eu_s = (1-par.share_X_us) * X_tot
    ss.CTH_us_s = par.share_X_us * X_tot

    # Market sizes (relative prices normalized to 1 in SS)
    ss.M_eu_s = ss.CTH_eu_s
    ss.M_us_s = ss.CTH_us_s

    par.M_eu_s_ss = ss.M_eu_s
    par.M_us_s_ss = ss.M_us_s

    # f. market clearing
    ss.clearing_YTH = ss.YTH - ss.CTH - ss.CTH_eu_s - ss.CTH_us_s 
    ss.clearing_YNT = ss.YNT - ss.CNT - ss.G

    # zero net foreign assets
    ss.NFA = ss.A_hh - ss.B

    # zero net foreign assets
    ss.GDP = ss.YTH + ss.YNT
    ss.NX = ss.GDP - ss.C_hh - ss.G
    ss.NFA = ss.A_hh - ss.B
    ss.CA = ss.NX + ss.ra*ss.NFA
    ss.Walras = ss.CA

    # g. disutility of labor for NKWPCs
    par.varphiTH = 1/par.muw*(1-ss.tau)*ss.wTH*ss.UC_TH_hh / ((ss.NTH/par.sT)**par.nu)
    par.varphiNT = 1/par.muw*(1-ss.tau)*ss.wNT*ss.UC_NT_hh / ((ss.NNT/(1-par.sT))**par.nu)

    ss.NKWCT_res = 0.0
    ss.NKWCNT_res = 0.0

def find_ss(model, do_print=False): 
    """ find the steady state """

    par = model.par
    ss = model.ss

    # a. find steady state
    t0 = time.time()

    evaluate_ss(model,do_print=do_print)

    # b. print
    if do_print:

        print(f'steady state found in {elapsed(t0)}')
        print(f'{ss.inc_TH = :.3f}')
        print(f'{ss.inc_NT = :.3f}')
        print(f'{par.alphaT = :.3f}')
        print(f'{par.alphaF = :.3f}')
        print(f'{par.varphiTH = :.3f}')
        print(f'{par.varphiNT = :.3f}')
        print(f'{ss.M_eu_s = :.3f}')
        print(f'{ss.M_us_s = :.3f}')
        print(f'{ss.clearing_YTH = :12.8f}')
        print(f'{ss.clearing_YNT = :12.8f}')
        print(f'{ss.G = :.3f}')
        print(f'{ss.NFA = :.3f}')
        print(f'{ss.CB = :.3f}')
        print(f'{ss.CB_us = :.3f}')
