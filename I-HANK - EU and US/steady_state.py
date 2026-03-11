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
    ss.x_eu=0.0 #Output gap is 0 in SS 
    ss.pi_eu=0.0 #no inflation in SS
    ss.i_eu=par.i_eu_ss
    ss.rn_eu=par.rn_eu_ss
    ss.i_shock_eu=0.0

    # --- EU production normalizations (needed to avoid NaNs later) ---
    ss.Z_eu = 1.0          # EU productivity (must be > 0)
    ss.Y_eu = 1.0          # EU output level normalization

    # US NK steady state
    ss.x_us = 0.0
    ss.pi_us = 0.0
    ss.i_us = par.i_us_ss
    ss.rn_us = par.rn_us_ss
    ss.i_shock_us = 0.0

    # US production normalizations
    ss.Z_us = 1.0
    ss.Y_us = 1.0

    # normalzied to 1
    #for varname in ['PF_eu_s','E', 'PF','PTH','PT','PNT','P','PTH_eu_s','Q']:
    #    ss.__dict__[varname] = 1.0

    for varname in ['PF_eu_s', 'E','PTH_eu_s','Q', 'PF_eu', #EU
                'Q_us', 'PF_us_s','E_us','PF_us','PTH_us_s', 'CB_us', #US
                'PF_TF', 'PTH','PT','PNT','P', #General
                ]:
        ss.__dict__[varname] = 1.0
    
    # zero inflation
#    for varname in ['pi_F_eu_s','pi_F','pi_TH','pi_T','pi_NT','pi','pi_TH_eu_s','piWTH','piWNT','x_eu','pi_eu']:
#        ss.__dict__[varname] = 0.0
    
    for varname in ['pi_F_eu_s','pi_F_eu','pi_TH_eu_s','x_eu','pi_eu', #EU
                    'pi_F_us_s','pi_F_us','pi_TH_us_s','x_us','pi_us', #US
                    'pi_FF','pi_TH','pi_T','pi_NT','pi','piWTH','piWNT' #General
                    ]:
        ss.__dict__[varname] = 0.0
    

    # real+nominal interest rates are equal to foreign interest rate
    ss.ra = ss.r  = ss.i = ss.rF_eu = par.rn_eu_ss
    ss.rF_us = par.rn_us_ss #WHY this
    ss.UIP_res = 0.0
    ss.UIP_res_us = 0.0
    # domestic interes rate shock:
    ss.i_shock = 0.0

    #EU NK residuals in SS
    ss.eu_IS_res=0.0
    ss.eu_NKPC_res=0.0
    ss.eu_TR_res=0.0

    #US NK residuals in SS
    ss.us_IS_res = 0.0
    ss.us_NKPC_res = 0.0
    ss.us_TR_res = 0.0

    # b. production

    # normalize TFP and labor
    ss.ZTH = 1.0
    ss.ZNT = 1.0
    ss.NTH = 1.0*par.sT
    ss.NNT = 1.0*(1-par.sT)
    
    # production
    ss.YTH = ss.ZTH*ss.NTH
    ss.YNT = ss.ZNT*ss.NNT

    # real = nominal wages = value of mpl
    ss.wTH = ss.WTH = ss.PTH*ss.ZTH
    ss.wNT = ss.WNT = ss.PNT*ss.ZNT
    
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

    # --- EU exports/consumption and production objects in SS ---
    #ss.X_eu_to_dk = ss.CTF              # EU exports to DK equal DK imports (foreign good)
    #ss.C_eu = ss.Y_eu - ss.X_eu_to_dk   # EU consumes the residual
    ss.N_eu = ss.Y_eu / ss.Z_eu         # production: Y = Z*N
    ss.N_us = ss.Y_us / ss.Z_us

    # EU real marginal cost (if your EU NKPC uses mc_eu)
    # Requires par.W_eu_ss in IHANKModel.setup(), e.g. par.W_eu_ss = 1.0
    ss.mc_eu = (par.W_eu_ss / ss.Z_eu) / ss.PF_eu_s
    ss.mc_us = (par.W_us_ss / ss.Z_us) / ss.PF_us_s

    # size of foreign market
    # Total exports of DK tradable good
    X_tot = ss.YTH - ss.CTH

    # Split across EU and US
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
