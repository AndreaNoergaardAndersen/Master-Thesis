import numpy as np
import numba as nb
from GEModelTools import prev, next, lag, lead, isclose
from GEModelTools import lag, lead

##############
## Helpers ##
##############

@nb.njit
def price_index(P1,P2,eta,alpha):
    return (alpha*P1**(1-eta)+(1-alpha)*P2**(1-eta))**(1/(1-eta))

@nb.njit
def inflation_from_price(P,inival):

    P_lag = lag(inival,P) 
    pi = P/P_lag - 1

    return pi

@nb.njit
def price_from_inflation(P,pi,T,iniP):

    for t in range(T):
        if t == 0:
            P[t] = iniP*(1+pi[t]) 
        else:
            P[t] = P[t-1]*(1+pi[t]) 
   
############
## Blocks ##
############
@nb.njit
def mon_pol(par,ini,ss,E,CB, E_us, CB_us):

    if par.float == True:
        E[:] = CB 
    else:
        E[:] = ss.E
    E_us[:]=CB_us

@nb.njit
def material_prices(par, ini, ss,
                       piM_eu_eu, piM_us_us,
                       E, E_us,
                       PM_eu_eu, PM_us_us, PM_eu_us, PM_eu, PM_us_eu, PM_us,
                       PM_dk_eu, PM_dk_us, PM_dk
                       ):
    """
    Separate material-input prices for the EU production block.
    """

    # EU-sourced material price in EUR
    price_from_inflation(PM_eu_eu, piM_eu_eu, par.T, ss.PM_eu_eu)

    # US-sourced material price in USD
    price_from_inflation(PM_us_us, piM_us_us, par.T, ss.PM_us_us)

    # convert to EUR
    PM_eu_us[:] = PM_us_us * E_us / E # add tariff somewhere around here

    # convert to USD for US production block    
    PM_us_eu[:] = PM_eu_eu * E / E_us

    # inner CES price index
    PM_eu[:] = price_index(PM_eu_us, PM_eu_eu, par.eta_M_eu, par.alpha_M_eu_us)

    # inner CES price index
    PM_us[:] = price_index(PM_us_us, PM_us_eu, par.eta_M_us, par.alpha_M_us_us)

    #DK materials price index for production
    PM_dk_eu[:] = PM_eu_eu * E
    PM_dk_us[:] = PM_us_us * E_us
    PM_dk[:] = price_index(PM_dk_us, PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us)

@nb.njit
def eu_nk(par, ini, ss,
          Z_eu, i_shock_eu,
          Y_eu, C_eu, N_eu, pi_eu, i_eu,
          PF_eu_s, rF_eu, M_eu_s, mc_eu, W_eu,
          PM_eu_eu, PM_eu_us, PM_eu, M_eu, M_eu_eu, M_eu_us,
          eu_Euler_res, eu_LS_res, eu_NKPC_res, eu_TR_res, eu_RC_res):
    
    # Forward-looking objects
    C_eu_plus = lead(C_eu, ss.C_eu)
    pi_eu_plus = lead(pi_eu, ss.pi_eu)

    # EU price level in EUR
    price_from_inflation(PF_eu_s, pi_eu, par.T, ss.PF_eu_s)

    # Fisher equation
    rF_eu[:] = (1.0 + i_eu) / (1.0 + pi_eu_plus) - 1.0

    #PRODUCTION
    # Wage implied by marginal cost under outer CES unit-cost function
    pow_ = 1.0 - par.eta_VA_eu
    rhs = ((mc_eu * Z_eu) ** pow_ - par.beta_M_eu * (PM_eu ** pow_)) / (1.0 - par.beta_M_eu)
    #rhs = np.maximum(rhs, 1e-12)
    #w_eu = rhs ** (1.0 / pow_)
    #W_eu[:] = PF_eu_s * w_eu

    # wage implied by marginal cost
    w_eu = mc_eu * Z_eu
    W_eu[:] = PF_eu_s * w_eu

    # Static cost-minimizing material demand from outer CES
    ratio_MN = (par.beta_M_eu / (1.0 - par.beta_M_eu)) * (w_eu / PM_eu) ** par.eta_VA_eu
    M_eu[:] = N_eu * ratio_MN

    # Inner CES allocation between EU and US materials
    M_eu_us[:] = par.alpha_M_eu_us * (PM_eu_us / PM_eu) ** (-par.eta_M_eu) * M_eu
    M_eu_eu[:] = (1.0 - par.alpha_M_eu_us) * (PM_eu_eu / PM_eu) ** (-par.eta_M_eu) * M_eu

    # Output from outer CES production function
    rho = (par.eta_VA_eu - 1.0) / par.eta_VA_eu
    inside = (1.0 - par.beta_M_eu) * (N_eu ** rho) + par.beta_M_eu * (M_eu ** rho)
    Y_eu[:] = Z_eu * (inside ** (1.0 / rho))
    


    # Euler equation
    eu_Euler_res[:] = C_eu**(-par.sigma_eu) - par.beta_eu * (1.0 + rF_eu) * C_eu_plus**(-par.sigma_eu)

    # Labor supply
    eu_LS_res[:] = par.varphi_eu * N_eu**(par.nu_eu) - w_eu * C_eu**(-par.sigma_eu)

    # Resource constraint: gross output net of material absorption
    eu_RC_res[:] = Y_eu - C_eu - (PM_eu / PF_eu_s) * M_eu

    # NKPC
    eu_NKPC_res[:] = pi_eu - (par.beta_eu * pi_eu_plus + par.kappa_eu * (mc_eu - 1.0))

    # Taylor rule
    eu_TR_res[:] = i_eu - (ss.i_eu
                           + par.phi_pi_eu * (pi_eu - ss.pi_eu) 
                           + i_shock_eu)

    # Resource-based market size for Danish exports
    M_eu_s[:] = ss.M_eu_s * (Y_eu / ss.Y_eu)



@nb.njit
def us_nk(par, ini, ss,
          Z_us, i_shock_us,
          Y_us, C_us, N_us, pi_us, i_us,
          PF_us_s, rF_us, M_us_s, mc_us, W_us,
          PM_us_us, PM_us_eu, PM_us, M_us, M_us_eu, M_us_us,
          us_Euler_res, us_LS_res, us_NKPC_res, us_TR_res, us_RC_res):

    # Forward-looking objects
    C_us_plus = lead(C_us, ss.C_us)
    pi_us_plus = lead(pi_us, ss.pi_us)

    # US price level in USD
    price_from_inflation(PF_us_s, pi_us, par.T, ss.PF_us_s)

    # Fisher equation
    rF_us[:] = (1.0 + i_us) / (1.0 + pi_us_plus) - 1.0

    #PRODUCTION
    # Wage implied by marginal cost under outer CES unit-cost function
    pow_ = 1.0 - par.eta_VA_us
    rhs = ((mc_us * Z_us) ** pow_ - par.beta_M_us * (PM_us ** pow_)) / (1.0 - par.beta_M_us)
    #rhs = np.maximum(rhs, 1e-12)
    #w_us = rhs ** (1.0 / pow_)
    #W_us[:] = PF_us_s * w_us

    # wage implied by marginal cost
    w_us = mc_us * Z_us
    W_us[:] = PF_us_s * w_us

    # Static cost-minimizing material demand from outer CES
    ratio_MN = (par.beta_M_us / (1.0 - par.beta_M_us)) * (w_us / PM_us) ** par.eta_VA_us
    M_us[:] = N_us * ratio_MN

    # Inner CES allocation between EU and US materials
    M_us_us[:] = par.alpha_M_us_us * (PM_us_us / PM_us) ** (-par.eta_M_us) * M_us
    M_us_eu[:] = (1.0 - par.alpha_M_us_us) * (PM_us_eu / PM_us) ** (-par.eta_M_us) * M_us

    # Output from outer CES production function
    rho = (par.eta_VA_us - 1.0) / par.eta_VA_us
    inside = (1.0 - par.beta_M_us) * (N_us ** rho) + par.beta_M_us * (M_us ** rho)
    Y_us[:] = Z_us * (inside ** (1.0 / rho))
    


    # Euler equation
    us_Euler_res[:] = C_us**(-par.sigma_us) - par.beta_us * (1.0 + rF_us) * C_us_plus**(-par.sigma_us)

    # Labor supply
    us_LS_res[:] = par.varphi_us * N_us**(par.nu_us) - w_us * C_us**(-par.sigma_us)

    # resource constraint
    us_RC_res[:] = Y_us - C_us - (PM_us / PF_us_s) * M_us

    # NKPC
    us_NKPC_res[:] = pi_us - (par.beta_us * pi_us_plus + par.kappa_us * (mc_us - 1.0))

    # Taylor rule
    us_TR_res[:] = i_us - (ss.i_us
                           + par.phi_pi_us * (pi_us - ss.pi_us)
                           + i_shock_us)

    # Resource-based market size for Danish exports
    M_us_s[:] = ss.M_us_s * (Y_us / ss.Y_us)



#@nb.njit
#def mon_pol_us(par,ini,ss, E_us, CB_us):

#    E_us[:]=CB_us

@nb.njit
def production(par,ini,ss,
               ZTH,ZNT,NTH,NNT,piWTH,piWNT,
               YTH,YNT,WTH,WNT,PTH,PNT,
               PM_dk, PM_dk_eu, PM_dk_us,
               M_dk, M_dk_eu, M_dk_us):
    
    # a. NONTRADEABLES
    #production function
    YNT[:] = ZNT*NNT
    #wages
    price_from_inflation(WNT,piWNT,par.T,ss.WNT)
    #prices (p=mc)
    PNT[:] = WNT/ZNT


    # b. TRADEABLES
    #wages
    price_from_inflation(WTH,piWTH,par.T,ss.WTH)
    
    pow_ = 1.0 - par.eta_VA_dk
    # Nominal unit-cost CES
    inside_cost = (1.0 - par.beta_M_dk) * WTH**pow_ + par.beta_M_dk * PM_dk**pow_
    PTH[:] = (inside_cost ** (1.0 / pow_)) / ZTH

    # mc_TH = 1 by construction under perfect competition (price = unit cost)
    #mc_TH = 1.0

    # Cost-minimizing material demand (ratio to labor, from outer CES FOCs)
    # M/N = (beta_M_dk / (1-beta_M_dk)) * (WTH / PM_dk)^eta_VA_dk
    ratio_MN = (par.beta_M_dk / (1.0 - par.beta_M_dk)) * (WTH / PM_dk) ** par.eta_VA_dk
    M_dk[:] = NTH * ratio_MN   # NOTE: uses NTH as the labor input — rename to N_dk if you add N_dk as unknown

    # Inner CES: split M_dk between EU- and US-sourced materials
    # alpha_M_dk_us = share parameter on US materials
    M_dk_us[:] = par.alpha_M_dk_us  * (PM_dk_us / PM_dk) ** (-par.eta_M_dk) * M_dk
    M_dk_eu[:] = (1.0 - par.alpha_M_dk_us) * (PM_dk_eu / PM_dk) ** (-par.eta_M_dk) * M_dk

    # Gross output from outer CES
    rho = (par.eta_VA_dk - 1.0) / par.eta_VA_dk
    inside_Y = (1.0 - par.beta_M_dk) * NTH**rho + par.beta_M_dk * M_dk**rho
    YTH[:] = ZTH * (inside_Y ** (1.0 / rho))

    #YTH[:] = ZTH*NTH
    # c. price = marginal cost
    #PTH[:] = WTH/ZTH
    

@nb.njit
def prices(par,ini,ss,
           PF_eu_s, PF_us_s, E, E_us, PTH, PNT, WTH, WNT,
           PF_eu, PF_us, PF_TF, PTH_eu_s, PTH_us_s, PT, P, Q, Q_us, wTH, wNT):

    # a. convert currency: foreign prices in DKK
    PF_eu[:] = PF_eu_s * E
    PF_us[:] = PF_us_s * E_us

    # home tradable price in foreign currencies
    PTH_eu_s[:] = PTH / E
    PTH_us_s[:] = PTH / E_us
    # make some changes here somewhere with tariffs
    # b. foreign bundle price index (EU vs US)
    PF_TF[:] = price_index(PF_us, PF_eu, par.etaF_us, par.alpha_us)

    # c. tradables and CPI indices
    PT[:] = price_index(PF_TF, PTH, par.etaF, par.alphaF)
    P[:]  = price_index(PT, PNT, par.etaT, par.alphaT)

    # d. real exchange rates
    Q[:]    = PF_eu / P     # keep Q as EU real exchange rate (for UIP with rF_eu)
    Q_us[:] = PF_us / P     # US real exchange rate (for UIP_us with rF_us)

    # e. real wages
    wTH[:] = WTH / P
    wNT[:] = WNT / P

@nb.njit
def inflation(par,ini,ss,
              PF_eu_s, PF_us_s, PF_eu, PF_us, PF_TF, PNT, PTH, PT, P, PTH_eu_s, PTH_us_s,
              pi_F_eu_s, pi_F_us_s, pi_F_eu, pi_F_us, pi_FF,
              pi_NT, pi_TH, pi_T, pi, pi_TH_eu_s, pi_TH_us_s):

    pi_F_eu_s[:] = inflation_from_price(PF_eu_s, ini.PF_eu_s)
    pi_F_us_s[:] = inflation_from_price(PF_us_s, ini.PF_us_s)

    pi_F_eu[:]   = inflation_from_price(PF_eu, ini.PF_eu)
    pi_F_us[:]   = inflation_from_price(PF_us, ini.PF_us)

    pi_FF[:]     = inflation_from_price(PF_TF, ini.PF_TF)

    pi_NT[:] = inflation_from_price(PNT, ini.PNT)
    pi_TH[:] = inflation_from_price(PTH, ini.PTH)
    pi_T[:]  = inflation_from_price(PT, ini.PT)
    pi[:]    = inflation_from_price(P, ini.P)

    pi_TH_eu_s[:] = inflation_from_price(PTH_eu_s, ini.PTH_eu_s)
    pi_TH_us_s[:] = inflation_from_price(PTH_us_s, ini.PTH_us_s)

@nb.njit
def central_bank(par,ini,ss,pi,i,r,ra,E,i_shock,CB):
    
    if par.float == True: # taylor rule
        pi_plus = lead(pi,ss.pi)
        i[:] = (1+ss.i) * ((1+pi_plus)/(1+ss.pi))**par.phi -1 + i_shock
    else: # fixed exchange rate
        i[:] = CB
    
    # b. fisher

    # ex ante
    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1

    # ex post
    lag_i = lag(ini.i,i)
    ra[:] = (1+lag_i)/(1+pi)-1

@nb.njit
def government(par,ini,ss,
               PNT,P,wTH,NTH,wNT,NNT,ra,G,B,tau,inc_TH,inc_NT):

    # a. government budget # add tariff revenue to budget here somewhere
    for t in range(par.T):

        tax_base = wTH[t]*NTH[t]+wNT[t]*NNT[t]
        
        B_lag = prev(B,t,ini.B)

        #G[t] = ss.G
        tau[t] = ss.tau + par.omega*(B_lag-ss.B)/(ss.YTH+ss.YNT)

        tax_base = wTH[t]*NTH[t]+wNT[t]*NNT[t]
        B[t] = (1+ra[t])*B_lag + PNT[t]/P[t]*G[t]-tau[t]*tax_base

    # b. household income
    inc_TH[:] = (1-tau)*wTH*NTH
    inc_NT[:] = (1-tau)*wNT*NNT

@nb.njit
def NKWCs(par,ini,ss,beta,piWTH,piWNT,NTH,NNT,wTH,wNT,tau,UC_TH_hh,UC_NT_hh,NKWCT_res,NKWCNT_res):

    # a. phillips curve tradeable
    piWTH_plus = lead(piWTH,ss.piWTH)

    LHS = piWTH
    RHS = par.kappa*(par.varphiTH*(NTH/par.sT)**par.nu-1/par.muw*(1-tau)*wTH*UC_TH_hh) + beta*piWTH_plus    
    
    NKWCT_res[:] = LHS-RHS

    # b. phillips curve non-tradeable
    piWNT_plus = lead(piWNT,ss.piWNT)

    LHS = piWNT
    RHS = par.kappa*(par.varphiNT*(NNT/(1-par.sT))**par.nu-1/par.muw*(1-tau)*wNT*UC_NT_hh) + beta*piWNT_plus
    
    NKWCNT_res[:] = LHS-RHS

@nb.njit
def UIP(par,ini,ss,Q,r,rF_eu,UIP_res, Q_us, rF_us, UIP_res_us):

    #EU UIP
    Q_plus = lead(Q,ss.Q)

    LHS = 1+r
    RHS = (1+rF_eu)*Q_plus/Q
    UIP_res[:] = LHS-RHS

    #US UIP
    Q_us_plus = lead(Q_us, ss.Q_us)

    LHS = 1 + r
    RHS = (1 + rF_us) * Q_us_plus / Q_us
    UIP_res_us[:] = LHS - RHS

@nb.njit
def consumption(par,ini,ss,
                C_hh, PT, PNT, P, PTH,
                PF_TF, PF_eu, PF_us,
                M_eu_s, M_us_s, PTH_eu_s, PTH_us_s, PF_eu_s, PF_us_s,
                CT, CNT, CTF, CTF_eu, CTF_us, CTH, CTH_eu_s, CTH_us_s):

    # a. tradeable vs non-tradeable
    CT[:]  = par.alphaT*(PT/P)**(-par.etaT)*C_hh
    CNT[:] = (1-par.alphaT)*(PNT/P)**(-par.etaT)*C_hh

    # b. home vs foreign bundle inside tradables
    CTF[:] = par.alphaF*(PF_TF/PT)**(-par.etaF)*CT
    CTH[:] = (1-par.alphaF)*(PTH/PT)**(-par.etaF)*CT

    # c. split foreign bundle into EU vs US
    CTF_us[:] = par.alpha_us*(PF_us/PF_TF)**(-par.etaF_us)*CTF
    CTF_eu[:] = (1-par.alpha_us)*(PF_eu/PF_TF)**(-par.etaF_us)*CTF

    # d. foreign demand for DK tradables (exports)
    CTH_eu_s[:] = (PTH_eu_s/PF_eu_s)**(-par.eta_s)*M_eu_s
    CTH_us_s[:] = (PTH_us_s/PF_us_s)**(-par.eta_s)*M_us_s

@nb.njit
def market_clearing(par,ini,ss,
             YTH,CTH,CTH_eu_s,CTH_us_s,YNT,CNT,G,
             clearing_YTH,clearing_YNT):
    
    clearing_YTH[:] = YTH-CTH-CTH_eu_s-CTH_us_s
    clearing_YNT[:] = YNT-CNT-G

@nb.njit
def accounting(par,ini,ss,
               PTH,YTH,PNT,YNT,P,C_hh,G,A_hh,B,ra,
               GDP,NX,CA,NFA,Walras,
               PM_dk, M_dk):
    
    GDP[:] = (PTH*YTH+PNT*YNT)/P 
    NX[:] = GDP-C_hh-PNT/P*G-PM_dk*M_dk/P

    NFA[:] = A_hh-B

    NFA_lag = lag(ini.NFA,NFA)
    CA[:] = NX + ra*NFA_lag

    Walras[:] = (NFA-NFA_lag) - CA