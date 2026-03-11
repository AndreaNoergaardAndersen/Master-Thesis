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
def eu_nk(par, ini, ss,
          rn_eu, i_shock_eu, Z_eu,
          x_eu, pi_eu, i_eu,
          PF_eu_s, rF_eu, M_eu_s,   # <-- ADD M_eu_s here
          Y_eu, N_eu, mc_eu,
          eu_IS_res, eu_NKPC_res, eu_TR_res):
    """Closed simple EU NK block (log-linear in x and pi), producing PF_eu_s (EU price level).

    Variables:
      x_eu   : output gap
      pi_eu  : inflation (e.g. 0.01 = 1%)
      i_eu   : nominal policy rate
      PF_eu_s: EU price level in EUR

    Shocks:
      rn_eu      : natural real rate
      i_shock_eu : EU monetary policy shock
    """

    # 1) Taylor rule residual
    eu_TR_res[:] = i_eu - (ss.i_eu
                           + par.phi_pi_eu * (pi_eu - ss.pi_eu)
                           + par.phi_x_eu * (x_eu - ss.x_eu)
                           )+ i_shock_eu

    # 2) IS curve residual: x_t = E x_{t+1} - (1/sigma)(i_t - E pi_{t+1} - rn_t)
    x_plus = lead(x_eu, ss.x_eu)
    pi_plus = lead(pi_eu, ss.pi_eu)
    eu_IS_res[:] = x_eu - (x_plus - (1.0 / par.sigma_eu) * (i_eu - pi_plus - rn_eu))

    # 4) EU price level from inflation (in EUR)
    price_from_inflation(PF_eu_s, pi_eu, par.T, ss.PF_eu_s)

    mc_eu[:]=(par.W_eu_ss/Z_eu)/PF_eu_s

    # NKPC in marginal cost
    eu_NKPC_res[:] = pi_eu - (par.beta_eu*pi_plus + par.kappa_eu*(mc_eu - 1.0))

    # EU ex ante real rate used in UIP: (1+i)/(1+E pi(+1)) - 1
    rF_eu[:] = (1.0 + i_eu)/(1.0 + pi_plus) - 1.0

    # -------------------------------------------------
    # NEW: EU market size for SOE exports (endogenous)
    # -------------------------------------------------
    # Contractionary monetary policy -> x_eu falls -> M_eu_s falls
    M_eu_s[:] = ss.M_eu_s * np.exp(par.chi_M_eu * x_eu)

    # Optional accounting: EU exports and absorption
    #X_eu_to_dk[:] = CTF
    Y_eu[:] = ss.Y_eu * (1.0 + x_eu)
    N_eu[:] = Y_eu / Z_eu
    #C_eu[:] = Y_eu - X_eu_to_dk

@nb.njit
def us_nk(par, ini, ss,
          rn_us, i_shock_us, Z_us,
          x_us, pi_us, i_us,
          PF_us_s, rF_us, M_us_s,
          Y_us, N_us, mc_us,
          us_IS_res, us_NKPC_res, us_TR_res):
    """Closed simple EU NK block (log-linear in x and pi), producing PF_us_s (US price level).
    """

    # 1) Taylor rule residual
    us_TR_res[:] = i_us - (ss.i_us
                           + par.phi_pi_us * (pi_us - ss.pi_us)
                           + par.phi_x_us  * (x_us  - ss.x_us)
                           ) + i_shock_us

    # 2) IS curve residual: x_t = E x_{t+1} - (1/sigma)(i_t - E pi_{t+1} - rn_t)
    x_plus  = lead(x_us, ss.x_us)
    pi_plus = lead(pi_us, ss.pi_us)
    us_IS_res[:] = x_us - (x_plus - (1.0 / par.sigma_us) * (i_us - pi_plus - rn_us))

    # 4) EU price level from inflation (in USD)
    price_from_inflation(PF_us_s, pi_us, par.T, ss.PF_us_s)

    mc_us[:] = (par.W_us_ss / Z_us) / PF_us_s

    # NKPC in marginal cost
    us_NKPC_res[:] = pi_us - (par.beta_us*pi_plus + par.kappa_us*(mc_us - 1.0))

    # EU ex ante real rate used in UIP: (1+i)/(1+E pi(+1)) - 1
    rF_us[:] = (1.0 + i_us)/(1.0 + pi_plus) - 1.0

    # -------------------------------------------------
    # NEW: EU market size for SOE exports (endogenous)
    # -------------------------------------------------
    # Contractionary monetary policy -> x_us falls -> M_us_s falls
    M_us_s[:] = ss.M_us_s * np.exp(par.chi_M_us * x_us)

    # accounting
    Y_us[:] = ss.Y_us * (1.0 + x_us)
    N_us[:] = Y_us / Z_us

@nb.njit
def mon_pol(par,ini,ss,E,CB):

    if par.float == True:
        E[:] = CB 
    else:
        E[:] = ss.E

@nb.njit
def mon_pol_us(par,ini,ss, E_us, CB_us):

    E_us[:]=CB_us

@nb.njit
def production(par,ini,ss,
               ZTH,ZNT,NTH,NNT,piWTH,piWNT,
               YTH,YNT,WTH,WNT,PTH,PNT):
    
    # a. production
    YTH[:] = ZTH*NTH
    YNT[:] = ZNT*NNT
    
    # b. wages
    price_from_inflation(WTH,piWTH,par.T,ss.WTH)
    price_from_inflation(WNT,piWNT,par.T,ss.WNT)

    # c. price = marginal cost
    PTH[:] = WTH/ZTH
    PNT[:] = WNT/ZNT

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

    # a. government budget
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
    CTH_us_s[:] = (PTH_us_s/PF_us_s)**(-par.eta_us_s)*M_us_s

@nb.njit
def market_clearing(par,ini,ss,
             YTH,CTH,CTH_eu_s,CTH_us_s,YNT,CNT,G,
             clearing_YTH,clearing_YNT):
    
    clearing_YTH[:] = YTH-CTH-CTH_eu_s-CTH_us_s
    clearing_YNT[:] = YNT-CNT-G

@nb.njit
def accounting(par,ini,ss,
               PTH,YTH,PNT,YNT,P,C_hh,G,A_hh,B,ra,
               GDP,NX,CA,NFA,Walras):
    
    GDP[:] = (PTH*YTH+PNT*YNT)/P 
    NX[:] = GDP-C_hh-PNT/P*G

    NFA[:] = A_hh-B

    NFA_lag = lag(ini.NFA,NFA)
    CA[:] = NX + ra*NFA_lag

    Walras[:] = (NFA-NFA_lag) - CA