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
                    PM_dk_eu, PM_dk_us, tau_m, tau_x):
    """
    Compute material price indices for EU and US production blocks, and
    the shared DK material price components (PM_dk_eu, PM_dk_us).
    Sector-specific DK aggregate prices (PM_dk_h, PM_dk_l) are computed
    in the production block using sector-specific alpha parameters.
    """

    # EU-sourced material price in EUR
    price_from_inflation(PM_eu_eu, piM_eu_eu, par.T, ss.PM_eu_eu)

    # US-sourced material price in USD
    price_from_inflation(PM_us_us, piM_us_us, par.T, ss.PM_us_us)

    # US materials priced in EUR (for EU production block)
    PM_eu_us[:] = (1.0 + tau_m) * PM_us_us * E_us / E

    # EU materials priced in USD (for US production block)
    PM_us_eu[:] = (1.0 + tau_x) * PM_eu_eu * E / E_us

    # Material price index for EU production
    PM_eu[:] = price_index(PM_eu_us, PM_eu_eu, par.eta_M_eu, par.alpha_M_eu_us)

    # Material price index for US production
    PM_us[:] = price_index(PM_us_us, PM_us_eu, par.eta_M_us, par.alpha_M_us_us)

    # Shared DK material price components in DKK
    PM_dk_eu[:] = PM_eu_eu * E
    PM_dk_us[:] = (1.0 + tau_m) * PM_us_us * E_us
    # Note: sector-specific PM_dk_h and PM_dk_l are computed in production block

@nb.njit
def eu_nk(par, ini, ss,
          Z_eu, i_shock_eu,
          Y_eu, C_eu, N_eu, pi_eu, i_eu,
          PF_eu_s, rF_eu, M_eu_s, mc_eu, W_eu,
          PM_eu_eu, PM_eu_us, PM_eu, M_eu, M_eu_eu, M_eu_us,
          eu_Euler_res, eu_LS_res, eu_NKPC_res, eu_TR_res, eu_RC_res):

    C_eu_plus = lead(C_eu, ss.C_eu)
    pi_eu_plus = lead(pi_eu, ss.pi_eu)

    price_from_inflation(PF_eu_s, pi_eu, par.T, ss.PF_eu_s)

    rF_eu[:] = (1.0 + i_eu) / (1.0 + pi_eu_plus) - 1.0

    pm_eu = PM_eu / PF_eu_s
    pow_ = 1.0 - par.eta_VA_eu
    rhs = ((mc_eu * Z_eu) ** pow_ - par.beta_M_eu * (pm_eu ** pow_)) / (1.0 - par.beta_M_eu)
    w_eu = rhs ** (1.0 / pow_)
    W_eu[:] = PF_eu_s * w_eu

    ratio_MN = (par.beta_M_eu / (1.0 - par.beta_M_eu)) * (w_eu / pm_eu) ** par.eta_VA_eu
    M_eu[:] = N_eu * ratio_MN

    M_eu_us[:] = par.alpha_M_eu_us * (PM_eu_us / PM_eu) ** (-par.eta_M_eu) * M_eu
    M_eu_eu[:] = (1.0 - par.alpha_M_eu_us) * (PM_eu_eu / PM_eu) ** (-par.eta_M_eu) * M_eu

    rho = (par.eta_VA_eu - 1.0) / par.eta_VA_eu
    inside = (1.0 - par.beta_M_eu)**(1.0/par.eta_VA_eu) * (N_eu ** rho) + par.beta_M_eu**(1.0/par.eta_VA_eu) * (M_eu ** rho)
    Y_eu[:] = Z_eu * (inside ** (1.0 / rho))

    eu_Euler_res[:] = C_eu**(-par.sigma_eu) - par.beta_eu * (1.0 + rF_eu) * C_eu_plus**(-par.sigma_eu)
    eu_LS_res[:] = par.varphi_eu * N_eu**(par.nu_eu) - w_eu * C_eu**(-par.sigma_eu)
    eu_RC_res[:] = Y_eu - C_eu - (PM_eu / PF_eu_s) * M_eu
    eu_NKPC_res[:] = pi_eu - (par.beta_eu * pi_eu_plus + par.kappa_eu * (mc_eu - 1.0))
    eu_TR_res[:] = i_eu - (ss.i_eu + par.phi_pi_eu * (pi_eu - ss.pi_eu) + i_shock_eu)

    M_eu_s[:] = ss.M_eu_s * (C_eu / ss.C_eu)

@nb.njit
def us_nk(par, ini, ss,
          Z_us, i_shock_us,
          Y_us, C_us, N_us, pi_us, i_us,
          PF_us_s, rF_us, M_us_s, mc_us, W_us,
          PM_us_us, PM_us_eu, PM_us, M_us, M_us_eu, M_us_us,
          us_Euler_res, us_LS_res, us_NKPC_res, us_TR_res, us_RC_res):

    C_us_plus = lead(C_us, ss.C_us)
    pi_us_plus = lead(pi_us, ss.pi_us)

    price_from_inflation(PF_us_s, pi_us, par.T, ss.PF_us_s)

    rF_us[:] = (1.0 + i_us) / (1.0 + pi_us_plus) - 1.0

    pm_us = PM_us / PF_us_s
    pow_ = 1.0 - par.eta_VA_us
    rhs = ((mc_us * Z_us) ** pow_ - par.beta_M_us * (pm_us ** pow_)) / (1.0 - par.beta_M_us)
    w_us = rhs ** (1.0 / pow_)
    W_us[:] = PF_us_s * w_us

    ratio_MN = (par.beta_M_us / (1.0 - par.beta_M_us)) * (w_us / pm_us) ** par.eta_VA_us
    M_us[:] = N_us * ratio_MN

    M_us_us[:] = par.alpha_M_us_us * (PM_us_us / PM_us) ** (-par.eta_M_us) * M_us
    M_us_eu[:] = (1.0 - par.alpha_M_us_us) * (PM_us_eu / PM_us) ** (-par.eta_M_us) * M_us

    rho = (par.eta_VA_us - 1.0) / par.eta_VA_us
    inside = (1.0 - par.beta_M_us)**(1.0/par.eta_VA_us) * (N_us ** rho) + par.beta_M_us**(1.0/par.eta_VA_us) * (M_us ** rho)
    Y_us[:] = Z_us * (inside ** (1.0 / rho))

    us_Euler_res[:] = C_us**(-par.sigma_us) - par.beta_us * (1.0 + rF_us) * C_us_plus**(-par.sigma_us)
    us_LS_res[:] = par.varphi_us * N_us**(par.nu_us) - w_us * C_us**(-par.sigma_us)
    us_RC_res[:] = Y_us - C_us - (PM_us / PF_us_s) * M_us
    us_NKPC_res[:] = pi_us - (par.beta_us * pi_us_plus + par.kappa_us * (mc_us - 1.0))
    us_TR_res[:] = i_us - (ss.i_us + par.phi_pi_us * (pi_us - ss.pi_us) + i_shock_us)

    M_us_s[:] = ss.M_us_s * (C_us / ss.C_us)

@nb.njit
def production(par, ini, ss,
               ZTH, ZNT, NHH, NHL, NNT, piWHH, piWHL, piWNT,
               YHH, YHL, YNT, WHH, WHL, WNT, PHH, PHL, PNT,
               PM_dk_eu, PM_dk_us,
               PM_dk_h, PM_dk_l,
               M_dk_h, M_dk_eu_h, M_dk_us_h,
               M_dk_l, M_dk_eu_l, M_dk_us_l):
    """
    Danish production block with two tradeable sectors:
      TH-High (HH): high US material exposure (alpha_M_dk_us_h, beta_M_dk_h)
      TH-Low  (HL): low  US material exposure (alpha_M_dk_us_l, beta_M_dk_l)
    Both sectors share the same TFP ZTH, outer-CES elasticity eta_VA_dk,
    and inner-CES elasticity eta_M_dk.
    PM_dk_eu and PM_dk_us (shared world prices in DKK) come from material_prices.
    Sector-specific aggregate material prices PM_dk_h, PM_dk_l are computed here.
    """

    # ---- Non-tradeable sector (unchanged) ----
    YNT[:] = ZNT * NNT
    price_from_inflation(WNT, piWNT, par.T, ss.WNT)
    PNT[:] = WNT / ZNT

    # shared CES exponents
    pow_ = 1.0 - par.eta_VA_dk
    rho  = (par.eta_VA_dk - 1.0) / par.eta_VA_dk

    # ---- TH-High sector ----
    price_from_inflation(WHH, piWHH, par.T, ss.WHH)

    # sector-specific aggregate material price (inner CES over EU and US materials)
    PM_dk_h[:] = price_index(PM_dk_us, PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_h)

    # output price = unit cost (perfect competition)
    inside_cost_h = (1.0 - par.beta_M_dk_h) * WHH**pow_ + par.beta_M_dk_h * PM_dk_h**pow_
    PHH[:] = (inside_cost_h ** (1.0 / pow_)) / ZTH

    # cost-minimising material demand (outer CES FOC)
    ratio_MN_h = (par.beta_M_dk_h / (1.0 - par.beta_M_dk_h)) * (WHH / PM_dk_h) ** par.eta_VA_dk
    M_dk_h[:] = NHH * ratio_MN_h

    # inner CES: split between EU and US materials
    M_dk_us_h[:] = par.alpha_M_dk_us_h  * (PM_dk_us / PM_dk_h) ** (-par.eta_M_dk) * M_dk_h
    M_dk_eu_h[:] = (1.0 - par.alpha_M_dk_us_h) * (PM_dk_eu / PM_dk_h) ** (-par.eta_M_dk) * M_dk_h

    # gross output
    inside_Y_h = ((1.0 - par.beta_M_dk_h)**(1.0/par.eta_VA_dk) * NHH**rho
                  + par.beta_M_dk_h**(1.0/par.eta_VA_dk) * M_dk_h**rho)
    YHH[:] = ZTH * (inside_Y_h ** (1.0 / rho))

    # ---- TH-Low sector ----
    price_from_inflation(WHL, piWHL, par.T, ss.WHL)

    PM_dk_l[:] = price_index(PM_dk_us, PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_l)

    inside_cost_l = (1.0 - par.beta_M_dk_l) * WHL**pow_ + par.beta_M_dk_l * PM_dk_l**pow_
    PHL[:] = (inside_cost_l ** (1.0 / pow_)) / ZTH

    ratio_MN_l = (par.beta_M_dk_l / (1.0 - par.beta_M_dk_l)) * (WHL / PM_dk_l) ** par.eta_VA_dk
    M_dk_l[:] = NHL * ratio_MN_l

    M_dk_us_l[:] = par.alpha_M_dk_us_l  * (PM_dk_us / PM_dk_l) ** (-par.eta_M_dk) * M_dk_l
    M_dk_eu_l[:] = (1.0 - par.alpha_M_dk_us_l) * (PM_dk_eu / PM_dk_l) ** (-par.eta_M_dk) * M_dk_l

    inside_Y_l = ((1.0 - par.beta_M_dk_l)**(1.0/par.eta_VA_dk) * NHL**rho
                  + par.beta_M_dk_l**(1.0/par.eta_VA_dk) * M_dk_l**rho)
    YHL[:] = ZTH * (inside_Y_l ** (1.0 / rho))

@nb.njit
def prices(par, ini, ss,
           PF_eu_s, PF_us_s, E, E_us,
           PHH, PHL, PNT, WHH, WHL, WNT,
           PF_eu, PF_us, PF_TF, PTH, PTH_eu_s, PTH_us_s, PT, P, Q, Q_us,
           wHH, wHL, wNT):
    """
    Price indices and real exchange rates.
    PTH = flat CES aggregate of the two home tradeable sector prices PHH and PHL.
    """

    # a. foreign prices in DKK
    PF_eu[:] = PF_eu_s * E
    PF_us[:] = PF_us_s * E_us

    # b. aggregate home tradeable price: flat CES of TH-High and TH-Low
    PTH[:] = price_index(PHH, PHL, par.eta_TH, par.omega_TH_H)

    # home tradeable price in foreign currencies (for Armington export demand)
    PTH_eu_s[:] = PTH / E
    PTH_us_s[:] = PTH / E_us

    # c. foreign tradeable bundle (EU vs US)
    PF_TF[:] = price_index(PF_us, PF_eu, par.etaF_us, par.alpha_us)

    # d. tradeable and CPI indices
    PT[:] = price_index(PF_TF, PTH, par.etaF, par.alphaF)
    P[:]  = price_index(PT,    PNT,  par.etaT,  par.alphaT)

    # e. real exchange rates
    Q[:]    = PF_eu / P
    Q_us[:] = PF_us / P

    # f. real wages
    wHH[:] = WHH / P
    wHL[:] = WHL / P
    wNT[:] = WNT / P

@nb.njit
def inflation(par, ini, ss,
              PF_eu_s, PF_us_s, PF_eu, PF_us, PF_TF,
              PNT, PHH, PHL, PTH, PT, P, PTH_eu_s, PTH_us_s,
              pi_F_eu_s, pi_F_us_s, pi_F_eu, pi_F_us, pi_FF,
              pi_NT, pi_PHH, pi_PHL, pi_TH, pi_T, pi,
              pi_TH_eu_s, pi_TH_us_s):

    pi_F_eu_s[:] = inflation_from_price(PF_eu_s, ini.PF_eu_s)
    pi_F_us_s[:] = inflation_from_price(PF_us_s, ini.PF_us_s)

    pi_F_eu[:]   = inflation_from_price(PF_eu,   ini.PF_eu)
    pi_F_us[:]   = inflation_from_price(PF_us,   ini.PF_us)

    pi_FF[:]     = inflation_from_price(PF_TF,   ini.PF_TF)

    pi_NT[:]     = inflation_from_price(PNT,     ini.PNT)
    pi_PHH[:]    = inflation_from_price(PHH,     ini.PHH)
    pi_PHL[:]    = inflation_from_price(PHL,     ini.PHL)
    pi_TH[:]     = inflation_from_price(PTH,     ini.PTH)
    pi_T[:]      = inflation_from_price(PT,      ini.PT)
    pi[:]        = inflation_from_price(P,       ini.P)

    pi_TH_eu_s[:] = inflation_from_price(PTH_eu_s, ini.PTH_eu_s)
    pi_TH_us_s[:] = inflation_from_price(PTH_us_s, ini.PTH_us_s)

@nb.njit
def central_bank(par,ini,ss,pi,i,r,ra,E,i_shock,CB):

    if par.float == True:
        pi_plus = lead(pi,ss.pi)
        i[:] = (1+ss.i) * ((1+pi_plus)/(1+ss.pi))**par.phi -1 + i_shock
    else:
        i[:] = CB

    pi_plus = lead(pi,ss.pi)
    r[:] = (1+i)/(1+pi_plus)-1

    lag_i = lag(ini.i,i)
    ra[:] = (1+lag_i)/(1+pi)-1

@nb.njit
def government(par, ini, ss,
               PNT, P, wHH, NHH, wHL, NHL, wNT, NNT, ra, G, B, tau,
               inc_HH, inc_HL, inc_NT,
               M_dk_us_h, M_dk_us_l, PM_dk_us, tau_m):
    """
    Government budget constraint and household income streams.
    Tariff revenue = tau_m * (PM_dk_us/P) * (M_dk_us_h + M_dk_us_l)
    PM_dk_us is shared across sectors; only quantities differ.
    """

    sNT = 1.0 - par.sHH - par.sHL

    #revenue = np.zeros(par.T)

    for t in range(par.T):

        tax_base = wHH[t]*NHH[t] + wHL[t]*NHL[t] + wNT[t]*NNT[t]

        B_lag = prev(B, t, ini.B)

        tau[t] = ss.tau + par.omega*(B_lag - ss.B) / (ss.YHH + ss.YHL + ss.YNT)
        revenue = tau_m[t] * PM_dk_us[t] / P[t] * (M_dk_us_h[t] + M_dk_us_l[t])

        if par.tariff_rev_lumpsum:
            B[t] = (1+ra[t])*B_lag + PNT[t]/P[t]*G[t] - tau[t]*tax_base
        else:
            B[t] = (1+ra[t])*B_lag + PNT[t]/P[t]*G[t] - tau[t]*tax_base - revenue

    # household income (after-tax wage income + optional lump-sum transfer)
    if par.tariff_rev_lumpsum:
        inc_HH[:] = (1-tau)*wHH*NHH + revenue*par.sHH
        inc_HL[:] = (1-tau)*wHL*NHL + revenue*par.sHL
        inc_NT[:] = (1-tau)*wNT*NNT + revenue*sNT
    else:
        inc_HH[:] = (1-tau)*wHH*NHH
        inc_HL[:] = (1-tau)*wHL*NHL
        inc_NT[:] = (1-tau)*wNT*NNT

@nb.njit
def NKWCs(par, ini, ss,
          beta, piWHH, piWHL, piWNT, NHH, NHL, NNT,
          wHH, wHL, wNT, tau,
          UC_HH_hh, UC_HL_hh, UC_NT_hh,
          NKWCHH_res, NKWCHL_res, NKWCNT_res):

    sNT = 1.0 - par.sHH - par.sHL

    # TH-High sector wage Phillips curve
    piWHH_plus = lead(piWHH, ss.piWHH)
    NKWCHH_res[:] = piWHH - (par.kappa*(par.varphiHH*(NHH/par.sHH)**par.nu
                              - 1/par.muw*(1-tau)*wHH*UC_HH_hh) + beta*piWHH_plus)

    # TH-Low sector wage Phillips curve
    piWHL_plus = lead(piWHL, ss.piWHL)
    NKWCHL_res[:] = piWHL - (par.kappa*(par.varphiHL*(NHL/par.sHL)**par.nu
                              - 1/par.muw*(1-tau)*wHL*UC_HL_hh) + beta*piWHL_plus)

    # NT sector wage Phillips curve
    piWNT_plus = lead(piWNT, ss.piWNT)
    NKWCNT_res[:] = piWNT - (par.kappa*(par.varphiNT*(NNT/sNT)**par.nu
                              - 1/par.muw*(1-tau)*wNT*UC_NT_hh) + beta*piWNT_plus)

@nb.njit
def UIP(par,ini,ss,Q,r,rF_eu,UIP_res, Q_us, rF_us, UIP_res_us):

    Q_plus = lead(Q,ss.Q)
    UIP_res[:] = (1+r) - (1+rF_eu)*Q_plus/Q

    Q_us_plus = lead(Q_us, ss.Q_us)
    UIP_res_us[:] = (1+r) - (1+rF_us)*Q_us_plus/Q_us

@nb.njit
def consumption(par, ini, ss,
                C_hh, PT, PNT, P, PTH, PHH, PHL,
                PF_TF, PF_eu, PF_us,
                M_eu_s, M_us_s, PTH_eu_s, PTH_us_s, PF_eu_s, PF_us_s,
                CT, CNT, CTF, CTF_eu, CTF_us,
                CTH, CTH_H, CTH_L,
                CTH_eu_s, CTH_us_s,
                CTH_HH_eu_s, CTH_HL_eu_s,
                CTH_HH_us_s, CTH_HL_us_s):
    """
    Consumption allocation:
      Outer nest  : T vs NT  (alphaT, etaT)
      Middle nest : home (PTH) vs foreign bundle (PF_TF)  (alphaF, etaF)
      Inner nest  : TH-High (PHH) vs TH-Low (PHL) — flat CES (omega_TH_H, eta_TH)
    Foreign bundle : EU vs US  (alpha_us, etaF_us)

    Foreign consumers (EU, US) face the same aggregate PTH for DK exports.
    Export quantities are split between sectors with the same flat-CES shares,
    so no export differentiation by sector is imposed.
    """

    # a. T vs NT
    CT[:]  = par.alphaT   * (PT  / P)**(-par.etaT) * C_hh
    CNT[:] = (1-par.alphaT) * (PNT / P)**(-par.etaT) * C_hh

    # b. home bundle vs foreign bundle
    CTF[:] = par.alphaF      * (PF_TF / PT)**(-par.etaF) * CT
    CTH[:] = (1-par.alphaF)  * (PTH   / PT)**(-par.etaF) * CT

    # c. inner flat CES: TH-High vs TH-Low within home bundle
    CTH_H[:] = par.omega_TH_H       * (PHH / PTH)**(-par.eta_TH) * CTH
    CTH_L[:] = (1.0-par.omega_TH_H) * (PHL / PTH)**(-par.eta_TH) * CTH

    # d. EU vs US inside foreign bundle
    CTF_us[:] = par.alpha_us      * (PF_us / PF_TF)**(-par.etaF_us) * CTF
    CTF_eu[:] = (1-par.alpha_us)  * (PF_eu / PF_TF)**(-par.etaF_us) * CTF

    # e. total export demand from EU and US (Armington on aggregate PTH)
    CTH_eu_s[:] = (PTH_eu_s / PF_eu_s)**(-par.eta_s) * M_eu_s
    CTH_us_s[:] = (PTH_us_s / PF_us_s)**(-par.eta_s) * M_us_s

    # f. split exports by sector using the same flat-CES shares
    #    (foreign consumers see the same PTH bundle as domestic consumers)
    CTH_HH_eu_s[:] = par.omega_TH_H       * (PHH / PTH)**(-par.eta_TH) * CTH_eu_s
    CTH_HL_eu_s[:] = (1.0-par.omega_TH_H) * (PHL / PTH)**(-par.eta_TH) * CTH_eu_s
    CTH_HH_us_s[:] = par.omega_TH_H       * (PHH / PTH)**(-par.eta_TH) * CTH_us_s
    CTH_HL_us_s[:] = (1.0-par.omega_TH_H) * (PHL / PTH)**(-par.eta_TH) * CTH_us_s

@nb.njit
def market_clearing(par, ini, ss,
                    YHH, CTH_H, CTH_HH_eu_s, CTH_HH_us_s,
                    YHL, CTH_L, CTH_HL_eu_s, CTH_HL_us_s,
                    YNT, CNT, G,
                    clearing_YHH, clearing_YHL, clearing_YNT):

    clearing_YHH[:] = YHH - CTH_H - CTH_HH_eu_s - CTH_HH_us_s
    clearing_YHL[:] = YHL - CTH_L - CTH_HL_eu_s - CTH_HL_us_s
    clearing_YNT[:] = YNT - CNT - G

@nb.njit
def accounting(par, ini, ss,
               PHH, YHH, PHL, YHL, PNT, YNT, P, C_hh, G, A_hh, B, ra,
               GDP, NX, CA, NFA, Walras,
               PM_dk_h, M_dk_h, PM_dk_l, M_dk_l):

    # GDP = value added = gross output minus intermediate inputs
    GDP[:] = (PHH*YHH - PM_dk_h*M_dk_h + PHL*YHL - PM_dk_l*M_dk_l + PNT*YNT) / P
    NX[:] = GDP - C_hh - PNT/P*G

    NFA[:] = A_hh - B

    NFA_lag = lag(ini.NFA, NFA)
    CA[:] = NX + ra*NFA_lag

    Walras[:] = (NFA - NFA_lag) - CA
