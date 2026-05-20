import numpy as np
import numba as nb
from GEModelTools import prev, next, lag, lead, isclose
from GEModelTools import lag, lead

##############
## Helpers ##
##############

@nb.njit
def price_index(P1,P2,eta,alpha):
    if eta == 1.0:
        return P1**alpha * P2**(1-alpha)
    return (alpha*P1**(1-eta)+(1-alpha)*P2**(1-eta))**(1/(1-eta))

@nb.njit
def price_index_4(P1,P2,P3,P4,eta,o1,o2,o3,o4):
    if eta == 1.0:
        return P1**o1 * P2**o2 * P3**o3 * P4**o4
    return (o1*P1**(1-eta)+o2*P2**(1-eta)+o3*P3**(1-eta)+o4*P4**(1-eta))**(1/(1-eta))

#@nb.njit
#def price_index_t(P1, P2, eta, alpha):
    # """ CES price index with a time-varying elasticity.
    #     Robust to inputs of shape (T,) or (T,1): GEModelTools passes
    #     2-D path slices during compute_jacs and 1-D paths at runtime. """
    # P1f  = P1.ravel()
    # P2f  = P2.ravel()
    # etaf = eta.ravel()
    # T_   = P1f.shape[0]
    # out  = np.empty(T_)
    # for t in range(T_):
    #     e    = etaf[t]
    #     diff = e - 1.0
    #     if diff < 0.0:
    #         diff = -diff
    #     if diff < 1e-10:
    #         out[t] = P1f[t]**alpha * P2f[t]**(1.0 - alpha)
    #     else:
    #         out[t] = (alpha*P1f[t]**(1.0 - e)
    #                   + (1.0 - alpha)*P2f[t]**(1.0 - e))**(1.0/(1.0 - e))
    # return out.reshape(P1.shape)

@nb.njit
def price_index_t(P1, P2, eta, alpha):
    P1f  = P1.ravel()
    P2f  = P2.ravel()
    etaf = eta.ravel()
    T_   = P1f.shape[0]
    out  = np.empty(T_)
    for t in range(T_):
        e = etaf[t]
        if np.abs(e - 1.0) < 1e-6:  # threshold raised: catches eta=1 and nearby perturbations
            out[t] = P1f[t]**alpha * P2f[t]**(1.0 - alpha)
        else:
            out[t] = (alpha * P1f[t]**(1.0 - e)
                      + (1.0 - alpha) * P2f[t]**(1.0 - e))**(1.0 / (1.0 - e))
    return out.reshape(P1.shape)

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
                    E, E_us,
                    PM_eu_eu, PM_us_us, PM_eu_us, PM_eu, PM_us_eu, PM_us,
                    PM_dk_eu, PM_dk_us, tau_m, tau_x, PT_eu_s, pi_T_eu, PT_us_s, pi_T_us):
    """
    PT_eu_s = EU tradable CPI in EUR (drives EU material costs and Armington).
    PT_us_s = US tradable CPI in USD (drives US material costs and Armington).
    pi_T_eu, pi_T_us = tradable inflation for each bloc (unknowns, drive T-sector NKPCs).
    """

    price_from_inflation(PT_eu_s, pi_T_eu, par.T, ss.PT_eu_s)
    PM_eu_eu[:] = PT_eu_s

    price_from_inflation(PT_us_s, pi_T_us, par.T, ss.PT_us_s)
    PM_us_us[:] = PT_us_s

    PM_eu_us[:] = (1.0 + tau_m) * PM_us_us * E_us / E
    PM_us_eu[:] = (1.0 + tau_x) * PM_eu_eu * E / E_us

    PM_eu[:] = price_index(PM_eu_us, PM_eu_eu, par.eta_M_eu, par.alpha_M_eu_us)
    PM_us[:] = price_index(PM_us_us, PM_us_eu, par.eta_M_us, par.alpha_M_us_us)

    PM_dk_eu[:] = PM_eu_eu * E
    PM_dk_us[:] = (1.0 + tau_m) * PM_us_us * E_us

@nb.njit
def eu_nk(par, ini, ss,
          Z_eu, ZNT_eu, i_shock_eu,
          Y_eu, C_eu, N_eu, NNT_eu, pi_T_eu, pi_NT_eu, i_eu,
          PF_eu_s, PT_eu_s, PNT_eu_s, rF_eu, M_eu_s, mc_eu, W_eu,
          C_T_eu, pi_eu,
          PM_eu_eu, PM_eu_us, PM_eu, M_eu, M_eu_eu, M_eu_us,
          eu_Euler_res, eu_LS_res, eu_NKPC_res, eu_NKPC_NT_res,
          eu_TR_res, eu_RC_res, eu_NT_res,
          tau_m):

    # NT price level from pi_NT_eu (unknown)
    price_from_inflation(PNT_eu_s, pi_NT_eu, par.T, ss.PNT_eu_s)

    # PT_eu_s already built in material_prices from pi_T_eu (unknown)
    # Aggregate CPI: CES of tradable and non-tradable price levels
    PF_eu_s[:] = price_index(PT_eu_s, PNT_eu_s, par.etaT_eu, par.alphaT_eu)

    # Derived aggregate inflation (used in Euler, Taylor rule, real rate)
    pi_eu[:] = inflation_from_price(PF_eu_s, ini.PF_eu_s)
    pi_eu_plus = lead(pi_eu, ss.pi_eu)

    # Real interest rate deflated by aggregate CPI
    rF_eu[:] = (1.0 + i_eu) / (1.0 + pi_eu_plus) - 1.0

    # Leads for NKPCs
    pi_T_eu_plus  = lead(pi_T_eu,  ss.pi_T_eu)
    pi_NT_eu_plus = lead(pi_NT_eu, ss.pi_NT_eu)

    # ---- Tradable sector: cost function solved in nominal space ----
    # W_eu/PM_eu = w_T/pm_eu (PT_eu_s cancels), so ratio_MN uses W_eu/PM_eu directly
    pow_ = 1.0 - par.eta_VA_eu
    rhs  = ((mc_eu * Z_eu * PT_eu_s)**pow_ - par.beta_M_eu * PM_eu**pow_) / (1.0 - par.beta_M_eu)
    W_eu[:] = rhs ** (1.0 / pow_)

    ratio_MN = (par.beta_M_eu / (1.0 - par.beta_M_eu)) * (W_eu / PM_eu) ** par.eta_VA_eu
    M_eu[:] = N_eu * ratio_MN

    M_eu_us[:] = par.alpha_M_eu_us * (PM_eu_us / PM_eu) ** (-par.eta_M_eu) * M_eu
    M_eu_eu[:] = (1.0 - par.alpha_M_eu_us) * (PM_eu_eu / PM_eu) ** (-par.eta_M_eu) * M_eu

    rho = (par.eta_VA_eu - 1.0) / par.eta_VA_eu
    inside = (1.0 - par.beta_M_eu)**(1.0/par.eta_VA_eu) * (N_eu ** rho) + par.beta_M_eu**(1.0/par.eta_VA_eu) * (M_eu ** rho)
    Y_eu[:] = Z_eu * (inside ** (1.0 / rho))

    # ---- NT sector ----
    Y_NT_eu = ZNT_eu * NNT_eu

    # ---- Household decisions ----
    C_T_eu[:]  = par.alphaT_eu       * (PT_eu_s  / PF_eu_s) ** (-par.etaT_eu) * C_eu
    C_NT_eu    = (1.0 - par.alphaT_eu) * (PNT_eu_s / PF_eu_s) ** (-par.etaT_eu) * C_eu

    # ---- Residuals ----
    C_eu_plus = lead(C_eu, ss.C_eu)
    eu_Euler_res[:] = C_eu**(-par.sigma_eu) - par.beta_eu * (1.0 + rF_eu) * C_eu_plus**(-par.sigma_eu)

    # Labor supply: real wage in terms of aggregate CPI basket
    w_agg = W_eu / PF_eu_s
    eu_LS_res[:] = par.varphi_eu * (N_eu + NNT_eu)**(par.nu_eu) - w_agg * C_eu**(-par.sigma_eu)

    # Aggregate resource constraint (deflated by PF_eu_s)
    tariff_rev_eu = tau_m / (1.0 + tau_m) * (PM_eu_us / PF_eu_s) * M_eu_us
    eu_RC_res[:] = ((PT_eu_s / PF_eu_s) * Y_eu
                    + (PNT_eu_s / PF_eu_s) * Y_NT_eu
                    - C_eu
                    - (PM_eu / PF_eu_s) * M_eu
                    + tariff_rev_eu)

    # T-sector NKPC
    eu_NKPC_res[:] = pi_T_eu - (par.beta_eu * pi_T_eu_plus + par.kappa_eu * (mc_eu - 1.0))

    # NT-sector NKPC (mc_NT_eu = W_eu / PNT_eu_s)
    mc_NT_eu = W_eu / PNT_eu_s
    eu_NKPC_NT_res[:] = pi_NT_eu - (par.beta_eu * pi_NT_eu_plus + par.kappa_eu * (mc_NT_eu - 1.0))

    # Taylor rule uses aggregate inflation pi_eu
    eu_TR_res[:] = i_eu - (ss.i_eu + par.phi_pi_eu * (pi_eu - ss.pi_eu) + i_shock_eu)

    # NT market clearing
    eu_NT_res[:] = Y_NT_eu - C_NT_eu

    # Export market size scales with tradable consumption
    M_eu_s[:] = ss.M_eu_s * (C_T_eu / ss.C_T_eu)

@nb.njit
def us_nk(par, ini, ss,
          Z_us, ZNT_us, i_shock_us,
          Y_us, C_us, C_T_us, N_us, NNT_us, pi_T_us, pi_NT_us, pi_us, i_us,
          PT_us_s, PNT_us_s, PF_us_s, rF_us, M_us_s, mc_us, W_us,
          PM_us_us, PM_us_eu, PM_us, M_us, M_us_eu, M_us_us,
          us_Euler_res, us_LS_res, us_NKPC_res, us_NKPC_NT_res,
          us_TR_res, us_RC_res, us_NT_res, tau_x):

    # PT_us_s already built in material_prices from pi_T_us (unknown)
    # NT price level from pi_NT_us (unknown)
    price_from_inflation(PNT_us_s, pi_NT_us, par.T, ss.PNT_us_s)

    # Aggregate CPI: CES of tradable and non-tradable price levels
    PF_us_s[:] = price_index(PT_us_s, PNT_us_s, par.etaT_us, par.alphaT_us)

    # Derived aggregate inflation (used in Euler, Taylor rule, real rate)
    pi_us[:] = inflation_from_price(PF_us_s, ini.PF_us_s)
    pi_us_plus   = lead(pi_us,   ss.pi_us)
    pi_T_us_plus = lead(pi_T_us, ss.pi_T_us)
    pi_NT_us_plus = lead(pi_NT_us, ss.pi_NT_us)

    # Real interest rate deflated by aggregate CPI
    rF_us[:] = (1.0 + i_us) / (1.0 + pi_us_plus) - 1.0

    # ---- Tradable sector: cost function in nominal space ----
    pow_ = 1.0 - par.eta_VA_us
    rhs = ((mc_us * Z_us * PT_us_s)**pow_ - par.beta_M_us * PM_us**pow_) / (1.0 - par.beta_M_us)
    W_us[:] = rhs ** (1.0 / pow_)

    ratio_MN = (par.beta_M_us / (1.0 - par.beta_M_us)) * (W_us / PM_us) ** par.eta_VA_us
    M_us[:] = N_us * ratio_MN

    M_us_us[:] = par.alpha_M_us_us * (PM_us_us / PM_us) ** (-par.eta_M_us) * M_us
    M_us_eu[:] = (1.0 - par.alpha_M_us_us) * (PM_us_eu / PM_us) ** (-par.eta_M_us) * M_us

    rho = (par.eta_VA_us - 1.0) / par.eta_VA_us
    inside = ((1.0 - par.beta_M_us)**(1.0/par.eta_VA_us) * N_us**rho
              + par.beta_M_us**(1.0/par.eta_VA_us) * M_us**rho)
    Y_us[:] = Z_us * (inside ** (1.0 / rho))

    # ---- NT sector ----
    Y_NT_us = ZNT_us * NNT_us

    # ---- Household decisions ----
    C_T_us[:] = par.alphaT_us        * (PT_us_s  / PF_us_s) ** (-par.etaT_us) * C_us
    C_NT_us   = (1.0 - par.alphaT_us) * (PNT_us_s / PF_us_s) ** (-par.etaT_us) * C_us

    # ---- Residuals ----
    C_us_plus = lead(C_us, ss.C_us)
    us_Euler_res[:] = C_us**(-par.sigma_us) - par.beta_us * (1.0 + rF_us) * C_us_plus**(-par.sigma_us)

    # Labor supply: real wage in terms of aggregate CPI basket
    w_agg = W_us / PF_us_s
    us_LS_res[:] = par.varphi_us * (N_us + NNT_us)**(par.nu_us) - w_agg * C_us**(-par.sigma_us)

    # Aggregate resource constraint (deflated by PF_us_s)
    tariff_rev_us = tau_x / (1.0 + tau_x) * (PM_us_eu / PF_us_s) * M_us_eu
    us_RC_res[:] = ((PT_us_s  / PF_us_s) * Y_us
                    + (PNT_us_s / PF_us_s) * Y_NT_us
                    - C_us
                    - (PM_us / PF_us_s) * M_us
                    + tariff_rev_us)

    # T-sector NKPC
    us_NKPC_res[:] = pi_T_us - (par.beta_us * pi_T_us_plus + par.kappa_us * (mc_us - 1.0))

    # NT-sector NKPC (mc_NT_us = W_us / PNT_us_s)
    mc_NT_us = W_us / PNT_us_s
    us_NKPC_NT_res[:] = pi_NT_us - (par.beta_us * pi_NT_us_plus + par.kappa_us * (mc_NT_us - 1.0))

    # Taylor rule uses aggregate inflation pi_us
    us_TR_res[:] = i_us - (ss.i_us + par.phi_pi_us * (pi_us - ss.pi_us) + i_shock_us)

    # NT market clearing
    us_NT_res[:] = Y_NT_us - C_NT_us

    # Export market size scales with tradable consumption
    M_us_s[:] = ss.M_us_s * (C_T_us / ss.C_T_us)

@nb.njit
def production(par, ini, ss,
               ZTH_HH, ZTH_HL, ZTH_LH, ZTH_LL, ZNT, 
               NHH, NHL, NLH, NLL, NNT,
               piWHH, piWHL, piWLH, piWLL, piWNT,
               YHH, YHL, YLH, YLL, YNT,
               WHH, WHL, WLH, WLL, WNT,
               PHH, PHL, PLH, PLL, PNT,
               PM_dk_eu, PM_dk_us,
               PM_dk_h, PM_dk_l,
               M_dk_h,  M_dk_eu_h,  M_dk_us_h,
               M_dk_l,  M_dk_eu_l,  M_dk_us_l,
               M_dk_hx, M_dk_eu_hx, M_dk_us_hx,
               M_dk_lx, M_dk_eu_lx, M_dk_us_lx):
    """
    Danish production block with four tradeable sectors:
      HH: high-material params (h), high US export share
      HL: high-material params (h), low  US export share
      LH: low-material  params (l), high US export share
      LL: low-material  params (l), low  US export share
    HH and HL share beta_M_dk_h, alpha_M_dk_us_h, PM_dk_h.
    LH and LL share beta_M_dk_l, alpha_M_dk_us_l, PM_dk_l.
    """

    # ---- Non-tradeable sector ----
    YNT[:] = ZNT * NNT
    price_from_inflation(WNT, piWNT, par.T, ss.WNT)
    PNT[:] = WNT / ZNT

    pow_ = 1.0 - par.eta_VA_dk
    rho  = (par.eta_VA_dk - 1.0) / par.eta_VA_dk

    # Shared aggregate material prices per material group
    PM_dk_h[:] = price_index(PM_dk_us, PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_h)
    PM_dk_l[:] = price_index(PM_dk_us, PM_dk_eu, par.eta_M_dk, par.alpha_M_dk_us_l)

    # ---- HH sector (high material, high US export) ----
    price_from_inflation(WHH, piWHH, par.T, ss.WHH)
    inside_cost_h = (1.0 - par.beta_M_dk_h) * WHH**pow_ + par.beta_M_dk_h * PM_dk_h**pow_
    PHH[:] = (inside_cost_h ** (1.0 / pow_)) / ZTH_HH
    ratio_MN_h = (par.beta_M_dk_h / (1.0 - par.beta_M_dk_h)) * (WHH / PM_dk_h) ** par.eta_VA_dk
    M_dk_h[:] = NHH * ratio_MN_h
    M_dk_us_h[:] = par.alpha_M_dk_us_h  * (PM_dk_us / PM_dk_h) ** (-par.eta_M_dk) * M_dk_h
    M_dk_eu_h[:] = (1.0 - par.alpha_M_dk_us_h) * (PM_dk_eu / PM_dk_h) ** (-par.eta_M_dk) * M_dk_h
    inside_Y_h = ((1.0 - par.beta_M_dk_h)**(1.0/par.eta_VA_dk) * NHH**rho
                  + par.beta_M_dk_h**(1.0/par.eta_VA_dk) * M_dk_h**rho)
    YHH[:] = ZTH_HH * (inside_Y_h ** (1.0 / rho))

    # ---- HL sector (high material, low US export) ----
    # Same production technology as HH (same h-params), different export structure
    price_from_inflation(WHL, piWHL, par.T, ss.WHL)
    inside_cost_hx = (1.0 - par.beta_M_dk_h) * WHL**pow_ + par.beta_M_dk_h * PM_dk_h**pow_
    PHL[:] = (inside_cost_hx ** (1.0 / pow_)) / ZTH_HL
    ratio_MN_hx = (par.beta_M_dk_h / (1.0 - par.beta_M_dk_h)) * (WHL / PM_dk_h) ** par.eta_VA_dk
    M_dk_hx[:] = NHL * ratio_MN_hx
    M_dk_us_hx[:] = par.alpha_M_dk_us_h  * (PM_dk_us / PM_dk_h) ** (-par.eta_M_dk) * M_dk_hx
    M_dk_eu_hx[:] = (1.0 - par.alpha_M_dk_us_h) * (PM_dk_eu / PM_dk_h) ** (-par.eta_M_dk) * M_dk_hx
    inside_Y_hx = ((1.0 - par.beta_M_dk_h)**(1.0/par.eta_VA_dk) * NHL**rho
                   + par.beta_M_dk_h**(1.0/par.eta_VA_dk) * M_dk_hx**rho)
    YHL[:] = ZTH_HL * (inside_Y_hx ** (1.0 / rho))

    # ---- LH sector (low material, high US export) ----
    price_from_inflation(WLH, piWLH, par.T, ss.WLH)
    inside_cost_l = (1.0 - par.beta_M_dk_l) * WLH**pow_ + par.beta_M_dk_l * PM_dk_l**pow_
    PLH[:] = (inside_cost_l ** (1.0 / pow_)) / ZTH_LH
    ratio_MN_l = (par.beta_M_dk_l / (1.0 - par.beta_M_dk_l)) * (WLH / PM_dk_l) ** par.eta_VA_dk
    M_dk_l[:] = NLH * ratio_MN_l
    M_dk_us_l[:] = par.alpha_M_dk_us_l  * (PM_dk_us / PM_dk_l) ** (-par.eta_M_dk) * M_dk_l
    M_dk_eu_l[:] = (1.0 - par.alpha_M_dk_us_l) * (PM_dk_eu / PM_dk_l) ** (-par.eta_M_dk) * M_dk_l
    inside_Y_l = ((1.0 - par.beta_M_dk_l)**(1.0/par.eta_VA_dk) * NLH**rho
                  + par.beta_M_dk_l**(1.0/par.eta_VA_dk) * M_dk_l**rho)
    YLH[:] = ZTH_LH * (inside_Y_l ** (1.0 / rho))

    # ---- LL sector (low material, low US export) ----
    # Same production technology as LH (same l-params), different export structure
    price_from_inflation(WLL, piWLL, par.T, ss.WLL)
    inside_cost_lx = (1.0 - par.beta_M_dk_l) * WLL**pow_ + par.beta_M_dk_l * PM_dk_l**pow_
    PLL[:] = (inside_cost_lx ** (1.0 / pow_)) / ZTH_LL
    ratio_MN_lx = (par.beta_M_dk_l / (1.0 - par.beta_M_dk_l)) * (WLL / PM_dk_l) ** par.eta_VA_dk
    M_dk_lx[:] = NLL * ratio_MN_lx
    M_dk_us_lx[:] = par.alpha_M_dk_us_l  * (PM_dk_us / PM_dk_l) ** (-par.eta_M_dk) * M_dk_lx
    M_dk_eu_lx[:] = (1.0 - par.alpha_M_dk_us_l) * (PM_dk_eu / PM_dk_l) ** (-par.eta_M_dk) * M_dk_lx
    inside_Y_lx = ((1.0 - par.beta_M_dk_l)**(1.0/par.eta_VA_dk) * NLL**rho
                   + par.beta_M_dk_l**(1.0/par.eta_VA_dk) * M_dk_lx**rho)
    YLL[:] = ZTH_LL * (inside_Y_lx ** (1.0 / rho))

@nb.njit
def prices(par, ini, ss,
           PT_eu_s, PT_us_s, E, E_us,
           PHH, PHL, PLH, PLL, PNT, WHH, WHL, WLH, WLL, WNT,
           PF_eu, PF_us, PF_TF,
           PTH, PTH_eu_s, PTH_us_s,
           PT, P, Q, Q_us,
           wHH, wHL, wLH, wLL, wNT, tau_x, tau_m,
           PF_us_s, PF_eu_s, PTH_eu_dom, PTH_us_dom, etaF):
    """
    Price indices and real exchange rates.
    PTH = single 4-sector CES aggregate (omega_TH weights), shared by all buyers.
    EU and US face the same PTH; sector allocation within their export demand
    uses destination-specific omega weights (omega_eu_i, omega_us_i) in the
    consumption block.
    """

    # a. foreign prices in DKK
    # Danish HH face EU tradable price (NT goods are not traded)
    PF_eu[:] = PT_eu_s * E
    PF_us[:] = (1+tau_m) * PT_us_s * E_us
    #PF_us[:] = PT_us_s * E_us

    # b. home tradeable price index (shared across domestic, EU and US buyers)
    PTH[:] = price_index_4(PHH, PHL, PLH, PLL, par.eta_TH,
                           par.omega_TH_HH, par.omega_TH_HL, par.omega_TH_LH, par.omega_TH_LL)

    # in foreign currency (for Armington) — both use same PTH
    PTH_eu_dom[:] = price_index_4(
        PHH, PHL, PLH, PLL, par.eta_TH,
        par.omega_TH_HH_eu,
        par.omega_TH_HL_eu,
        par.omega_TH_LH_eu,
        par.omega_TH_LL_eu
    )

    PTH_us_dom[:] = price_index_4(
        PHH, PHL, PLH, PLL, par.eta_TH,
        par.omega_TH_HH_us,
        par.omega_TH_HL_us,
        par.omega_TH_LH_us,
        par.omega_TH_LL_us
    )

    PTH_eu_s[:] = PTH_eu_dom / E
    #PTH_us_s[:] = (1.0 + tau_x) * PTH_us_dom / E_us
    tau_x_LH = tau_x * (1.0 - par.tau_x_LH_exempt)
    PTH_us_s[:] = price_index_4(
        (1.0 + tau_x)    * PHH,
        (1.0 + tau_x)    * PHL,
        (1.0 + tau_x_LH) * PLH,
        (1.0 + tau_x)    * PLL,
        par.eta_TH,
        par.omega_TH_HH_us, par.omega_TH_HL_us,
        par.omega_TH_LH_us, par.omega_TH_LL_us,
    ) / E_us    

    # c. foreign tradeable bundle (EU vs US)
    PF_TF[:] = price_index(PF_us, PF_eu, par.etaF_us, par.alpha_us)

    # d. tradeable and CPI indices
    PT[:] = price_index_t(PF_TF, PTH, etaF, par.alphaF)
    P[:]  = price_index(PT,    PNT,  par.etaT,  par.alphaT)

    # e. real exchange rates
    Q[:]    = PF_eu_s*E / P
    Q_us[:] = PF_us_s*E_us / P

    # f. real wages
    wHH[:] = WHH / P
    wHL[:] = WHL / P
    wLH[:] = WLH / P
    wLL[:] = WLL / P
    wNT[:] = WNT / P

@nb.njit
def inflation(par, ini, ss,
              PF_eu_s, PF_us_s, PF_eu, PF_us, PF_TF,
              PNT, PHH, PHL, PLH, PLL, PTH, PT, P,
              PTH_eu_s, PTH_us_s,
              pi_F_eu_s, pi_F_us_s, pi_F_eu, pi_F_us, pi_FF,
              pi_NT, pi_PHH, pi_PHL, pi_PLH, pi_PLL, pi_TH, pi_T, pi,
              pi_TH_eu_s, pi_TH_us_s):

    pi_F_eu_s[:] = inflation_from_price(PF_eu_s, ini.PF_eu_s)
    pi_F_us_s[:] = inflation_from_price(PF_us_s, ini.PF_us_s)

    pi_F_eu[:]   = inflation_from_price(PF_eu,   ini.PF_eu)
    pi_F_us[:]   = inflation_from_price(PF_us,   ini.PF_us)

    pi_FF[:]     = inflation_from_price(PF_TF,   ini.PF_TF)

    pi_NT[:]     = inflation_from_price(PNT,     ini.PNT)
    pi_PHH[:]    = inflation_from_price(PHH,     ini.PHH)
    pi_PHL[:]    = inflation_from_price(PHL,     ini.PHL)
    pi_PLH[:]    = inflation_from_price(PLH,     ini.PLH)
    pi_PLL[:]    = inflation_from_price(PLL,     ini.PLL)
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

# --- Original government block (no revenue smoothing fund) ---
@nb.njit
def government(par, ini, ss,
               PNT, P, wHH, NHH, wHL, NHL, wLH, NLH, wLL, NLL, wNT, NNT,
               ra, G, B, tau,
               inc_HH, inc_HL, inc_LH, inc_LL, inc_NT,
               M_dk_us_h, M_dk_us_hx, M_dk_us_l, M_dk_us_lx,
               PM_dk_us, tau_m, PF_us, CTF_us):
    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL
    for t in range(par.T):
        tax_base = (wHH[t]*NHH[t] + wHL[t]*NHL[t]
                    + wLH[t]*NLH[t] + wLL[t]*NLL[t] + wNT[t]*NNT[t])
        B_lag = prev(B, t, ini.B)
        M_us_total = M_dk_us_h[t] + M_dk_us_hx[t] + M_dk_us_l[t] + M_dk_us_lx[t]
        revenue = tau_m[t]/(1+tau_m[t]) * PM_dk_us[t] / P[t] * M_us_total + tau_m[t]/(1+tau_m[t]) * PF_us[t] / P[t] *CTF_us[t]
        B[t] = ss.B + par.phi_B*((B_lag - ss.B) - revenue)
        tau[t] = ((1.0 + ra[t])*B_lag + PNT[t]/P[t]*G[t] - revenue - B[t]) / tax_base
        inc_HH[:] = (1-tau)*wHH*NHH
        inc_HL[:] = (1-tau)*wHL*NHL
        inc_LH[:] = (1-tau)*wLH*NLH
        inc_LL[:] = (1-tau)*wLL*NLL
        inc_NT[:] = (1-tau)*wNT*NNT

@nb.njit
def NKWCs(par, ini, ss,
        beta, piWHH, piWHL, piWLH, piWLL, piWNT,
        NHH, NHL, NLH, NLL, NNT,
        wHH, wHL, wLH, wLL, wNT, tau,
        UC_HH_hh, UC_HL_hh, UC_LH_hh, UC_LL_hh, UC_NT_hh,
        NKWCHH_res, NKWCHL_res, NKWCLH_res, NKWCLL_res, NKWCNT_res):

    sNT = 1.0 - par.sHH - par.sHL - par.sLH - par.sLL

    piWHH_plus = lead(piWHH, ss.piWHH)
    NKWCHH_res[:] = piWHH - (par.kappa*(par.varphiHH*(NHH/par.sHH)**par.nu
                              - 1/par.muw*(1-tau)*wHH*UC_HH_hh) + beta*piWHH_plus)

    piWHL_plus = lead(piWHL, ss.piWHL)
    NKWCHL_res[:] = piWHL - (par.kappa*(par.varphiHL*(NHL/par.sHL)**par.nu
                              - 1/par.muw*(1-tau)*wHL*UC_HL_hh) + beta*piWHL_plus)

    piWLH_plus = lead(piWLH, ss.piWLH)
    NKWCLH_res[:] = piWLH - (par.kappa*(par.varphiLH*(NLH/par.sLH)**par.nu
                              - 1/par.muw*(1-tau)*wLH*UC_LH_hh) + beta*piWLH_plus)

    piWLL_plus = lead(piWLL, ss.piWLL)
    NKWCLL_res[:] = piWLL - (par.kappa*(par.varphiLL*(NLL/par.sLL)**par.nu
                              - 1/par.muw*(1-tau)*wLL*UC_LL_hh) + beta*piWLL_plus)

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
                C_hh, PT, PNT, P, PTH, PHH, PHL, PLH, PLL,
                PF_TF, PF_eu, PF_us,
                M_eu_s, M_us_s, PTH_eu_s, PTH_us_s, PF_eu_s, PF_us_s,
                CT, CNT, CTF, CTF_eu, CTF_us,
                CTH, CTH_HH, CTH_HL, CTH_LH, CTH_LL,
                CTH_eu_s, CTH_us_s,
                CTH_HH_eu_s, CTH_HL_eu_s, CTH_LH_eu_s, CTH_LL_eu_s,
                CTH_HH_us_s, CTH_HL_us_s, CTH_LH_us_s, CTH_LL_us_s,
                CTF_us_res, PTH_us_dom, PTH_eu_dom,
                etaF, eta_s,
                E_us, tau_x):
    """
    Consumption allocation with 4 home-tradeable sectors.

    All buyers (domestic, EU, US) consume the same PTH bundle with the same
    omega_TH sector weights.  Sectors differ in US export intensity only through
    the calibration of employment shares (sHH/sHL/sLH/sLL), not through
    destination-specific CES weights.

    tau_x (US tariff on DK exports) is embedded in PTH_us_s via the prices block,
    reducing total US demand for the DK bundle.
    """
    # a. T vs NT
    CT[:]  = par.alphaT   * (PT  / P)**(-par.etaT) * C_hh
    CNT[:] = (1-par.alphaT) * (PNT / P)**(-par.etaT) * C_hh

    # b. home bundle vs foreign bundle
    CTF[:] = par.alphaF      * (PF_TF / PT)**(- etaF) * CT
    CTH[:] = (1-par.alphaF)  * (PTH   / PT)**(- etaF) * CT

    # c. 4-sector split — same omega_TH weights for domestic and foreign buyers
    CTH_HH[:] = par.omega_TH_HH * (PHH / PTH)**(-par.eta_TH) * CTH
    CTH_HL[:] = par.omega_TH_HL * (PHL / PTH)**(-par.eta_TH) * CTH
    CTH_LH[:] = par.omega_TH_LH * (PLH / PTH)**(-par.eta_TH) * CTH
    CTH_LL[:] = par.omega_TH_LL * (PLL / PTH)**(-par.eta_TH) * CTH

    # d. EU vs US inside foreign bundle
    CTF_us_res[:] = CTF_us - par.alpha_us      * (PF_us / PF_TF)**(-par.etaF_us) * CTF
    CTF_eu[:] = (1-par.alpha_us)  * (PF_eu / PF_TF)**(-par.etaF_us) * CTF

    # e. total export demand from EU and US (Armington on aggregate PTH)
    #    tau_x already embedded in PTH_us_s (prices block)
    CTH_eu_s[:] = (PTH_eu_s / PF_eu_s)**(- eta_s) * M_eu_s
    CTH_us_s[:] = (PTH_us_s / PF_us_s)**(- eta_s) * M_us_s

    # f. sector-level export allocation — destination-specific omega_TH weights
    # (old version used same omega_TH weights for EU and US, neutralising H/L exposure)
    # CTH_HH_eu_s[:] = par.omega_TH_HH * (PHH / PTH)**(-par.eta_TH) * CTH_eu_s
    # CTH_HL_eu_s[:] = par.omega_TH_HL * (PHL / PTH)**(-par.eta_TH) * CTH_eu_s
    # CTH_LH_eu_s[:] = par.omega_TH_LH * (PLH / PTH)**(-par.eta_TH) * CTH_eu_s
    # CTH_LL_eu_s[:] = par.omega_TH_LL * (PLL / PTH)**(-par.eta_TH) * CTH_eu_s
    # CTH_HH_us_s[:] = par.omega_TH_HH * (PHH / PTH)**(-par.eta_TH) * CTH_us_s
    # CTH_HL_us_s[:] = par.omega_TH_HL * (PHL / PTH)**(-par.eta_TH) * CTH_us_s
    # CTH_LH_us_s[:] = par.omega_TH_LH * (PLH / PTH)**(-par.eta_TH) * CTH_us_s
    # CTH_LL_us_s[:] = par.omega_TH_LL * (PLL / PTH)**(-par.eta_TH) * CTH_us_s

    CTH_HH_eu_s[:] = par.omega_TH_HH_eu * (PHH / PTH_eu_dom)**(-par.eta_TH) * CTH_eu_s
    CTH_HL_eu_s[:] = par.omega_TH_HL_eu * (PHL / PTH_eu_dom)**(-par.eta_TH) * CTH_eu_s
    CTH_LH_eu_s[:] = par.omega_TH_LH_eu * (PLH / PTH_eu_dom)**(-par.eta_TH) * CTH_eu_s
    CTH_LL_eu_s[:] = par.omega_TH_LL_eu * (PLL / PTH_eu_dom)**(-par.eta_TH) * CTH_eu_s

    #CTH_HH_us_s[:] = par.omega_TH_HH_us * (PHH / PTH_us_dom)**(-par.eta_TH) * CTH_us_s
    #CTH_HL_us_s[:] = par.omega_TH_HL_us * (PHL / PTH_us_dom)**(-par.eta_TH) * CTH_us_s
    #CTH_LH_us_s[:] = par.omega_TH_LH_us * (PLH / PTH_us_dom)**(-par.eta_TH) * CTH_us_s
    #CTH_LL_us_s[:] = par.omega_TH_LL_us * (PLL / PTH_us_dom)**(-par.eta_TH) * CTH_us_s

    PTH_us_dom_tariff = PTH_us_s * E_us  # tariff-inclusive bundle in DKK
    tau_x_LH = tau_x * (1.0 - par.tau_x_LH_exempt)
    CTH_HH_us_s[:] = par.omega_TH_HH_us * ((1.0 + tau_x)    * PHH / PTH_us_dom_tariff)**(-par.eta_TH) * CTH_us_s
    CTH_HL_us_s[:] = par.omega_TH_HL_us * ((1.0 + tau_x)    * PHL / PTH_us_dom_tariff)**(-par.eta_TH) * CTH_us_s
    CTH_LH_us_s[:] = par.omega_TH_LH_us * ((1.0 + tau_x_LH) * PLH / PTH_us_dom_tariff)**(-par.eta_TH) * CTH_us_s
    CTH_LL_us_s[:] = par.omega_TH_LL_us * ((1.0 + tau_x)    * PLL / PTH_us_dom_tariff)**(-par.eta_TH) * CTH_us_s

@nb.njit
def market_clearing(par, ini, ss,
                    YHH, CTH_HH, CTH_HH_eu_s, CTH_HH_us_s,
                    YHL, CTH_HL, CTH_HL_eu_s, CTH_HL_us_s,
                    YLH, CTH_LH, CTH_LH_eu_s, CTH_LH_us_s,
                    YLL, CTH_LL, CTH_LL_eu_s, CTH_LL_us_s,
                    YNT, CNT, G,
                    clearing_YHH, clearing_YHL, clearing_YLH, clearing_YLL, clearing_YNT):

    clearing_YHH[:] = YHH - CTH_HH - CTH_HH_eu_s - CTH_HH_us_s
    clearing_YHL[:] = YHL - CTH_HL - CTH_HL_eu_s - CTH_HL_us_s
    clearing_YLH[:] = YLH - CTH_LH - CTH_LH_eu_s - CTH_LH_us_s
    clearing_YLL[:] = YLL - CTH_LL - CTH_LL_eu_s - CTH_LL_us_s
    clearing_YNT[:] = YNT - CNT - G

@nb.njit
def accounting(par, ini, ss,
               PHH, YHH, PHL, YHL, PLH, YLH, PLL, YLL, PNT, YNT,
               P, C_hh, G, A_hh, B, ra,
               GDP, NX, CA, NFA, Walras,
               PM_dk_h,  M_dk_h,  PM_dk_l,  M_dk_l,
               M_dk_hx, M_dk_lx,
               tau_m, PM_dk_us,
               M_dk_us_h, M_dk_us_hx, M_dk_us_l, M_dk_us_lx,
               PF_us, CTF_us):

    M_us_total = M_dk_us_h + M_dk_us_hx + M_dk_us_l + M_dk_us_lx
    tariff_rev = tau_m/(1+tau_m) * PM_dk_us/P * M_us_total + tau_m/(1+tau_m) * PF_us/P * CTF_us

    # GDP = value added across all sectors
    GDP[:] = (PHH*YHH - PM_dk_h*M_dk_h
              + PHL*YHL - PM_dk_h*M_dk_hx
              + PLH*YLH - PM_dk_l*M_dk_l
              + PLL*YLL - PM_dk_l*M_dk_lx
              + PNT*YNT) / P + tariff_rev

    NX[:] = GDP - C_hh - PNT/P*G

    NFA[:] = A_hh - B

    NFA_lag = lag(ini.NFA, NFA)
    CA[:] = NX + ra*NFA_lag

    Walras[:] = (NFA - NFA_lag) - CA