import numpy as np
from EconModel import EconModelClass
from GEModelTools import GEModelClass

import household_problem
import steady_state
import blocks

class IHANKModelClass(EconModelClass,GEModelClass):

    #########
    # setup #
    #########

    def settings(self):
        """ fundamental settings """

        # a. namespaces
        self.namespaces = ['par','ss','ini','path','sim']

        # b. household
        self.grids_hh = ['a'] # grids
        self.pols_hh = ['a'] # policy functions
        self.inputs_hh = ['beta','ra','inc_HH','inc_HL','inc_NT'] # direct inputs — one income stream per sector
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','uc_HH','uc_HL','uc_NT','c_HH','c_HL','c_NT'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZTH','ZNT', #domestic TFPs
                       'beta','G', #Domestic preference and fiscal shocks
                       'i_shock', #domestic monetary shock (keep at zero under peg)
                       'i_shock_eu', 'Z_eu', 'piM_eu_eu', # EU natural-rate, monetary shocks and foreign TFP
                       'i_shock_us', 'Z_us', 'piM_us_us', # US natural-rate, monetary shocks and foreign TFP
                       'tau_x',  # US tariff on DK+EA exports
                       'tau_m']  # DK+EA tariff on US-origin materials

        # Two tradeable sectors (HH=high-input, HL=low-input) + NT = 18 unknowns/targets
        self.unknowns = ['CB','NNT','NHH','NHL','piWHH','piWHL','piWNT', 'CB_us',
                         'C_eu', 'N_eu', 'pi_eu', 'i_eu', 'mc_eu',   # EU
                         'C_us', 'N_us', 'pi_us', 'i_us', 'mc_us']   # US

        self.targets = ['NKWCHH_res','NKWCHL_res','NKWCNT_res',
                        'clearing_YHH','clearing_YHL','clearing_YNT',
                        'eu_Euler_res', 'eu_NKPC_res', 'eu_TR_res', 'eu_LS_res', 'eu_RC_res', 'UIP_res',
                        'us_Euler_res', 'us_NKPC_res', 'us_TR_res', 'us_LS_res', 'us_RC_res', 'UIP_res_us']

        # d. block sequence (order preserved)
        self.blocks = [
            'blocks.mon_pol',
            'blocks.material_prices',
            'blocks.eu_nk',
            'blocks.us_nk',
            'blocks.production',   # computes PHH, PHL, YHH, YHL, YNT, sector material demands
            'blocks.prices',       # computes PTH (flat CES of PHH,PHL), PT, P, real wages
            'blocks.inflation',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.NKWCs',
            'blocks.UIP',
            'blocks.consumption',  # inner nest splits CTH into CTH_H, CTH_L
            'blocks.market_clearing',
            'blocks.accounting',
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. discrete states
        par.Nfix = 3  # TH-High (i=0), TH-Low (i=1), NT (i=2)
        par.Nz = 7    # idiosyncratic productivity

        # Employment shares (from Danish sectoral data, Table 1.9)
        # High-input tradeable (ih_eh + ih_el): 37%+1% = 38%
        # Low-input tradeable  (il_eh + il_el): 13%+6% = 19%
        # Non-tradeable: 1 - 0.38 - 0.19 = 0.43
        par.sHH = 0.38  # share of workers in high-input tradeable sector
        par.sHL = 0.19  # share of workers in low-input tradeable sector
        # par.sNT = 1 - par.sHH - par.sHL  (derived, not stored separately)

        # b. preferences
        par.beta = 0.975
        par.sigma = 2.0  # inverse IES

        par.alphaT = np.nan  # tradeable share in consumption (calibrated in SS)
        par.etaT = 2.0       # T vs NT substitution elasticity

        par.alphaF = 1/3     # foreign share inside tradeable bundle
        par.alpha_us = 0.5   # US share inside foreign tradeable bundle

        par.etaF = 2.0       # home vs foreign substitution elasticity
        par.etaF_us = 2.0    # EU vs US substitution elasticity

        # Inner nest: flat CES between TH-High and TH-Low (home goods only)
        par.omega_TH_H = np.nan  # share of TH-High in home tradeable bundle (calibrated in SS)
        par.eta_TH = 2.0         # elasticity between TH-High and TH-Low

        # Labor disutility (sector-specific, calibrated in SS)
        par.varphiHH = np.nan
        par.varphiHL = np.nan
        par.varphiNT = np.nan
        par.nu = 1.0  # Frisch elasticity of labor supply

        # c. income parameters
        par.rho_z = 0.95
        par.sigma_psi = 0.10

        # d. price setting
        par.kappa = 0.1   # slope of wage Phillips curve
        par.muw = 1.2     # wage mark-up

        # Danish production — sector-specific CES parameters (Table 1.3)
        par.beta_M_dk_h = 0.32      # material share, high-input sector (omega_H)
        par.beta_M_dk_l = 0.35      # material share, low-input sector  (omega_L)
        par.alpha_M_dk_us_h = 0.27  # US share in materials, high-input sector (alpha_H)
        par.alpha_M_dk_us_l = 0.17  # US share in materials, low-input sector  (alpha_L)
        par.eta_VA_dk = 1.50        # outer CES elasticity (labor vs materials), shared
        par.eta_M_dk = 1.50         # inner CES elasticity (EU vs US materials), shared

        # e. foreign economy
        par.share_X_us = 0.5  # share of DK exports going to US in SS
        par.eta_s = 2.0       # Armington elasticity of foreign demand

        # EU economy
        par.i_eu_ss = 0.005
        par.beta_eu = 1.0/(1.0+par.i_eu_ss)
        par.sigma_eu = par.sigma
        par.nu_eu = par.nu
        par.varphi_eu = np.nan

        par.kappa_eu = 0.05
        par.phi_pi_eu = 1.5

        par.W_eu_ss = 1.0
        par.Y_eu_ss = 1.0
        par.chi_M_eu = 1.0

        par.beta_M_eu = 0.10
        par.eta_VA_eu = 1.50
        par.alpha_M_eu_us = 0.10
        par.eta_M_eu = 1.50

        par.M_eu_s_ss = np.nan

        # US economy
        par.i_us_ss = 0.005
        par.beta_us = 1.0/(1.0+par.i_us_ss)
        par.sigma_us = par.sigma
        par.nu_us = par.nu
        par.varphi_us = np.nan

        par.kappa_us = 0.05
        par.phi_pi_us = 1.5

        par.W_us_ss = 1.0
        par.Y_us_ss = 1.0
        par.chi_M_us = 1.0

        par.beta_M_us = 0.10
        par.eta_VA_us = 1.50
        par.alpha_M_us_us = 0.10
        par.eta_M_us = 1.50

        par.M_us_s_ss = np.nan

        # f. government
        par.tau_ss = 0.30
        par.omega = 0.10

        # central bank
        par.float = False
        par.phi = 1.5

        # g. grids
        par.a_min = 0.0
        par.a_max = 50.0
        par.Na = 200

        # h. shocks
        par.jump_tau_m = 0.0
        par.rho_tau_m = 0.00
        par.std_tau_m = 0.00

        par.jump_tau_x = 0.0
        par.rho_tau_x = 0.00
        par.std_tau_x = 0.00
        par.tariff_rev_lumpsum = False

        par.jump_beta = 0.00
        par.rho_beta = 0.00
        par.std_beta = 0.00

        par.jump_G = 0.00
        par.rho_G = 0.00
        par.std_G = 0.00

        par.jump_i_shock = 0.00
        par.rho_i_shock = 0.00
        par.std_i_shock = 0.00

        par.jump_i_shock_eu = 0.00
        par.rho_i_shock_eu = 0.00
        par.std_i_shock_eu = 0.00

        par.jump_i_shock_us = 0.00
        par.rho_i_shock_us = 0.00
        par.std_i_shock_us = 0.00

        # i. misc.
        par.T = 100

        par.max_iter_solve = 50_000
        par.max_iter_simulate = 50_000
        par.max_iter_broyden = 100

        par.tol_ss = 1e-12
        par.tol_solve = 1e-12
        par.tol_simulate = 1e-12
        par.tol_broyden = 1e-10

        par.py_hh = False
        par.py_blocks = False

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss
