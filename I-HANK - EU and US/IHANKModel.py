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
        self.grids_hh = ['a']
        self.pols_hh = ['a']
        self.inputs_hh = ['beta','ra','inc_HH','inc_HL','inc_LH','inc_LL','inc_NT']
        self.inputs_hh_z = []
        self.outputs_hh = ['a','c','uc_HH','uc_HL','uc_LH','uc_LL','uc_NT',
                           'c_HH','c_HL','c_LH','c_LL','c_NT']
        self.intertemps_hh = ['vbeg_a']

        # c. GE
        self.shocks = ['ZTH_HH', 'ZTH_HL', 'ZTH_LH', 'ZTH_LL', 'ZNT',
                       'beta','G',
                       'i_shock',
                       'i_shock_eu', 'Z_eu', 'ZNT_eu',
                       'i_shock_us', 'Z_us', 'ZNT_us',
                       'tau_x',
                       'tau_m',
                       'etaF', 'eta_s']

        # Four tradeable sectors (HH, HL, LH, LL) + NT
        self.unknowns = ['CB','NNT','NHH','NHL','NLH','NLL',
                         'piWHH','piWHL','piWLH','piWLL','piWNT', 'CB_us',
                         'C_eu', 'N_eu', 'NNT_eu', 'pi_T_eu', 'pi_NT_eu', 'i_eu', 'mc_eu',
                         'C_us', 'N_us', 'NNT_us', 'pi_T_us', 'pi_NT_us', 'i_us', 'mc_us',
                         'CTF_us']

        self.targets = ['NKWCHH_res','NKWCHL_res','NKWCLH_res','NKWCLL_res','NKWCNT_res',
                        'clearing_YHH','clearing_YHL','clearing_YLH','clearing_YLL','clearing_YNT',
                        'eu_Euler_res','eu_NKPC_res','eu_NKPC_NT_res','eu_TR_res','eu_LS_res','eu_RC_res','eu_NT_res','UIP_res',
                        'us_Euler_res','us_NKPC_res','us_NKPC_NT_res','us_TR_res','us_LS_res','us_RC_res','us_NT_res','UIP_res_us',
                        'CTF_us_res']

        # d. block sequence
        self.blocks = [
            'blocks.mon_pol',
            'blocks.material_prices',
            'blocks.eu_nk',
            'blocks.us_nk',
            'blocks.production',
            'blocks.prices',
            'blocks.inflation',
            'blocks.central_bank',
            'blocks.government',
            'hh',
            'blocks.NKWCs',
            'blocks.UIP',
            'blocks.consumption',
            'blocks.market_clearing',
            'blocks.accounting',
        ]

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards

    def setup(self):
        """ set baseline parameters """

        par = self.par

        # a. discrete states
        par.Nfix = 5  # HH(0), HL(1), LH(2), LL(3), NT(4)
        par.Nz = 5

        # Employment shares
        # High-material sectors (HH+HL): total 38%, split evenly
        par.sHH = 0.37 #0.19  # high material, high US export share
        par.sHL = 0.01 #0.19  # high material, low US export share
        # Low-material sectors (LH+LL): total 19%, split evenly
        par.sLH = 0.13 #0.095 # low material, high US export share
        par.sLL = 0.06 #0.095 # low material, low US export share
        # sNT = 1 - sHH - sHL - sLH - sLL (derived)

        # b. preferences
        # HtM calibration (folded into evaluate_ss)
        par.HtM_target      = 0.15
        par.HtM_threshold   = 0.05         # fraction of avg labor income for "loose" HtM
        par.recalibrate_beta = True        # turn off to pin beta in sensitivity exercises
        par.beta_low        = 0.93         # search bracket
        par.beta_high       = 0.99
        par.beta = 0.975 # Bliver sat automatisk til at match HtM target
        par.sigma = 1.5 # 2.0 #1.5 #Forsøg med etaT # 2.0

        par.alphaT = np.nan
        par.etaT = 0.5 #2.0 #0.5 VIRKER IKKE

        par.alphaF = 0.15 #1/3 VIRKER
        par.alpha_us = 0.20 #0.05 VIRKER

        par.etaF = 2.0 #2.0 #VIRKER
        par.etaF_us = 3.0 #VIRKER

        # Home-tradeable 4-sector CES weights (calibrated in SS, shared by all buyers)
        par.omega_TH_HH = np.nan
        par.omega_TH_HL = np.nan
        par.omega_TH_LH = np.nan
        par.omega_TH_LL = np.nan  # stored explicitly for symmetry
        par.eta_TH = 1.0 # 2.0 #VIRKER

        # Destination-specific sector CES weights (calibrated in SS from share_X_us_H/L)
        par.omega_TH_HH_eu = np.nan
        par.omega_TH_HL_eu = np.nan
        par.omega_TH_LH_eu = np.nan
        par.omega_TH_LL_eu = np.nan
        par.omega_TH_HH_us = np.nan
        par.omega_TH_HL_us = np.nan
        par.omega_TH_LH_us = np.nan
        par.omega_TH_LL_us = np.nan

        # Labor disutility (calibrated in SS)
        par.varphiHH = np.nan
        par.varphiHL = np.nan
        par.varphiLH = np.nan
        par.varphiLL = np.nan
        par.varphiNT = np.nan
        par.nu = 2.0 # 1.0 # 2.0 #forsøg med etaT

        # c. income parameters
        par.rho_z = 0.95
        par.sigma_psi = 0.10 #0.13 VIRKER IKKE

        # d. price setting
        par.kappa = 0.05 #0.05 # 0.10 #VIRKER
        par.muw = 1.2

        # Danish production — material parameters shared within H/L group
        par.sM_dk_h_target=1/3
        par.sM_dk_l_target=1/3

        par.beta_M_dk_h = np.nan       # material share, high-material sectors (HH, HL)
        par.beta_M_dk_l = np.nan       # material share, low-material sectors  (LH, LL)
        par.alpha_M_dk_us_h = 0.27  # US share in materials, high-material sectors
        par.alpha_M_dk_us_l = 0.17  # US share in materials, low-material sectors
        par.eta_VA_dk = 0.50 #VIRKER
        par.eta_M_dk = 0.50 #VIRKER

        # e. foreign economy — sector-specific US export shares
        par.share_X_us_H = 0.40   # HH and LH: high US export share #VIRKER
        par.share_X_us_L = 0.02   # HL and LL: low US export share #VIRKER
        par.eta_s = par.etaF  #VIRKER

        # EU economy
        par.i_eu_ss = 0.005
        par.beta_eu = 1.0/(1.0+par.i_eu_ss)
        par.sigma_eu = par.sigma
        par.nu_eu = par.nu
        par.varphi_eu = np.nan

        par.kappa_eu = 0.05
        par.phi_pi_eu = 1.5 #VIRKER

        par.W_eu_ss = 1.0
        par.Y_eu_ss = 1.0

        par.beta_M_eu = 0.72  # 1/3 VIRKER
        par.eta_VA_eu = par.eta_VA_dk #VIRKER
        par.alpha_M_eu_us = 0.03 #VIRKER
        par.eta_M_eu = par.eta_M_dk #VIRKER

        par.M_eu_s_ss = np.nan

        # EU non-tradable sector
        par.alphaT_eu = 0.70   # tradable share in EU consumption (free parameter)
        par.etaT_eu  = 0.50 #VIRKER #    # T vs NT substitution elasticity in EU

        # US economy
        par.i_us_ss = 0.005
        par.beta_us = 1.0/(1.0+par.i_us_ss)
        par.sigma_us = par.sigma
        par.nu_us = par.nu
        par.varphi_us = np.nan

        par.kappa_us = 0.05
        par.phi_pi_us = 1.5 #VIRKER

        par.W_us_ss = 1.0
        par.Y_us_ss = 1.0

        par.beta_M_us = 0.74 # par.beta_M_eu #VIRKER
        par.eta_VA_us = par.eta_VA_dk #VIRKER
        par.alpha_M_us_us = 0.97 #VIRKER
        par.eta_M_us = par.eta_M_dk #VIRKER

        par.M_us_s_ss = np.nan

        # US non-tradable sector (mirrors EU)
        par.alphaT_us = 0.70   # tradable share in US consumption
        par.etaT_us   = 0.50 #VIRKER #   # T vs NT substitution elasticity in US

        # f. government
        par.tau_ss = 0.30
        par.phi_B = 0.93

        # central bank
        par.float = False
        par.phi = 1.5
        par.psi_NFA = 0.0 #5e-2

        # g. grids
        par.a_min = 0.0
        par.a_max = 50.0
        par.Na = 300

        # h. shocks
        par.jump_tau_m = 0.0
        par.rho_tau_m = 0.00
        par.std_tau_m = 0.00

        par.jump_tau_x = 0.0
        par.rho_tau_x = 0.00
        par.std_tau_x = 0.00

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

        par.jump_ZNT_eu = 0.00
        par.rho_ZNT_eu = 0.00
        par.std_ZNT_eu = 0.00

        par.jump_i_shock_us = 0.00
        par.rho_i_shock_us = 0.00
        par.std_i_shock_us = 0.00

        par.jump_ZNT_us = 0.00
        par.rho_ZNT_us = 0.00
        par.std_ZNT_us = 0.00

        par.jump_etaF = 0.00
        par.rho_etaF  = 0.00
        par.std_etaF  = 0.00

        par.jump_eta_s = 0.00
        par.rho_eta_s  = 0.00
        par.std_eta_s  = 0.00

        # i. misc.
        par.T = 200

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