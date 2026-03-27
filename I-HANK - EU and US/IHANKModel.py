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
        self.inputs_hh = ['beta','ra','inc_TH','inc_NT'] # direct inputs # add more sectors here
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','uc_TH','uc_NT','c_TH','c_NT'] # outputs # add more sectors here
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZTH','ZNT', #domestic TFPs
                       'beta','G', #Domestic preference and fiscal shocks
                       'i_shock', #domestic monetary shock (keep at zero under peg)
                       'i_shock_eu', 'Z_eu', 'piM_eu_eu', # EU natural-rate, monetary shocks and foreign TFP
                       'i_shock_us', 'Z_us', 'piM_us_us', # US natural-rate, monetary shocks and foreign TFP
                       'tau_x',  # US initial tariff on DK+EA exports (raises price of DK/EA goods in US market)
                       'tau_m']  # DK+EA retaliatory tariff on US-origin materials (EU sets external trade policy for DK)
        self.unknowns = ['CB','NNT','NTH','piWTH','piWNT', 'CB_us', #original # endogenous inputs
                         'C_eu', 'N_eu', 'pi_eu', 'i_eu', 'mc_eu', #EU
                         'C_us', 'N_us', 'pi_us', 'i_us', 'mc_us'] #US 
        self.targets = ['NKWCT_res','NKWCNT_res','clearing_YTH','clearing_YNT',  # domestic wage NKPCs + market clearing #targets
                        'eu_Euler_res', 'eu_NKPC_res', 'eu_TR_res', 'eu_LS_res', 'eu_RC_res', 'UIP_res', #EU NK residuals + peg conditions
                        'us_Euler_res', 'us_NKPC_res', 'us_TR_res', 'us_LS_res', 'us_RC_res', 'UIP_res_us'] #US NK residuals
        
        # d. all variables
        self.blocks = [
            'blocks.mon_pol', # sets the exchange rates first
            'blocks.material_prices', # sets the foreign prices (PF_eu and PF_us)
            'blocks.eu_nk', # closed-economy EU NK block (triangular, no SOE feedback)
            'blocks.us_nk', # closed-economy US NK block (triangular, no SOE feedback)
            #'blocks.mon_pol_us',# sets the  E_us (float)
            'blocks.production', # add more sectors here
            'blocks.prices', # add more price indexes here
            'blocks.inflation', # add more inflation rates here
            'blocks.central_bank',
            'blocks.government',
            'hh', # add more sectors here
            'blocks.NKWCs',
            'blocks.UIP',
            'blocks.consumption', # add more goods here; choice after T / NT
            'blocks.market_clearing', # add more sectors here            
            'blocks.accounting',            
        ]        

        # e. functions
        self.solve_hh_backwards = household_problem.solve_hh_backwards
        
    def setup(self):
        """ set baseline parameters """ # calibrate to match Danish economy

        par = self.par

        # a. discrete states
        par.Nfix = 2 # number of sectors # change to 5
        par.Nz = 7 # idiosyncratic productivity
        par.sT = 0.25 # share of workers in tradeable sector - change!

        # b. preferences
        par.beta = 0.975 # discount factor
        par.sigma = 2.0 # inverse of intertemporal elasticity of substitution

        par.alphaT = np.nan # share of tradeable goods in home consumption (determined in ss)
        par.etaT = 2.0 # elasticity of substitution between tradeable and non-tradeable goods
        
        par.alphaF = 1/3 # share of foreign goods in home tradeable consumption
        par.alpha_us = 0.5 # share of US goods in home tradeable consumption

        par.etaF = 2.0 # elasticity of substitution between home and foreign tradeable goods
        par.etaF_us = 2.0 # elasticity of substitution between US and EU tradeable goods
          
        par.varphiTH = np.nan # disutility of labor in tradeable sector (determined in s)
        par.varphiNT = np.nan # disutility of labor in non-tradeable sector (determined in s)
        par.nu = 1.0 # Frisch elasticity of labor supply
              
        # c. income parameters
        par.rho_z = 0.95 # AR(1) parameter
        par.sigma_psi = 0.10 # std. of psi
        
        # d. price setting
        par.kappa = 0.1 # slope of wage Phillips curve
        par.muw = 1.2 # wage mark-up

        #DK production
        par.beta_M_dk = 0.10        # material share in outer CES (labor vs. materials)
        par.eta_VA_dk = 1.50        # elasticity in outer CES
        par.alpha_M_dk_us = 0.10    # US share in EU materials bundle
        par.eta_M_dk = 1.50         # elasticity EU vs US materials in inner CES       
 
        # e. foreign Economy
        
        par.share_X_us = 0.5 # share of DK exports goint to US in SS
        par.eta_s = 2.0 # Armington elasticity of foreign demand
        # e1) EU economy
        par.i_eu_ss = 0.005           #zero inflation in SS
        par.beta_eu = 1.0/(1.0+par.i_eu_ss)     # can be set separately  
        par.sigma_eu = par.sigma    # inverse of intertemporal elasticity of substitution
        par.nu_eu= par.nu           # Frisch elasticity of labor supply
        par.varphi_eu = np.nan      # disutility of labor (determined in ss)

        par.kappa_eu = 0.05         # NKPC slope
        par.phi_pi_eu = 1.5         # Taylor rule on inflation
        
        par.W_eu_ss=1.0             #EU nominal wage in EUR (numeraire)
        par.Y_eu_ss=1.0             #EU outputlevel normalization for reporting
        par.chi_M_eu=1.0            #sensitivity of EU market size to EU activity 

        # EU materials in production
        par.beta_M_eu = 0.10        # material share in outer CES (labor vs. materials)
        par.eta_VA_eu = 1.50        # elasticity in outer CES
        par.alpha_M_eu_us = 0.10    # US share in EU materials bundle
        par.eta_M_eu = 1.50         # elasticity EU vs US materials in inner CES

        #EU demand for SOE exports (armington)
        par.M_eu_s_ss = np.nan # size of foreign market (determined in ss)

         # e2) US economy
        par.i_us_ss=0.005           #zero inflation in SS
        par.beta_us = 1.0/(1.0+par.i_us_ss)     # can be set separately  
        par.sigma_us = par.sigma    # inverse of intertemporal elasticity of substitution
        par.nu_us= par.nu           # Frisch elasticity of labor supply
        par.varphi_us = np.nan      # disutility of labor (determined in ss)

        par.kappa_us = 0.05         # NKPC slope
        par.phi_pi_us = 1.5         # Taylor rule on inflation
        
        par.W_us_ss=1.0             #US nominal wage in USD (numeraire)
        par.Y_us_ss=1.0             #US outputlevel normalization for reporting
        par.chi_M_us=1.0            #sensitivity of US market size to US activity

        # US materials in production
        par.beta_M_us = 0.10        # material share in outer CES (labor vs. materials)
        par.eta_VA_us = 1.50        # elasticity in outer CES
        par.alpha_M_us_us = 0.10    # US share in US materials bundle
        par.eta_M_us = 1.50         # elasticity US vs EU materials in inner CES 

        #US demand for SOE exports (armington)
        par.M_us_s_ss = np.nan # size of foreign market (determined in ss)

        # f. government
        par.tau_ss = 0.30 # tax rate on labor income
        par.omega = 0.10 # tax sensitivity to debt

        # central bank
        par.float = True # float or fix exchange rate
        par.phi = 1.5 # Taylor rule coefficient on inflation (only if float)

        # g. grids         
        par.a_min = 0.0 # maximum point in grid for a
        par.a_max = 50.0 # maximum point in grid for a
        par.Na = 500 # number of grid points #SÆT TIL 500 IGEN

        # h. shocks
        # Tariff parameters
        
        par.jump_tau_m = 0.0     # Jump size for tau_m shock
        par.rho_tau_m = 0.00     # Persistence of tau_m
        par.std_tau_m = 0.00    # std.
        
        par.jump_tau_x = 0.0     # Jump size for tau_m shock
        par.rho_tau_x = 0.00     # Persistence of tau_m
        par.std_tau_x = 0.00    # std.
        par.tariff_rev_lumpsum = False  # Revenue allocation mode

        par.jump_beta = 0.00 # initial jump
        par.rho_beta = 0.00 # AR(1) coefficeint
        par.std_beta = 0.00 # std.

        par.jump_G = 0.00 # initial jump
        par.rho_G = 0.00 # AR(1) coefficeint
        par.std_G = 0.00 # std.

        par.jump_i_shock = 0.00 # initial jump
        par.rho_i_shock = 0.00 # AR(1) coefficeint
        par.std_i_shock = 0.00 # std.

        # EU shocks
        par.jump_i_shock_eu = 0.00
        par.rho_i_shock_eu = 0.00
        par.std_i_shock_eu = 0.00

        # US shocks
        par.jump_i_shock_us = 0.00
        par.rho_i_shock_us = 0.00
        par.std_i_shock_us = 0.00

        # i. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state - set to -12 again
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system set to -10 again

        par.py_hh = False # use python in household problem
        par.py_blocks = False # use python in blocks

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        