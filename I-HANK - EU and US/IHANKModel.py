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
        self.inputs_hh = ['beta','ra','inc_TH','inc_NT'] # direct inputs
        self.inputs_hh_z = [] # transition matrix inputs
        self.outputs_hh = ['a','c','uc_TH','uc_NT','c_TH','c_NT'] # outputs
        self.intertemps_hh = ['vbeg_a'] # intertemporal variables

        # c. GE
        self.shocks = ['ZTH','ZNT', #domestic TFPs
                       'beta','G', #Domestic preference and fiscal shocks
                       'i_shock', #domestic monetary shock (keep at zero under peg)
                       'rn_eu','i_shock_eu', 'Z_eu', # EU natural-rate, monetary shocks and foreign TFP
                       'rn_us','i_shock_us', 'Z_us'] # US natural-rate, monetary shocks and foreign TFP
        self.unknowns = ['CB','NNT','NTH','piWTH','piWNT', #original # endogenous inputs
                         'x_eu', 'pi_eu', 'i_eu',#EU
                         'x_us', 'pi_us', 'i_us', 'CB_us'] #US 
        self.targets = ['NKWCT_res','NKWCNT_res','clearing_YTH','clearing_YNT',  # domestic wage NKPCs + market clearing #targets
                        'eu_IS_res', 'eu_NKPC_res', 'eu_TR_res', 'UIP_res', #EU NK residuals + peg conditions
                        'us_IS_res', 'us_NKPC_res', 'us_TR_res', 'UIP_res_us']
        
        # d. all variables
        self.blocks = [
            'blocks.eu_nk', # closed-economy EU NK block (triangular, no SOE feedback)
            'blocks.us_nk', # closed-economy US NK block (triangular, no SOE feedback)
            'blocks.mon_pol', # sets the peg E (fixed)
            'blocks.mon_pol_us',# sets the  E_us (float)
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
        par.Nfix = 2 # number of sectors sectors
        par.Nz = 7 # idiosyncratic productivity
        par.sT = 0.25 # share of workers in tradeable sector

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
 
        # e. foreign Economy
        par.share_X_us=0.5 #share of DK exports goint to US in SS
        # e1) EU economy
        par.beta_eu = par.beta      # can be set separately
        par.sigma_eu = 1.0          # IS slope (1/intertemporal elasticity)
        par.kappa_eu = 0.05         # NKPC slope
        par.phi_pi_eu = 1.5         # Taylor rule on inflation
        par.phi_x_eu = 0.0          # Taylor rule on output gap
        
        par.W_eu_ss=1.0             #EU nominal wage in EUR (numeraire)
        par.Y_eu_ss=1.0             #EU outputlevel normalization for reporting
        par.chi_M_eu=1.0            #sensitivity of EU market size to EU activity 

        par.rn_eu_ss=0.005
        par.i_eu_ss=par.rn_eu_ss #zero inflation in SS

        #EU demand for SOE exports (armington)
        par.eta_s = 2.0 # Armington elasticity of foreign demand
        par.M_eu_s_ss = np.nan # size of foreign market (determined in ss)

         # e2) US economy
        par.beta_us = par.beta      # can be set separately
        par.sigma_us = 1.0          # IS slope (1/intertemporal elasticity)
        par.kappa_us = 0.05         # NKPC slope
        par.phi_pi_us = 1.5         # Taylor rule on inflation
        par.phi_x_us = 0.0          # Taylor rule on output gap
        
        par.W_us_ss=1.0             #EU nominal wage in EUR (numeraire)
        par.Y_us_ss=1.0             #EU outputlevel normalization for reporting
        par.chi_M_us=1.0            #sensitivity of US market size to US activity 

        par.rn_us_ss=0.005
        par.i_us_ss=par.rn_us_ss #zero inflation in SS

        #US demand for SOE exports (armington)
        par.eta_us_s = 2.0 # Armington elasticity of foreign demand
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
        par.Na = 500 # number of grid points

        # h. shocks
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
        par.jump_rn_eu = 0.00
        par.rho_rn_eu = 0.00
        par.std_rn_eu = 0.00

        par.jump_i_shock_eu = 0.00
        par.rho_i_shock_eu = 0.00
        par.std_i_shock_eu = 0.00

        # US shocks
        par.jump_rn_us = 0.00
        par.rho_rn_us = 0.00
        par.std_rn_us = 0.00

        par.jump_i_shock_us = 0.00
        par.rho_i_shock_us = 0.00
        par.std_i_shock_us = 0.00

        # i. misc.
        par.T = 500 # length of path        
        
        par.max_iter_solve = 50_000 # maximum number of iterations when solving
        par.max_iter_simulate = 50_000 # maximum number of iterations when simulating
        par.max_iter_broyden = 100 # maximum number of iteration when solving eq. system
        
        par.tol_ss = 1e-12 # tolerance when finding steady state
        par.tol_solve = 1e-12 # tolerance when solving
        par.tol_simulate = 1e-12 # tolerance when simulating
        par.tol_broyden = 1e-10 # tolerance when solving eq. system

        par.py_hh = False # use python in household problem
        par.py_blocks = False # use python in blocks

    def allocate(self):
        """ allocate model """

        par = self.par
        self.allocate_GE()

    prepare_hh_ss = steady_state.prepare_hh_ss
    find_ss = steady_state.find_ss        