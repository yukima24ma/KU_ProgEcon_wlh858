from Worker import WorkerClass
import numpy as np
from scipy.optimize import minimize

class GovernmentClass(WorkerClass):

    def __init__(self,par=None):

        # a. defaul setup
        self.setup_worker()
        self.setup_government()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

        # c. random number generator
        self.rng = np.random.default_rng(12345)

    def setup_government(self):

        par = self.par

        # a. workers
        par.N = 100  # number of workers
        par.sigma_p = 0.3  # std dev of productivity

        # b. pulic good
        par.chi = 50.0 # weight on public good in SWF
        par.eta = 0.1 # curvature of public good in SWF

    def draw_productivities(self):

        par = self.par

        #log pi ~ N(-0.5 * sigma_p^2, sigma_p^2)
        mu = -0.5 * par.sigma_p**2
        sigma = par.sigma_p
        
        log_p = self.rng.normal(loc=mu, scale=sigma, size=par.N)
        ps = np.exp(log_p)
        
        # store productivity
        par.ps = ps
        self.sol.ps = ps
        
        

    def solve_workers(self):

        par = self.par
        sol = self.sol

        ps = par.ps
        N = par.N
        
        ells = np.empty(N)
        cs = np.empty(N)
        Us = np.empty(N)
        
        for i, p in enumerate(ps):
            opt = self.optimal_choice(p) # from WorkerClass
            ells[i] = opt.ell
            cs[i] = opt.c
            Us[i] = opt.U
            
        sol.ells = ells
        sol.cs = cs
        sol.Us = Us
        
    def tax_revenue(self):

        par = self.par
        sol = self.sol

        ps = sol.ps
        ells = sol.ells
        
        taxes = np.empty(par.N)
        for i, (p, ell) in enumerate(zip(ps, ells)):
            pre_tax = self.income(p, ell)
            taxes[i] = self.tax(pre_tax)
        
        tax_revenue = np.sum(taxes) - par.N * par.b
        return tax_revenue
    
    def SWF(self):

        par = self.par
        sol = self.sol

        G =  self.tax_revenue()
        if G < 0:
            SWF = np.nan
        else:
            SWF = par.chi * (G ** par.eta) + np.sum(sol.Us)

        return SWF
    
    def optimal_taxes(self, x0=None):
        """
        Find approximately optimal (tau, zeta) by grid search.
        x0 is ignored (kept only for compatibility with earlier calls).
        """

        par = self.par
        sol = self.sol

        # 1. Draw productivities ONCE and fix them
        self.draw_productivities()

        # 2. Define grids for tau and zeta
        tau_grid  = np.linspace(0.0, 0.7, 71)    # 0.00, 0.01, ..., 0.70
        zeta_grid = np.linspace(-0.2, 0.2, 81)   # -0.20, ..., 0.20

        best_SWF  = -1e18
        best_tau  = None
        best_zeta = None

        # 3. Grid search
        for tau in tau_grid:
            for zeta in zeta_grid:

                par.tau  = tau
                par.zeta = zeta

                # solve workers for this (tau, zeta)
                self.solve_workers()

                # feasibility: tax revenue must be >= 0
                T_val = self.tax_revenue()
                if T_val < 0:
                    continue

                # compute SWF
                SWF_val = self.SWF()
                if np.isnan(SWF_val):
                    continue

                # update best
                if SWF_val > best_SWF:
                    best_SWF  = SWF_val
                    best_tau  = tau
                    best_zeta = zeta

        # 4. Store results
        sol.tau_star  = best_tau
        sol.zeta_star = best_zeta
        sol.SWF_star  = best_SWF

        return best_tau, best_zeta, best_SWF