from types import SimpleNamespace

import numpy as np

from scipy.optimize import minimize_scalar
from scipy.optimize import root_scalar

class WorkerClass:

    def __init__(self,par=None):

        # a. setup
        self.setup_worker()

        # b. update parameters
        if not par is None: 
            for k,v in par.items():
                self.par.__dict__[k] = v

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax
        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate

    def utility(self,c,ell):

        par = self.par

        if c <= 0:
            return -1e10
        
        u = np.log(c) - par.nu*(ell**(1+par.epsilon))/(1+par.epsilon)
        
        return u
    
    def tax(self,pre_tax_income):

        par = self.par

        tax = par.tau * pre_tax_income + par.zeta
        
        if not np.isnan(par.kappa):
            extra_income = max(pre_tax_income - par.kappa, 0.0)
            tax += par.omega * extra_income

        return tax
    
    def income(self,p,ell):

        par = self.par

        return par.w * p * ell

    def post_tax_income(self,p,ell):

        pre_tax_income = self.income(p,ell)
        tax = self.tax(pre_tax_income)
        
        # added for UBI b post tax income 
        return pre_tax_income - tax + self.par.b

        return pre_tax_income - tax
    
    def max_post_tax_income(self,p):

        par = self.par
        return self.post_tax_income(p,par.ell_max)

    def value_of_choice(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        U = self.utility(c,ell)

        return U
    
    def get_min_ell(self,p):
    
        par = self.par

        min_ell = par.zeta/(par.w*p*(1-par.tau))

        return np.fmax(min_ell,0.0) + 1e-8
    
    def optimal_choice(self,p):

        par = self.par
        opt = SimpleNamespace()

        # a. objective function
        def obj(ell):
            return -self.value_of_choice(p,ell)

        # b. bounds and minimization
        ell_min = self.get_min_ell(p)
        bounds = (ell_min,par.ell_max)
        
        res = minimize_scalar(obj,bounds=bounds,method='bounded')

        # c. results
        opt.ell = res.x
        opt.U = -res.fun
        opt.c = self.post_tax_income(p,opt.ell)

        return opt
    
    def FOC(self,p,ell):

        par = self.par

        c = self.post_tax_income(p,ell)
        if c <= 0:
            return 1e10
        
        t_marg = par.tau
        pre_tax = self.income(p,ell)
        if (not np.isnan(par.kappa)) and (pre_tax > par.kappa):
            t_marg += par.omega
            
        marginal_net = (1.0 - t_marg) * par.w * p
        
        FOC = marginal_net / c - par.nu * ell**par.epsilon

        return FOC
    
    def optimal_choice_FOC(self,p):

        par = self.par
        opt = SimpleNamespace()

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max
        
        def phi(ell):
            return self.FOC(p,ell)
        
        grid = np.linspace(ell_min, ell_max, 100)
        vals = np.array([phi(ell) for ell in grid])
        
        root_ell = None
        for a,b,fa,fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
            if np.sign(fa) == 0:
                root_ell = a
                break
            if np.sign(fa)*np.sign(fb) < 0:
                res = root_scalar(phi, bracket=(a,b))
                if res.converged:
                    root_ell = res.root
                    break
        
        if root_ell is None:
            opt_fallback = self.optimal_choice(p)
            root_ell = opt_fallback.ell
            
        opt.ell = root_ell
        opt.c = self.post_tax_income(p, root_ell)
        opt.U = self.utility(opt.c, root_ell)
        
        return opt
    
    
    def optimal_choice_top_FOC(self, p):
        """
        Four-step FOC approach for the kinked top tax (Question 3.1).
        Returns SimpleNamespace(ell, c, U, region).
        """

        from types import SimpleNamespace
        par = self.par
        opt = SimpleNamespace()

        ell_min = self.get_min_ell(p)
        ell_max = par.ell_max
        ell_kappa = par.kappa / (par.w * p)

        # Helper: utility at ell
        def U_at(ell):
            c = self.post_tax_income(p, ell)
            return self.utility(c, ell)

        # ---- Step 1: region below the kink ----
        Ub = -1e18
        ell_b = None
        if ell_min < ell_kappa:
            def phi_b(ell):
                c = self.post_tax_income(p, ell)
                if c <= 0:
                    return 1e10
                t_marg = par.tau
                marginal_net = (1 - t_marg) * par.w * p
                return marginal_net / c - par.nu * ell**par.epsilon

            grid = np.linspace(ell_min, min(ell_kappa, ell_max), 100)
            vals = np.array([phi_b(x) for x in grid])

            from scipy.optimize import root_scalar
            root_ell = None

            for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
                if np.sign(fa) == 0:
                    root_ell = a
                    break
                if np.sign(fa) * np.sign(fb) < 0:
                    res = root_scalar(phi_b, bracket=(a, b))
                    if res.converged:
                        root_ell = res.root
                        break

            if root_ell is not None and ell_min <= root_ell <= ell_kappa:
                ell_b = root_ell
                Ub = U_at(ell_b)

        # kink point
        Uk = -1e18
        ell_k = None
        if ell_min <= ell_kappa <= ell_max:
            ell_k = ell_kappa
            Uk = U_at(ell_k)

        # region above the kink
        Ua = -1e18
        ell_a = None
        if ell_kappa < ell_max:
            def phi_a(ell):
                c = self.post_tax_income(p, ell)
                if c <= 0:
                    return 1e10
                t_marg = par.tau + par.omega
                marginal_net = (1 - t_marg) * par.w * p
                return marginal_net / c - par.nu * ell**par.epsilon

            grid = np.linspace(max(ell_min, ell_kappa), ell_max, 100)
            vals = np.array([phi_a(x) for x in grid])
            root_ell = None

            for a, b, fa, fb in zip(grid[:-1], grid[1:], vals[:-1], vals[1:]):
                if np.sign(fa) == 0:
                    root_ell = a
                    break
                if np.sign(fa) * np.sign(fb) < 0:
                    res = root_scalar(phi_a, bracket=(a, b))
                    if res.converged:
                        root_ell = res.root
                        break

            if root_ell is not None and max(ell_min, ell_kappa) <= root_ell <= ell_max:
                ell_a = root_ell
                Ua = U_at(ell_a)

        # choose best
        candidates = [(Ub, ell_b, "below"), (Uk, ell_k, "kink"), (Ua, ell_a, "above")]
        best_U, best_ell, region = max(candidates, key=lambda x: x[0])

        # Fallback to numerical optimizer if needed
        if best_ell is None or best_U < -1e17:
            fallback = self.optimal_choice(p)
            best_ell = fallback.ell
            best_U = fallback.U
            region = "fallback"

        opt.ell = best_ell
        opt.c = self.post_tax_income(p, best_ell)
        opt.U = best_U
        opt.region = region
        return opt

    def setup_worker(self):

        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # a. preferences
        par.nu = 0.015 # weight on labor disutility
        par.epsilon = 1.0 # curvature of labor disutility
        
        # b. productivity and wages
        par.w = 1.0 # wage rate
        par.ps = np.linspace(0.5,3.0,100) # productivities
        par.ell_max = 16.0 # max labor supply
        
        # c. taxes
        par.tau = 0.50 # proportional tax rate
        par.zeta = 0.10 # lump-sum tax

        # NEW: universal basic income (UBI), default 0
        par.b = 0.0

        par.kappa = np.nan # income threshold for top tax
        par.omega = 0.20 # top rate rate
