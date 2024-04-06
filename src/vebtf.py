import numpy as np
from utils import sym_tridiagonal_inverse, denoise_wavelet_ti,trend_filter
from skimage.restoration import estimate_sigma
from scipy.linalg import solveh_banded
import timeit

class VEBTF:
    """
    Variational empirical Bayes trend filtering
    Parameters:
        lambda0: prior strength for the first group
        tol: tolerance for convergence
        maxiter: maximum number of iterations
        verbose: whether to print the ELBO at each iteration
        printevery: print the ELBO every printevery iterations
        sigma2: initial value for the noise variance
        fix_sigma2: whether to fix the noise variance
        prior: prior for the second group, either 'point_normal' or 'ash' or 'ash_update'
        point_mass_sd: standard deviation of the point mass for the prior
    """
    def __init__(self,lambda0=0,tol=1e-6,maxiter=1000,verbose=True,printevery=10,sigma2=None,fix_sigma2=False,prior='ash_update',point_mass_sd = 'auto',sparse_threshold=0,num_shift_wavelet=1,mode_wavelet='soft',method_wavelet='VisuShrink'):
        self.fix_sigma2 = fix_sigma2
        self.sigma2 = sigma2
        if self.sigma2 is None:
            self.fix_sigma2 = False

        self.tol = tol
        self.maxiter = maxiter
        self.verbose = verbose
        self.printevery = printevery
        self.lambda0 = lambda0
        self.prior = prior
        if self.prior == 'ash':
            self.est_sk2 = False
        else:
            self.est_sk2 = True
        self.point_mass_sd = point_mass_sd
        self.sparse_threshold=sparse_threshold

        self.num_shift_wavelet=num_shift_wavelet
        self.mode_wavelet=mode_wavelet
        self.method_wavelet=method_wavelet

        self.model_name = f'VEBTF-{prior}'

    def elbo(self):
        v_d,v_e = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        #vinvs_d, _ = sym_tridiagonal_inverse(self.precision_d*self.s2, self.precision_e*self.s2[1:])
        eloglik = -0.5*self.n*np.log(2*np.pi*self.sigma2) - 0.5*np.sum(np.log(self.s2)) - (np.sum(self.y**2/self.s2)-2*np.sum(self.y*self.mu/self.s2) + np.sum(self.mu**2/self.s2) + np.sum(v_d/self.s2))/2/self.sigma2
        b = self.get_diff()
        vec_part = -np.sum(b**2*self.w)/2/self.sigma2 - (np.sum(self.w*self.get_DVDt_diag(v_d,v_e)))/2/self.sigma2 - self.get_log_det_tri(self.precision_d,self.precision_e)/2
        alpha_part = np.sum(self.alpha * (np.log(self.pi.reshape(1,-1))-np.log(self.alpha)-np.log(2*self.sigma2*np.pi*self.sk2.reshape(1,-1))/2))
        return eloglik + vec_part + alpha_part + self.lambda0*np.log(self.pi[0])

    def fit(self, y, s2=None, mu_init='wavelet'):
        """
        Fit the model to the data
        
        Input:
            y: observed data
            s2: variance of the data
            mu_init: initial value for the trend, either 'wavelet' or 'const'
        """
        start_time = timeit.default_timer()
        self.init_fit(y, s2, mu_init)
        previous_elbo = -np.inf
        for i in range(self.maxiter):
            self._update_alpha()
            #print(f"after alpha, elbo = {self.elbo()}")
            self._update_V()
            #print(f"after V, elbo = {self.elbo()}")
            self._update_mu()
            #print(f"after mu, elbo = {self.elbo()}")
            self._update_pi()
            #print(f"after pi, elbo = {self.elbo()}")
            if not self.fix_sigma2:
                self._update_sigma2()
                #print(f"after sigma2, elbo = {self.elbo()}")
            if self.est_sk2:
                self._update_sk2()
            #print(f"after sk2, elbo = {self.elbo()}")
            elbo = self.elbo()
            self.elbo_trace[i] = elbo
            if self.verbose and i % self.printevery == 0:
                print(f'Iteration {i}: ELBO = {elbo}')
            if np.abs(elbo - previous_elbo)/self.n < self.tol:
                break
            previous_elbo = elbo
        self.elbo_trace = self.elbo_trace[:i+1]
        v_d,_ = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        self.v = v_d
        self.run_time = timeit.default_timer() - start_time


    def _update_alpha(self):
        v_d,v_e = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        alpha = np.log(self.pi.reshape(1,-1)) - np.log(self.sk2.reshape(1,-1))/2 - (self.get_diff()**2 + self.get_DVDt_diag(v_d,v_e)).reshape(-1,1)/2/self.sigma2/self.sk2.reshape(1,-1)
        alpha = np.exp(alpha - np.max(alpha,axis=1).reshape(-1,1))
        alpha = alpha / np.sum(alpha,axis=1).reshape(-1,1)
        alpha = np.maximum(alpha,1e-10)
        self.alpha = alpha
        self.w = np.sum(alpha/self.sk2.reshape(1,-1),axis=1)

    def _update_V(self):
        d,e = self.get_DTWD()
        self.precision_d = (d + 1/self.s2)/self.sigma2
        self.precision_e = e/self.sigma2

    def _update_mu(self):
        e_temp = np.append(self.precision_e,0)
        ab = np.concatenate((self.precision_d.reshape(1,-1), e_temp.reshape(1,-1)))
        self.mu = solveh_banded(ab,self.y/self.sigma2,lower=True)

    def _update_pi(self):
        self.pi = np.sum(self.alpha,axis=0)
        self.pi[0] += self.lambda0
        self.pi = self.pi / np.sum(self.pi)
        self.pi = np.maximum(self.pi,1e-20)

    def _update_sigma2(self):
        # vinvs_d, _ = sym_tridiagonal_inverse(self.precision_d*self.s2, self.precision_e*self.s2[1:])
        v_d,v_e = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        a = np.sum(self.y**2/self.s2)-2*np.sum(self.y*self.mu/self.s2) + np.sum(self.mu**2/self.s2) + np.sum(v_d/self.s2)
        b = self.get_diff()
        # v_d,v_e = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        b = np.sum(b**2*self.w) + (np.sum(self.w*self.get_DVDt_diag(v_d,v_e)))
        self.sigma2 = (a+b)/(2*self.n-1)


    def _update_sk2(self):
        v_d,v_e = sym_tridiagonal_inverse(self.precision_d, self.precision_e)
        sk2_new = np.sum(self.alpha*(self.get_diff()**2 + self.get_DVDt_diag(v_d,v_e)).reshape(-1,1),axis=0) / self.sigma2 / np.sum(self.alpha,axis=0)
        sk2_new[0] = self.sk2[0]
        self.sk2 = sk2_new
        self.w = np.sum(self.alpha/self.sk2.reshape(1,-1),axis=1)

    def init_fit(self, y, s2, mu_init):
        self.y = y
        self.n = y.shape[0]
        self.init_mu(mu_init)
        if s2 is None:
            s2 = np.ones(self.n)
        self.s2 = s2
        if self.sigma2 is None and not self.fix_sigma2:
            self.sigma2 = estimate_sigma(y)**2
        self.precision_d = s2*self.n
        self.precision_e = np.zeros(self.n-1)
        if self.prior == 'point_normal':
            self.choose_sk2_two_group()
        else:
            self.choose_sk2_ash()
        self.K = len(self.sk2)
        self.pi = np.zeros(self.K)
        self.pi[0] = 0.99
        self.pi[1:] = (1-self.pi[0])/(self.K-1)
        self.elbo_trace = np.zeros(self.maxiter)
        if self.point_mass_sd == 'auto':
            self.point_mass_sd = np.std(1/self.n)/2

    def init_mu(self,mu_init):
        if type(mu_init) == str:
            if mu_init == 'wavelet':
                mu_init = denoise_wavelet_ti(self.y,num_shifts=self.num_shift_wavelet,mode=self.mode_wavelet,method=self.method_wavelet)
                mu_init[np.abs(mu_init)<self.sparse_threshold] = 0
            elif mu_init == 'lasso':
                mu_init = trend_filter(self.y)
                mu_init[np.abs(mu_init)<self.sparse_threshold] = 0
            elif mu_init == 'const':
                mu_init = np.ones(self.n)*np.mean(self.y)
            else:
                raise ValueError('mu_init should be either "wavelet" or "const"')
            self.mu = mu_init
        else:
            if len(mu_init) != self.n:
                raise ValueError('mu_init should have the same length as y')
            self.mu = mu_init


    def get_diff(self):
        return self.mu[1:] - self.mu[:-1]
    
    def get_DTWD(self):
        d = np.zeros(self.n)
        d[0] = self.w[0]
        d[-1] = self.w[-1]
        d[1:-1] = self.w[1:] + self.w[:-1]
        # for i in range(1,self.n-1):
        #     d[i] = self.w[i] + self.w[i-1]
        e = -self.w
        return d,e
    
    def get_DVDt_diag(self,d,e):
        """
        Get the diagonal of the matrix D V D^T
        
        Input:
            d: diagonal elements of the matrix V
            e: super-diagonal elements of the matrix V
        Return:
            diagonal_DVDt: diagonal of the matrix D V D^T
        """
        diagonal_DVDt = np.zeros(self.n-1)
        diagonal_DVDt = d[:-1] + d[1:] - 2 * e
        return diagonal_DVDt
    
    def get_log_det_tri(self,d,e):
        """
        Get the log determinant of a tridiagonal matrix
        
        Input:
            d: diagonal elements of the matrix
            e: super-diagonal elements of the matrix
        Return:
            log_det: log determinant of the matrix
        """
        n = len(d)
        log_det = np.log(d[0])
        if n == 1:
            return log_det
        prev_det = 1
        curr_det = d[0]

        for i in range(1, n):
            temp_det = d[i] - e[i-1] ** 2 / curr_det
            log_det += np.log(abs(temp_det))
            prev_det, curr_det = curr_det, temp_det

        return log_det
    def choose_sk2_ash(self, grid_mult=np.sqrt(2)):
        z = self.y[1:] - self.y[:-1]  # Differences between successive elements
        sz = np.sqrt(self.s2[1:] + self.s2[:-1])  # sqrt of sum of v elements, excluding first and last
        smin = np.min(sz) / 10
        # Ensure non-negative argument for sqrt by using np.maximum with 0
        smax = 2 * np.sqrt(np.maximum(np.max(z**2 - sz**2), 0))
        # Use np.linspace to create a log-spaced vector between log(smin) and log(smax)
        if smax > 0:  # Ensure smax is positive
            sk = np.exp(np.arange(np.log(smin), np.log(smax), step=np.log(grid_mult)))
            if sk[0] > self.point_mass_sd:
                sk = np.insert(sk, 0, self.point_mass_sd)
        else:
            sk = np.array([self.point_mass_sd,np.std(z)])
        self.sk2 = sk**2

    def choose_sk2_two_group(self):
        z = self.y[1:] - self.y[:-1]
        self.sk2 = np.array([self.point_mass_sd**2,np.var(z)])

    



