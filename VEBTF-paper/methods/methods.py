
import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr
import timeit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

class mean_predictor:
    def __init__(self):
        self.model_name = "mean_predictor"

    def fit(self, y):
        """
        Apply the mean predictor to a given array y.

        Parameters:
        - y: NumPy array containing the input data.

        Returns:
        - fit_mean: NumPy array containing the mean of y.
        """
        start_time = timeit.default_timer()
        fit_mean = np.mean(y) * np.ones_like(y)
        self.mu = fit_mean
        self.run_time = timeit.default_timer() - start_time
        #return fit_mean

class wavelet_denoise:
    def __init__(self, wavelet='haar', mode='hard', wavelet_levels=None, method='VisuShrink',num_shifts=1,sigma=None):
        self.wavelet = wavelet
        self.mode = mode
        self.wavelet_levels = wavelet_levels
        self.method = method
        self.num_shifts = num_shifts
        self.sigma = sigma
        self.model_name = f"{wavelet}_{method}"

    def fit(self, y):
        """
        Apply translation-invariant wavelet denoising to an y.
        
        Parameters:
        - y: The input y to be denoised.
        - sigma, wavelet, mode, wavelet_levels, convert2ycbcr, method: Parameters for the denoise_wavelet function.
        - num_shifts: Number of shifts for cycle spinning.
        
        Returns:
        - Denoised y with translation-invariant wavelet denoising.
        """
        start_time = timeit.default_timer()
        # mirror the y
        n = len(y)
        y = np.concatenate([y, y[::-1]])
        denoised_y = np.zeros_like(y)
        self.num_shifts = min(self.num_shifts, y.shape[0] - 1)
        shifts = np.linspace(0, y.shape[0] - 1, num=self.num_shifts, dtype=int)
        if self.sigma is None:
            self.sigma = estimate_sigma(y)
        for shift in shifts:
            # Shift the y
            shifted_y = np.roll(y, shift, axis=0)

            # Denoise the shifted y
            denoised_shifted_y = denoise_wavelet(shifted_y, sigma=self.sigma, wavelet=self.wavelet,
                                                    mode=self.mode, wavelet_levels=self.wavelet_levels,method=self.method)

            # Shift back and accumulate the result
            denoised_y += np.roll(denoised_shifted_y, -shift, axis=0)

        # Average the accumulated denoised ys
        denoised_y /= self.num_shifts
        self.mu = denoised_y[0:n]
        self.run_time = timeit.default_timer() - start_time
        #return denoised_y[0:n]


class genlasso_tf:
    def __init__(self,ord=0):
        self.ord = ord
        self.model_name = f"genlasso_tf{ord}"

    def fit(self, y):
        """
        Apply trend filtering to a given array y with specified order ord.

        Parameters:
        - y: NumPy array containing the input data.
        - ord: Order of the trend filter (0 for piecewise constant, 1 for piecewise linear, etc.).

        Returns:
        - fit_tf: NumPy array containing the trend-filtered output.
        """
        start_time = timeit.default_timer()
        numpy2ri.activate()
        genlasso = importr('genlasso')
        y_r = ro.FloatVector(y)
        fit_tf_r = genlasso.trendfilter(y_r, ord=self.ord)
        fit_tf_cv_r = genlasso.cv_trendfilter(fit_tf_r, k=5, verbose=False)
        fit_tf_coef_r = genlasso.coef_genlasso(fit_tf_r, fit_tf_cv_r.rx2('lambda.1se')) 
        fit_tf = np.array(fit_tf_coef_r.rx2('beta'))
        self.mu=fit_tf.squeeze()
        self.run_time = timeit.default_timer() - start_time
        #return fit_tf.squeeze()

class GP_sklearn:
    def __init__(self, kernel='RBF'):
        self.kernel = kernel
        self.model_name = f"GP_{kernel}"

    def fit(self, y):
        """
        Apply Gaussian Process regression to a given array y with specified kernel.

        Parameters:
        - y: NumPy array containing the input data.
        - kernel: Kernel type for the Gaussian Process (default is 'RBF').

        Returns:
        - fit_gp: NumPy array containing the Gaussian Process regression output.
        """
        start_time = timeit.default_timer()
        n = len(y)
        X = np.arange(n).reshape(-1, 1)
        if self.kernel == 'RBF':
            kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1)
        else:
            raise ValueError("Unsupported kernel type.")
        
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(X, y)
        fit_gp = gp.predict(X)
        self.mu=fit_gp.squeeze()
        self.fitted_model = gp
        self.run_time = timeit.default_timer() - start_time

class btf:
    def __init__(self,ord=0,prior='DHS',verbose=False):
        self.ord = ord
        self.prior = prior
        self.verbose = verbose
        self.model_name = f"btf{ord}"

    def fit(self, y):
        """
        Apply Bayesian trend filtering to a given array y with specified order ord.

        Parameters:
        - y: NumPy array containing the input data.
        - ord: Order of the trend filter (0 for piecewise constant, 1 for piecewise linear, etc.).

        Returns:
        - fit_tf: NumPy array containing the trend-filtered output.
        """
        start_time = timeit.default_timer()
        numpy2ri.activate()
        dsp = importr('dsp')
        y_r = ro.FloatVector(y)
        fit_tf = dsp.btf(y_r, D=self.ord, evol_error=self.prior, verbose=self.verbose)
        mu_r = fit_tf.rx2('mu')
        mu_np = np.array(mu_r)
        self.mu=mu_np.mean(axis=0)
        self.run_time = timeit.default_timer() - start_time

class susie_tf:
    def __init__(self,L=10):
        self.L = L
        self.model_name = f"susie_tf{L}"

    def fit(self, y):
        """
        Apply trend filtering to a given array y with specified order ord.

        Parameters:
        - y: NumPy array containing the input data.
        - ord: Order of the trend filter (0 for piecewise constant, 1 for piecewise linear, etc.).

        Returns:
        - fit_tf: NumPy array containing the trend-filtered output.
        """
        start_time = timeit.default_timer()
        numpy2ri.activate()
        susieR = importr('susieR')
        y_r = ro.FloatVector(y)
        fit_tf_r = susieR.susie_trendfilter(y_r,L=self.L)
        fit_tf = np.array(susieR.predict_susie(fit_tf_r))
        self.mu = fit_tf.squeeze()
        self.run_time = timeit.default_timer() - start_time
        #return fit_tf.squeeze()

