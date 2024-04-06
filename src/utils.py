import numpy as np
from skimage.restoration import denoise_wavelet, estimate_sigma




def sym_tridiagonal_inverse(d, e):
    sd = np.sqrt(d)
    n = len(d)
    d = np.ones(n)
    e = e / sd[:-1] / sd[1:]
    log_theta = get_log_theta_cor(n, e)
    log_phi = get_log_phi_cor(n, e)
    inv_d = np.zeros(n)
    inv_e = np.zeros(n - 1)
    log_theta_last = log_theta[-1]
    for i in range(n):
        inv_d[i] = np.exp(log_theta[i] + log_phi[i + 1] - log_theta_last)
        if i < n - 1:
            inv_e[i] = (-1) ** (i + i + 1) * e[i] * np.exp(log_theta[i] + log_phi[i + 2] - log_theta_last)
    inv_d = inv_d / sd ** 2
    inv_e = inv_e / sd[:-1] / sd[1:]
    return inv_d, inv_e


def get_log_theta_cor(n, e):
    log_theta = np.zeros(n + 1)
    log_theta[0] = 0  # log(1) = 0
    log_theta[1] = 0  # log(1) = 0
    for i in range(2, n + 1):
        max_log_theta = max(log_theta[i - 1], log_theta[i - 2])
        log_theta[i] = max_log_theta + np.log(np.exp(log_theta[i - 1] - max_log_theta) - e[i - 2] ** 2 * np.exp(log_theta[i - 2] - max_log_theta))
    return log_theta

def get_log_phi_cor(n, e):
    log_phi = np.zeros(n + 1)
    log_phi[n] = 0  # log(1) = 0
    log_phi[n - 1] = 0  # log(1) = 0
    for i in range(n - 2, -1, -1):
        max_log_phi = max(log_phi[i + 1], log_phi[i + 2])
        log_phi[i] = max_log_phi + np.log(np.exp(log_phi[i + 1] - max_log_phi) - e[i] ** 2 * np.exp(log_phi[i + 2] - max_log_phi))
    return log_phi

def denoise_wavelet_ti(y, sigma=None, wavelet='haar', mode='soft',
                       wavelet_levels=None, method='VisuShrink', num_shifts=1):
    """
    Apply translation-invariant wavelet denoising to an y.
    
    Parameters:
    - y: The input y to be denoised.
    - sigma, wavelet, mode, wavelet_levels, convert2ycbcr, method: Parameters for the denoise_wavelet function.
    - num_shifts: Number of shifts for cycle spinning.
    
    Returns:
    - Denoised y with translation-invariant wavelet denoising.
    """
    # mirror the y
    n = len(y)
    y = np.concatenate([y, y[::-1]])
    denoised_y = np.zeros_like(y)
    num_shifts = min(num_shifts, y.shape[0] - 1)
    shifts = np.linspace(0, y.shape[0] - 1, num=num_shifts, dtype=int)
    if sigma is None:
        sigma = estimate_sigma(y)
    for shift in shifts:
        # Shift the y
        shifted_y = np.roll(y, shift, axis=0)

        # Denoise the shifted y
        denoised_shifted_y = denoise_wavelet(shifted_y, sigma=sigma, wavelet=wavelet,
                                                 mode=mode, wavelet_levels=wavelet_levels,method=method)

        # Shift back and accumulate the result
        denoised_y += np.roll(denoised_shifted_y, -shift, axis=0)

    # Average the accumulated denoised ys
    denoised_y /= num_shifts

    return denoised_y[0:n]


import numpy as np
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.packages import importr


def trend_filter(y, ord=0):
    """
    Apply trend filtering to a given array y with specified order ord.

    Parameters:
    - y: NumPy array containing the input data.
    - ord: Order of the trend filter (0 for piecewise constant, 1 for piecewise linear, etc.).

    Returns:
    - fit_tf: NumPy array containing the trend-filtered output.
    """
    numpy2ri.activate()
    genlasso = importr('genlasso')
    y_r = ro.FloatVector(y)
    fit_tf_r = genlasso.trendfilter(y_r, ord=ord)
    fit_tf_cv_r = genlasso.cv_trendfilter(fit_tf_r, k=5, verbose=False)
    fit_tf_coef_r = genlasso.coef_genlasso(fit_tf_r, fit_tf_cv_r.rx2('lambda.1se')) 
    fit_tf = np.array(fit_tf_coef_r.rx2('beta'))

    return fit_tf.squeeze()


def trend_filter_susie(y,L=20):
    """
    Apply trend filtering to a given array y with specified order ord.

    Parameters:
    - y: NumPy array containing the input data.
    - ord: Order of the trend filter (0 for piecewise constant, 1 for piecewise linear, etc.).

    Returns:
    - fit_tf: NumPy array containing the trend-filtered output.
    """
    numpy2ri.activate()
    susieR = importr('susieR')
    y_r = ro.FloatVector(y)
    fit_tf_r = susieR.susie_trendfilter(y_r,L=L)
    fit_tf = np.array(susieR.predict_susie(fit_tf_r))

    return fit_tf.squeeze()








# def sym_tridiagonal_inverse(d,e):
#     """
#     Inverse of a symmetric tridiagonal covariance matrix
#     Input:
#         d: diagonal elements of the covariance matrix
#         e: super-diagonal elements of the covariance matrix
#     Return:
#         inv_d: diagonal elements of the inverse of the covariance matrix
#         inv_e: super-diagonal elements of the inverse of the covariance matrix
#     """
#     sd = np.sqrt(d)
#     n = len(d)
#     d = np.ones(n)
#     e = e / sd[:-1]/ sd[1:]
#     theta = get_theta_cor(n,e)
#     phi = get_phi_cor(n,e)
#     inv_d = np.zeros(n)
#     inv_e = np.zeros(n-1)
#     theta_last = theta[-1]
#     # print(phi)
#     for i in range(n):
#         inv_d[i] = theta[i]*phi[i+1]/theta_last
#         if i < n-1:
#             inv_e[i] = (-1)**(i+i+1)*e[i]*theta[i]*phi[i+2]/theta_last
#     inv_d = inv_d / sd**2
#     inv_e = inv_e / sd[:-1] / sd[1:]
#     return inv_d, inv_e

# def get_theta_cor(n,e):
#     theta = np.zeros(n+1)
#     theta[0] = 1
#     theta[1] = 1
#     for i in range(2,n+1):
#         theta[i] = theta[i-1]-e[i-2]**2*theta[i-2]
#     return theta
# def get_phi_cor(n,e):
#     phi = np.zeros(n+1)
#     phi[n] = 1
#     phi[n-1] = 1
#     for i in range(n-2,-1,-1):
#         phi[i] = phi[i+1]-e[i]**2*phi[i+2]
#     return phi

# def get_theta(d,e):
#     n = len(d)
#     theta = np.zeros(n+1)
#     theta[0] = 1
#     theta[1] = d[0]
#     for i in range(2,n+1):
#         theta[i] = (d[i-1]*theta[i-1]-e[i-2]**2*theta[i-2])
#     return theta

# def get_phi(d,e):
#     n = len(d)
#     phi = np.zeros(n+1)
#     phi[n] = 1
#     phi[n-1] = d[-1]
#     for i in range(n-2,-1,-1):
#         phi[i] = d[i]*phi[i+1]-e[i]**2*phi[i+2]
#     return phi

