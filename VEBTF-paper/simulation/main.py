import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.realpath(__file__))
myPy_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(os.path.join(myPy_dir, 'VEBTF', 'src'))


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from util_func import generate_signals
import timeit
import argparse
import pickle
import pandas as pd
from methods.methods import genlasso_tf, wavelet_denoise, susie_tf
from vebtf import VEBTF

def main(args):
    repetitions = args.repetitions
    sigma = args.sigma
    snr = args.snr
    n = args.n
    signal_name = args.signal_name
    if signal_name is "blocks":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n)/3,num_shift_wavelet=100,prior='ash_update') 
    elif signal_name is "step":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n)/3,num_shift_wavelet=100,prior='ash_update')
    elif signal_name is "bumps":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n)/2,num_shift_wavelet=100,prior='ash_update') 
    elif signal_name is "heavi":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n),num_shift_wavelet=1,prior='ash_update')
    elif signal_name is "linear":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n),num_shift_wavelet=1,prior='ash_update') 
    elif signal_name is "gauss":
        ebtf_model = VEBTF(verbose=False,tol=1e-5,point_mass_sd=np.sqrt(1/n),num_shift_wavelet=1,prior='ash_update',method_wavelet='BayesShrink') 
    models = [genlasso_tf(ord=0), 
              wavelet_denoise(method='VisuShrink'),
              wavelet_denoise(method='BayesShrink'),
              susie_tf(L=10),
              susie_tf(L=20),
              ebtf_model]
    metrics = [mean_squared_error, mean_absolute_error]
    # Generate the signal
    results = []
    # datax = []
    fitted_models = []
    seed = 0
    for rep in range(repetitions):
        print(f"running rep {rep} for signal {signal_name}")
        start_t = timeit.default_timer()
        mu = generate_signals(n=n, signal_name=signal_name, snr=snr, sigma=sigma)
        np.random.seed(seed)
        y = mu + sigma * np.random.randn(n)
        # datax.append({
        #     'n': n,
        #     'signal_name': signal_name,
        #     'snr': snr,
        #     'rep': rep,
        #     'mu': mu,
        #     'y': y
        # })
        for model in models:
            model_name = model.model_name
            try:  
                model.fit(y)
                fitted_models.append({
                    'n': n,
                    'signal_name': signal_name,
                    'snr': snr,
                    'rep': rep,
                    'seed': seed,
                    'true_mu': mu,
                    'model_name': model_name,
                    'fitted_model': model,
                
                })
                for metric in metrics:
                    score = metric(mu, model.mu)
                    results.append({
                        'n': n,
                        'signal_name': signal_name,
                        'snr': snr,
                        'rep': rep,
                        'seed': seed,
                        'metric': metric.__name__,
                        'score': score,
                        'run_time': model.run_time,
                        'model': model_name
                    })
            except Exception as e:
                print(e)
                continue

        end_t = timeit.default_timer()
        seed += 1
        print(f"Rep {rep} took {end_t-start_t} seconds")
        with open(f"simulation/results/simu_fitted_model_{args.file_name}.pkl", "wb") as fp:
            pickle.dump(fitted_models, fp)
        with open(f"simulation/results/simu_metric_{args.file_name}.pkl", "wb") as fp:
            pickle.dump(results, fp)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run simulation for VEBTF')
    parser.add_argument('--repetitions', type=int, default=20, help='Number of repetitions')
    parser.add_argument('--sigma', type=float, default=1, help='Standard deviation of the noise')
    parser.add_argument('--snr', type=float, default=3, help='Signal to noise ratio')
    parser.add_argument('--n', type=int, default=1024, help='Number of observations')
    parser.add_argument('--signal_name', type=str, default='blocks', help='Name of the signal')
    parser.add_argument('--file_name', type=str, default='blocks', help='Name of the file')
    args = parser.parse_args()
    results=main(args)
    results_df = pd.DataFrame(results)
    res = results_df.groupby(['model', 'metric','n','signal_name','snr']).mean().reset_index()
    print(res[res['metric']=='mean_squared_error'])