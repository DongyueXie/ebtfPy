import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
current_dir = os.path.dirname(os.path.realpath(__file__))
myPy_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(os.path.join(myPy_dir, 'VEBTF', 'src'))


import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from util_func import cv_trendfilter_error
import timeit
import argparse
import pickle
import pandas as pd
from methods.methods import genlasso_tf, wavelet_denoise, susie_tf, btf, GP_sklearn, mean_predictor
from vebtf import VEBTF

def main(args):
    K = args.K
    file_name = args.file_name
    data_name = args.data_name
    n = args.n
    start_idx = args.start_idx

    # Generate the signal
    results = []
    # read in data
    data = pd.read_csv(f"~/VEBTF/VEBTF-paper/realdata/dataset/benchmark/{data_name}.csv")
    y = data['target'].values.squeeze()
    n = min(n, len(y))
    y = y[start_idx:(start_idx+n)]
    # if y is too large, scale y so the max of abs(y) is of order 100
    if np.mean(np.abs(y)) > 1e4:
        y = y/1e4 

    models = [
            VEBTF,
            GP_sklearn,
            genlasso_tf, 
            wavelet_denoise,
            susie_tf,
            mean_predictor
            ]
    for model in models:
        start_t = timeit.default_timer()
        model_name = model.__name__
        model_kwargs = get_model_kwargs(model,n)
        model_fit_kwargs = get_model_fit_kwargs(model)
        print(f"fitting {model_name} for data {data_name}")
        try:  
            rmse,mae,run_time = cv_trendfilter_error(y, K, model, model_kwargs,**model_fit_kwargs)
            results.append({
                'n': n,
                'data_name': data_name,
                'rmse': rmse,
                'mae': mae,
                'run_time': run_time,
                'model': model_name
            })
                
        except Exception as e:
            print(e)
            continue

        end_t = timeit.default_timer()
        print(f"Model {model_name} took {end_t-start_t} seconds")
        print(f"Model {model_name} RMSE: {rmse}, MAE: {mae} on data {data_name}")
        with open(f"realdata/results/real_{args.file_name}.pkl", "wb") as fp:
            pickle.dump(results, fp)
    return results

# def choose_ebtf_model(data_name,n):
#     if data_name == "ETTh1":
#         ebtf_model = VEBTF(sigma2=1,printevery=100,prior="ash_update",tol=1e-5,point_mass_sd=np.sqrt(1/n)/2,maxiter=1000,num_shift_wavelet = n)
#     elif data_name == "illness":
#         ebtf_model = VEBTF(sigma2=1,printevery=100,prior="ash_update",tol=1e-5,point_mass_sd=np.sqrt(1/n)/2,maxiter=1000,num_shift_wavelet = n)
#     elif data_name == "weather":
#         ebtf_model = VEBTF(sigma2=1,printevery=100,prior="ash_update",tol=1e-5,point_mass_sd=np.sqrt(1/n)/2,maxiter=1000,num_shift_wavelet = n)
#     else:
#         ebtf_model = VEBTF(sigma2=1,printevery=100,prior="ash_update",tol=1e-5,point_mass_sd=np.sqrt(1/n)/2,maxiter=1000,num_shift_wavelet = n)
    
#     return ebtf_model

def get_model_kwargs(model_class,n):
    if issubclass(model_class, genlasso_tf):
        return {'ord': 0}
    elif issubclass(model_class, wavelet_denoise):
        return {'method': 'VisuShrink',"num_shifts":n}
    elif issubclass(model_class, susie_tf):
        return {'L': 100}
    elif issubclass(model_class, VEBTF):
        return {'sigma2': 1, 'printevery': 100, 'prior': 'ash_update', 'tol': 1e-5, 'point_mass_sd': np.sqrt(1/n)/2, 'maxiter': 1000, 'num_shift_wavelet': n}
    elif issubclass(model_class, GP_sklearn):
        return {'kernel': 'RBF'}
    else:
        return {}

def get_model_fit_kwargs(model_class):
    if issubclass(model_class, VEBTF):
        return {'mu_init': 'wavelet'}
    else:
        return {}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run rea data holdout for VEBTF')
    parser.add_argument('--K', type=int, default=20, help='Number of CV folds')
    parser.add_argument('--n', type=int, default=1000, help='Number of samples')
    parser.add_argument('--data_name', type=str, default='blocks', help='Name of the dataset')
    parser.add_argument('--file_name', type=str, default='blocks', help='Name of the file to be saved')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the data')
    args = parser.parse_args()
    results=main(args)
    results_df = pd.DataFrame(results)